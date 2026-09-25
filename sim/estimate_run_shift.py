"""各データセット(run)の温度由来の周波数シフト ε を s00000(全閉) スペクトルから見積もる。

背景: 空調で室温が変わると音速が変わり (c ∝ √T, 約 +0.17 %/°C @25°C)、模型内共鳴の
周波数が一様に (1+ε) 倍になる。同じ日でもエアコンのオン/オフで 10°C 以上違うことがあるため、
日付ではなく run ごと (さらに run 内の時刻ごと) にシフトを見積もる。

手法
  A. s00000 global shift: 各 run の s00000 log-PSD (nfft 8192, ≈1.95 Hz/bin) を ≈31 Hz 電力平滑し (励振リップル除去)、
     400–3000 Hz で「run j ≈ shift_db(run i, ε_ij)」の ε_ij を格子+放物線補間で推定。
     全ペアを最小二乗で ε_i − ε_j に分解 (家族内平均 = 0)。閉路残差で自己整合性を確認。
  B. peak tracking: 基準 run の顕著な共鳴ピークを各 run で追跡し、ピーク毎の比 f_run/f_ref − 1。
     一様伸縮 (温度) なら周波数に依らず一定 → 中央値・IQR・周波数への傾きを報告。
  C. run 内ドリフト: 各状態の平均 PSD を基準 run の同状態と比較して ε(state) を求め、
     step 開始時刻に対して並べる (s00000 は run 冒頭にしか無いため)。

符号: ε > 0 = 共鳴が高域へ = 暖かい。ΔT ≈ ε / 0.0017 °C (相対値)。
家族 (FRDM 世代 / UNO Q) はマイク・スピーカが違うので家族間の比較はしない。
"""
from __future__ import annotations

import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
CACHE = ROOT / "sim" / "_cache"
FRDM = Path(r"D:/GitHub/IchiPing/pc/captures")
UNOQ = Path(r"D:/GitHub/IchiPing-UNO-Q/pc/captures")

FS = 16_000
NFFT, HOP = 8192, 2048
BIN_HZ = FS / NFFT
LO_HZ, HI_HZ = 400.0, 3000.0
LO, HI = int(LO_HZ / BIN_HZ), int(HI_HZ / BIN_HZ)
GRID = np.linspace(-0.06, 0.06, 1201)       # 0.01 % 刻み
TREND_BINS = 101                            # ≈200 Hz 移動平均 (ピーク検出用トレンド除去)
SMOOTH_FIT = 16                             # ≈31 Hz 電力平滑 (shift 相関用)
SMOOTH_PEAK = 8                             # ≈16 Hz 電力平滑 (ピーク追跡用)
# 注: 励振 PRBS は全 frame・全 run で同一系列なので、その periodogram の細かいリップルが
# 固定周波数に乗る。平滑せずに相関を取ると常に ε=0 が最大になる (実測で確認)。
# UNO Q 既知値 (evening −2.15 %, survey −1.05 %, crowd −0.80 %) に対し、平滑 16 bin・
# トレンド除去なしで −1.76 / −0.88 / −0.70 % と順序・比率が一致することを確認して採用。
EPS_PER_DEGC = 0.0017

FAMILIES = {
    "frdm": [FRDM / f"full_32_train_v{i}" for i in (2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25)]
            + [FRDM / "full_32_train_v5_part2", FRDM / "full_32_eval_v1",
               FRDM / "eval_quiet", FRDM / "eval_noise_low", FRDM / "eval_noise_high"],
    "unoq": [UNOQ / f"uno_q_train_20260912_session{i}_wav" for i in range(1, 7)]
            + [UNOQ / f"uno_q_train_20260912_session{i}_loud_wav" for i in (7, 8)]
            + [UNOQ / f"uno_q_eval_20260912_{n}_wav" for n in ("gray", "evening", "survey", "crowd")],
}
REF = {"frdm": "full_32_train_v12", "unoq": "uno_q_train_20260912_session6_wav"}


def short(run: Path) -> str:
    n = run.name
    return (n.replace("full_32_train_", "").replace("uno_q_train_20260912_", "")
             .replace("uno_q_eval_20260912_", "eval_").replace("_wav", ""))


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


_WIN = np.hanning(NFFT)


def power(a: np.ndarray) -> np.ndarray:
    starts = range(0, len(a) - NFFT + 1, HOP)
    seg = np.stack([a[s:s + NFFT] for s in starts]) * _WIN
    return np.mean(np.abs(np.fft.rfft(seg, axis=1)) ** 2, axis=0)[1:]   # DC 除外


def state_dirs(run: Path):
    for d in sorted(run.iterdir()):
        n = d.name
        if d.is_dir() and n.startswith("s") and len(n) == 6 and set(n[1:]) <= {"0", "1"}:
            yield n, d


def frame_times(d: Path) -> dict[str, str]:
    """wav 名 → 時刻。UNO Q は frame 毎、FRDM は step 開始時刻のみ。"""
    m = d / "meta.json"
    if not m.exists():
        return {}
    meta = json.loads(m.read_text(encoding="utf-8"))
    if "frames" in meta:
        return {f["wav"]: f.get("time") for f in meta["frames"] if "wav" in f}
    t = meta.get("started_at")
    return {"*": t} if t else {}


def baseline_wavs(run: Path, d: Path) -> list[Path]:
    """UNO Q は group=="baseline" の frame、FRDM は s00000 全 frame。"""
    m = d / "meta.json"
    wavs = sorted(d.glob("frame_*.wav"))
    if m.exists():
        meta = json.loads(m.read_text(encoding="utf-8"))
        if "frames" in meta:
            base = {f["wav"] for f in meta["frames"] if f.get("group") == "baseline"}
            if base:
                wavs = [w for w in wavs if w.name in base]
    return wavs


def run_spectra(run: Path) -> dict:
    """{state: (mean log-PSD dB, time)}; s00000 は baseline 群のみ。キャッシュする。"""
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"shift_psd_{run.parent.name[:4]}_{run.name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["spec"].item()
    spec = {}
    for name, d in state_dirs(run):
        wavs = baseline_wavs(run, d) if name == "s00000" else sorted(d.glob("frame_*.wav"))
        if not wavs:
            continue
        P = np.mean([power(load_wav(w)) for w in wavs], axis=0)
        times = frame_times(d)
        ts = [times.get(w.name) or times.get("*") for w in wavs]
        ts = [t for t in ts if t]
        t = ts[len(ts) // 2] if ts else None
        spec[name] = (10 * np.log10(P + 1e-20), t, len(wavs))
    np.savez_compressed(cache, spec=np.array(spec, dtype=object))
    print(f"  cached {run.name}: {len(spec)} states", flush=True)
    return spec


def smooth(db: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return db
    P = 10 ** (db / 10)
    return 10 * np.log10(np.convolve(P, np.ones(w) / w, mode="same") + 1e-20)


def detrend(db: np.ndarray) -> np.ndarray:
    k = np.ones(TREND_BINS) / TREND_BINS
    pad = TREND_BINS // 2
    trend = np.convolve(np.pad(db, pad, mode="edge"), k, mode="valid")
    return db - trend


def shift_db(db: np.ndarray, eps: float) -> np.ndarray:
    bins = np.arange(1, db.shape[-1] + 1, dtype=np.float64)
    return np.interp(bins / (1.0 + eps), bins, db)


def fit_eps(ref: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    """tgt ≈ shift_db(ref, ε). 戻り値 (ε, 最大相関)。放物線補間でサブ格子。"""
    r, t = smooth(ref, SMOOTH_FIT), smooth(tgt, SMOOTH_FIT)
    tt = t[LO:HI] - t[LO:HI].mean()
    tn = np.linalg.norm(tt)
    cs = np.empty(len(GRID))
    for i, e in enumerate(GRID):
        w = shift_db(r, float(e))[LO:HI]
        w = w - w.mean()
        cs[i] = float(w @ tt / (np.linalg.norm(w) * tn + 1e-12))
    k = int(np.argmax(cs))
    e = GRID[k]
    if 0 < k < len(GRID) - 1:
        y0, y1, y2 = cs[k - 1], cs[k], cs[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            e += 0.5 * (y0 - y2) / den * (GRID[1] - GRID[0])
    return float(e), float(cs[k])


def pairwise_ls(names: list[str], base: dict[str, np.ndarray]):
    n = len(names)
    E = np.zeros((n, n)); C = np.ones((n, n))
    rows, rhs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            e, c = fit_eps(base[names[i]], base[names[j]])
            E[i, j], E[j, i], C[i, j], C[j, i] = e, -e, c, c
            r = np.zeros(n); r[j], r[i] = 1, -1          # ε_j − ε_i = e_ij
            rows.append(r); rhs.append(e)
    rows.append(np.ones(n)); rhs.append(0.0)               # 平均 0
    A, b = np.array(rows), np.array(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A[:-1] @ x - b[:-1]
    return x, E, C, float(np.sqrt(np.mean(resid ** 2)))


def find_peaks(db: np.ndarray, n_max: int = 25, min_prom_db: float = 3.0, min_sep: int = 8):
    d = detrend(smooth(db, SMOOTH_PEAK))
    cand = [k for k in range(LO + 2, HI - 2) if d[k] > d[k - 1] and d[k] >= d[k + 1]]
    cand = [k for k in cand if d[k] - min(d[max(LO, k - 15):k].min(), d[k + 1:k + 16].min()) >= min_prom_db]
    cand.sort(key=lambda k: -d[k])
    out = []
    for k in cand:
        if all(abs(k - o) >= min_sep for o in out):
            out.append(k)
        if len(out) >= n_max:
            break
    return sorted(out)


def refine(db: np.ndarray, k: int) -> float:
    y0, y1, y2 = db[k - 1], db[k], db[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return (k + 1 + off) * BIN_HZ           # bin index k ↔ (k+1)·BIN_HZ (DC 除外)


def track_peaks(ref: np.ndarray, tgt: np.ndarray, peaks: list[int], win: float = 0.05):
    ref, tgt = smooth(ref, SMOOTH_PEAK), smooth(tgt, SMOOTH_PEAK)
    d = detrend(tgt)
    res = []
    for k in peaks:
        f0 = refine(ref, k)
        lo = max(LO, int(k * (1 - win))); hi = min(HI, int(k * (1 + win)) + 1)
        j = lo + int(np.argmax(d[lo:hi]))
        if j in (lo, hi - 1):
            continue
        res.append((f0, refine(tgt, j) / f0 - 1.0))
    return res


def to_dt(t: str | None):
    return datetime.fromisoformat(t) if t else None


def main():
    report = {"method": __doc__, "eps_per_degC": EPS_PER_DEGC, "families": {}}
    for fam, runs in FAMILIES.items():
        runs = [r for r in runs if r.exists()]
        print(f"[{fam}] {len(runs)} runs", flush=True)
        specs = {short(r): run_spectra(r) for r in runs}
        names = [n for n in specs if "s00000" in specs[n]]
        base = {n: specs[n]["s00000"][0] for n in names}
        ref = short(Path(REF[fam]))

        # A. global shift (pairwise LS)
        x, E, C, rms = pairwise_ls(names, base)
        # B. peaks relative to ref
        peaks = find_peaks(base[ref])
        runs_out = {}
        for i, n in enumerate(names):
            pk = track_peaks(base[ref], base[n], peaks)
            ratios = np.array([p[1] for p in pk]) if pk else np.array([np.nan])
            fr = np.array([p[0] for p in pk]) if pk else np.array([np.nan])
            slope = float(np.polyfit(fr / 1000, ratios, 1)[0]) if len(pk) >= 4 else float("nan")
            e_ref, c_ref = fit_eps(base[ref], base[n])
            runs_out[n] = {
                "time": specs[n]["s00000"][1], "n_baseline": specs[n]["s00000"][2],
                "eps_ls": float(x[i]), "eps_vs_ref_global": e_ref, "corr_vs_ref": c_ref,
                "eps_vs_ref_peak_median": float(np.nanmedian(ratios)),
                "eps_vs_ref_peak_iqr": float(np.nanpercentile(ratios, 75) - np.nanpercentile(ratios, 25)),
                "peak_slope_per_kHz": slope, "n_peaks": len(pk),
            }
        # C. in-run drift: 各状態を ref の同状態と比較
        drift = {}
        for n in names:
            pts = []
            for st, (db, t, _) in sorted(specs[n].items(), key=lambda kv: kv[1][1] or ""):
                if st not in specs[ref]:
                    continue
                e, c = fit_eps(specs[ref][st][0], db)
                pts.append({"state": st, "time": t, "eps_vs_ref": e, "corr": c})
            drift[n] = pts
        report["families"][fam] = {
            "ref": ref, "ls_closure_rms": rms, "ref_peaks_hz": [round(refine(base[ref], k), 1) for k in peaks],
            "runs": runs_out, "pairwise_eps": {"names": names, "E": E.tolist(), "corr": C.tolist()},
            "drift": drift,
        }
        print(f"  closure rms {rms*100:.3f}%  peaks {len(peaks)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RUN_SHIFT.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(report)
    plot(report)


def pct(v: float) -> str:
    return "—" if v != v else f"{v*100:+.2f}%"


def drift_summary(pts: list[dict]):
    good = [p for p in pts if p["time"] and p["corr"] > 0.5]
    if len(good) < 3:
        return None
    t0 = to_dt(good[0]["time"])
    mins = np.array([(to_dt(p["time"]) - t0).total_seconds() / 60 for p in good])
    e = np.array([p["eps_vs_ref"] for p in good])
    return {"span_min": float(mins.max()), "eps_start": float(np.median(e[:4])),
            "eps_end": float(np.median(e[-4:])), "eps_range": float(np.percentile(e, 95) - np.percentile(e, 5)),
            "median_corr": float(np.median([p["corr"] for p in good]))}


def write_md(rep: dict):
    L = ["# Run別 周波数シフト見積もり (s00000 全閉スペクトル)", "",
         "温度ログが無いため、全閉 s00000 の共鳴位置から各 run の周波数スケール ε を推定した。",
         f"ε>0 = 共鳴が高域 = 暖かい。温度換算は ΔT ≈ ε / {EPS_PER_DEGC*100:.2f}%/°C (相対値)。",
         "手法: A=400–3000 Hz 31 Hz 平滑 log-PSD の shift 相関 (全ペア最小二乗, 家族平均=0),",
         "B=基準 run の共鳴ピーク追跡 (一様伸縮なら周波数に依らず一定), C=各状態を基準 run の同状態と比較した run 内ドリフト。", ""]
    for fam, F in rep["families"].items():
        L += [f"## {fam}  (基準 run = {F['ref']}, LS 閉路残差 RMS {F['ls_closure_rms']*100:.3f}%)", "",
              "| run | s00000時刻 | ε (LS,家族平均基準) | ≈ΔT | ε vs基準 (global) | ε vs基準 (peak中央値) | peak IQR | peak傾き /kHz | 相関 | run内ドリフト (開始→終了, 5–95%幅, 分) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for n, r in sorted(F["runs"].items(), key=lambda kv: kv[1]["time"] or ""):
            ds = drift_summary(F["drift"][n])
            dtxt = "—" if not ds else f"{pct(ds['eps_start'])}→{pct(ds['eps_end'])}, 幅{ds['eps_range']*100:.2f}%, {ds['span_min']:.0f}分"
            L.append(f"| {n} | {(r['time'] or '')[:16]} | {pct(r['eps_ls'])} | {r['eps_ls']/EPS_PER_DEGC:+.1f}°C | "
                     f"{pct(r['eps_vs_ref_global'])} | {pct(r['eps_vs_ref_peak_median'])} | {r['eps_vs_ref_peak_iqr']*100:.2f}% | "
                     f"{r['peak_slope_per_kHz']*100:+.2f}% | {r['corr_vs_ref']:.2f} | {dtxt} |")
        L += ["", f"基準 run の追跡ピーク (Hz): {', '.join(str(p) for p in F['ref_peaks_hz'])}", ""]
    L += ["## 読み方と限界", "",
          "- 採用値は A (ε LS / ε vs基準 global)。UNO Q 既知値 evening −2.15 / survey −1.05 / crowd −0.80 % に対し"
          " −1.76 / −0.87 / −0.70 % で順序・比率が一致 (大きさは約 0.8 倍。基準 run・FFT 長の違い)。",
          "- B (peak 追跡) は IQR が 1.5–4 % と大きく中央値が 0 付近に張り付くため参考外。平滑後も一部ピークが"
          "励振由来の固定リップルに捕まっている。",
          "- C (run 内ドリフト) は全 run が同じ状態順序で収録されているため、『全 run に共通する時間位置依存のドリフト』"
          "と『基準 run 自身のドリフト』を分離できない (基準を session6→session1 に替えると UNO Q 終盤の跳ね上がりが消える)。"
          "run 間の相対的な形として読む。絶対的な run 内ドリフトには run 末尾に s00000 を再測定するのが必要。",
          "- 温度換算は音速 (c ∝ √T) のみを仮定した目安。湿度・模型材の熱膨張は無視。", ""]
    (OUT / "RUN_SHIFT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(rep: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(len(rep["families"]), 1, figsize=(11, 4.2 * len(rep["families"])))
    for ax, (fam, F) in zip(np.atleast_1d(axes), rep["families"].items()):
        for n, pts in F["drift"].items():
            good = [p for p in pts if p["time"] and p["corr"] > 0.5]
            if not good:
                continue
            t = [to_dt(p["time"]) for p in good]
            if fam != "unoq":                    # 複数日に跨るので run 開始からの分
                t = [(x - t[0]).total_seconds() / 60 for x in t]
            ax.plot(t, [p["eps_vs_ref"] * 100 for p in good], ".-", ms=3, lw=0.8, label=n)
        ax.set_xlabel("clock time" if fam == "unoq" else "minutes from run start")
        ax.set_title(f"{fam}: per-state eps vs same state of {F['ref']}")
        ax.set_ylabel("ε [%]"); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "RUN_SHIFT_drift.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
