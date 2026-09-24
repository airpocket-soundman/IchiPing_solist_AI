"""Solist-AI 向け学習データ生成 (UNO Q + IchiPing データセット, 周波数ワープ augmentation 込み)。

特徴: sim/bench_v612.py と同一の D=167
  時間波形 baseline 減算 → 1024 点 FFT (hop 512, Hann) → |X| を 61 窓平均 → 20log10 →
  floor -80 dB → DC 除外 → bin 26..192 (406.25..3000 Hz) の 167 値。
  (実機 feature_schema_id: tdiff-rfft1024-h512-symhann-magmean-log20-floor80-bin26-192-f32-v1)

学習ソース
  unoq : D:/GitHub/IchiPing-UNO-Q/pc/captures/uno_q_train_20260912_session{1..8}*_wav
         (INMP441/MAX98357A, 48k→16k, PRBS seed 20260912 ⇒ IR warp 可)
  frdm : D:/GitHub/IchiPing/pc/captures/full_32_train_v21..v25 (FRDM 世代, 励振非再現 ⇒ feature warp)
評価ソース (学習に混ぜない)
  UNO Q eval gray/evening/survey/crowd (baseline = meta.json group=="baseline" の frame)
  FRDM full_32_eval_v1 (baseline = s00000)

augmentation
  cross-baseline : 各 frame を複数セッションの baseline で diff
  freq warp      : ε ~ U(-WARP_MAX, WARP_MAX) の warped copy を N_WARP 個 (UNO Q は IR warp)
  specaug (任意) : SpectralJitter σ0.6 dB + LevelJitter ±2 dB

出力
  sim/_cache/solist_ds_<variant>.npz                : 全学習/評価特徴 (float32, 未標準化) + メタ
  sim_export/solist_ds/train_<variant>_{14,32}cls_5k.csv : Solist-AI Sim 用 (標準化済, one-hot, ≤1M セル)
  sim_export/solist_ds/test_<set>_<variant>_{14,32}cls.csv
  sim_export/solist_ds/norm_<variant>.npz            : mu/sd (標準化統計)
  sim_export/solist_ds/MANIFEST_<variant>.json

usage: python sim/make_solist_dataset.py [--sources unoq,frdm] [--warp ir|feat|none]
                                         [--n-warp 2] [--warp-max 0.035] [--specaug] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from freq_warp import IRWarper, prbs16k, warp_db_spectrum, fit_warp  # noqa: E402

ROOT = HERE.parent
CACHE = HERE / "_cache"
OUT = ROOT / "sim_export" / "solist_ds"
UNOQ = Path(r"D:/GitHub/IchiPing-UNO-Q/pc/captures")
FRDM = Path(r"D:/GitHub/IchiPing/pc/captures")

NFFT, HOP = 1024, 512
BIN_HZ = 16_000 / NFFT
LO_HZ, HI_HZ = 400, 3000
_WIN = np.hanning(NFFT)
LO_BIN = max(0, int(round(LO_HZ / BIN_HZ)) - 1)          # 25 (0-index, DC 除外後)
HI_BIN = int(round(HI_HZ / BIN_HZ))                       # 192
D = HI_BIN - LO_BIN                                       # 167
MAX_CELLS = 1_000_000                                     # Solist-AI Sim 制約

SOURCES = {
    "unoq": {
        "domain": "unoq", "warp": "ir", "seed": 20260912,
        "runs": [UNOQ / f"uno_q_train_20260912_session{i}_wav" for i in range(1, 7)]
              + [UNOQ / f"uno_q_train_20260912_session{i}_loud_wav" for i in (7, 8)],
        "baselines": [0, 3, 5],       # cross-baseline に使う run index (session1/4/6)
    },
    "frdm": {
        "domain": "frdm", "warp": "feat", "seed": None,
        "runs": [FRDM / f"full_32_train_v{i}" for i in (21, 22, 23, 24, 25)],
        "baselines": [0, 2, 4],
    },
}
EVALS = {
    "unoq_gray":    (UNOQ / "uno_q_eval_20260912_gray_wav",    "unoq"),
    "unoq_evening": (UNOQ / "uno_q_eval_20260912_evening_wav", "unoq"),
    "unoq_survey":  (UNOQ / "uno_q_eval_20260912_survey_wav",  "unoq"),
    "unoq_crowd":   (UNOQ / "uno_q_eval_20260912_crowd_wav",   "unoq"),
    "frdm_eval_v1": (FRDM / "full_32_eval_v1",                 "frdm"),
}
CLASS_ORDER_14 = ("A1", "A2", "B1", "B2", "B3", "B4",
                  "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")


# ---------------------------------------------------------------- labels / io
def parse_state(name: str):
    if not (name.startswith("s") and len(name) == 6 and set(name[1:]) <= {"0", "1"}):
        return None
    return [int(c) for c in name[1:]]          # a,b,c,AB,BC (1=OPEN)


def class14(bits) -> int:
    a, b, c, AB, BC = bits
    if AB == 0:
        tag = "A1" if a == 0 else "A2"
    elif BC == 0:
        tag = {(0, 0): "B1", (1, 0): "B2", (0, 1): "B3", (1, 1): "B4"}[(a, b)]
    else:
        tag = "C" + str(1 + a + 2 * b + 4 * c)
    return CLASS_ORDER_14.index(tag)


def class32(bits) -> int:
    return sum(int(b) << k for k, b in enumerate(bits))


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def list_frames(run: Path, baseline_only: bool = False, exclude_baseline_group: bool = False):
    """[(wav_path, bits)] を返す。meta.json の group で baseline 群を選別できる。"""
    out = []
    for sd in sorted(run.iterdir()):
        bits = parse_state(sd.name)
        if bits is None:
            continue
        groups = {}
        meta = sd / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            for fr in m.get("frames", []):
                if "wav" in fr and "group" in fr:
                    groups[fr["wav"]] = fr["group"]
        for wav in sorted(sd.glob("frame_*.wav")):
            g = groups.get(wav.name)
            if baseline_only and g != "baseline":
                continue
            if exclude_baseline_group and g == "baseline":
                continue
            out.append((wav, bits))
    return out


def baseline_of(run: Path, group_only: bool) -> np.ndarray:
    frames = list_frames(run, baseline_only=group_only)
    frames = [f for f in frames if f[1] == [0, 0, 0, 0, 0]]
    if not frames:
        raise RuntimeError(f"no baseline frames in {run}")
    return np.mean(np.stack([load_wav(p)[:32_000] for p, _ in frames]), axis=0)


# ---------------------------------------------------------------- features
def diff_spec(audio: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """bench_v612.diff_spec と同一 (512 bin dB)。"""
    n = min(len(audio), len(baseline))
    d = audio[:n] - baseline[:n]
    seg = np.stack([d[s:s + NFFT] for s in range(0, n - NFFT + 1, HOP)]) * _WIN
    mag = np.abs(np.fft.rfft(seg, axis=1)).mean(0)[1:]
    return np.maximum(20 * np.log10(mag + 1e-9), -80.0).astype(np.float32)


def crop(X: np.ndarray) -> np.ndarray:
    return X[..., LO_BIN:HI_BIN]


def full_spec_db(audio: np.ndarray) -> np.ndarray:
    seg = np.stack([audio[s:s + NFFT] for s in range(0, len(audio) - NFFT + 1, HOP)]) * _WIN
    return (20 * np.log10(np.abs(np.fft.rfft(seg, axis=1)).mean(0)[1:] + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- build
def build_train(src: dict, warp_mode: str, n_warp: int, warp_max: float, seed: int, log):
    """1 ソースの学習特徴を生成。returns dict of arrays."""
    runs = src["runs"]
    bls = {bi: baseline_of(runs[bi], group_only=False) for bi in src["baselines"]}
    warper = IRWarper(prbs16k(src["seed"])) if (warp_mode == "ir" and src["seed"] is not None) else None
    eff_mode = "none" if warp_mode == "none" else ("ir" if warper is not None else "feat")
    rng = np.random.default_rng(seed)
    X, y14, y32, run_id, bl_id, eps_arr = [], [], [], [], [], []
    for ri, run in enumerate(runs):
        frames = list_frames(run)
        t0 = time.time()
        for wav, bits in frames:
            a = load_wav(wav)[:32_000]
            c14, c32 = class14(bits), class32(bits)
            variants = [(0.0, a)]
            if eff_mode != "none":
                for _ in range(n_warp):
                    e = float(rng.uniform(-warp_max, warp_max))
                    variants.append((e, warper.warp_audio(a, e) if eff_mode == "ir" else a))
            for e, aw in variants:
                for bi, bl in bls.items():
                    sp = diff_spec(aw, bl)
                    if eff_mode == "feat" and e != 0.0:
                        sp = warp_db_spectrum(sp, e)
                    X.append(crop(sp)); y14.append(c14); y32.append(c32)
                    run_id.append(ri); bl_id.append(bi); eps_arr.append(e)
        log(f"  {src['domain']} run{ri} {run.name}: {len(frames)} frames, warp={eff_mode}, "
            f"{time.time()-t0:.0f}s")
    return dict(X=np.stack(X).astype(np.float32), y14=np.array(y14), y32=np.array(y32),
                run_id=np.array(run_id), bl_id=np.array(bl_id), eps=np.array(eps_arr, dtype=np.float32),
                warp_mode=eff_mode)


def build_eval(name: str, run: Path, domain: str, log):
    group_only = domain == "unoq"
    bl = baseline_of(run, group_only=group_only)
    frames = list_frames(run, exclude_baseline_group=group_only)
    X = np.stack([crop(diff_spec(load_wav(p)[:32_000], bl)) for p, _ in frames]).astype(np.float32)
    y14 = np.array([class14(b) for _, b in frames]); y32 = np.array([class32(b) for _, b in frames])
    log(f"  eval {name}: {len(frames)} frames (baseline frames excluded={group_only})")
    return dict(X=X, y14=y14, y32=y32)


def stratified_subsample(y: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y); per = n_rows // len(classes)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]; rng.shuffle(ci); idx.extend(ci[:per])
    idx = np.array(sorted(idx))
    return idx


def write_csv(path: Path, X: np.ndarray, y: np.ndarray, C: int, mu, sd):
    Xs = (X - mu) / sd
    oh = np.zeros((len(y), C), dtype=np.float32); oh[np.arange(len(y)), y] = 1.0
    head = [f"f{i+1}" for i in range(X.shape[1])] + [f"c{j}" for j in range(C)]
    np.savetxt(path, np.hstack([Xs, oh]), delimiter=",", fmt="%.6f", header=",".join(head), comments="")
    return Xs.shape[0], Xs.shape[1] + C


# ---------------------------------------------------------------- selftest
def selftest():
    """(1) diff_spec が bench_v612 と一致 (2) IR warp の再構成 SNR と符号 (3) 夕方ドリフトの再現。"""
    print("[selftest]")
    # bench_v612 のキャッシュ (v612_full_32_train_v12__od_v12.npz: v12 を自己 baseline で diff) と照合
    ref_npz = CACHE / "v612_full_32_train_v12__od_v12.npz"
    if ref_npz.exists():
        z = np.load(ref_npz); v12 = FRDM / "full_32_train_v12"
        bl12 = baseline_of(v12, group_only=False)
        fr = list_frames(v12)
        # bench_v612.build_run は sorted(iterdir) の sXXXXX 順に frame を並べる (list_frames と同順)
        errs = [np.abs(crop(diff_spec(load_wav(fr[i][0])[:32_000], bl12)) - z["X"][i, LO_BIN:HI_BIN]).max()
                for i in (0, 1, 500, len(fr) - 1)]
        print(f"  diff_spec vs bench_v612 cache (v12 self-baseline, 4 frames): max|Δ|={max(errs):.2e}  "
              f"(cache rows={len(z['X'])}, frames={len(fr)})")
    else:
        print("  (bench_v612 cache not found; diff_spec is a verbatim copy)")
    run = SOURCES["unoq"]["runs"][0]
    bl = baseline_of(run, group_only=False)
    wav = list_frames(run)[120][0]; a = load_wav(wav)[:32_000]
    w = IRWarper(prbs16k(20260912))
    print(f"  IR warp reconstruction SNR = {w.reconstruction_snr_db(a):.1f} dB")
    S0 = full_spec_db(a); Sw = full_spec_db(w.warp_audio(a, +0.03))
    print(f"  fitted ε for IR warp(+3%) = {fit_warp(S0, Sw, LO_BIN, HI_BIN):+.4f} (expect +0.030)")
    # 夕方ドリフト: session6 (16:33 まで) baseline → evening (19:36) baseline
    s6 = baseline_of(SOURCES["unoq"]["runs"][5], group_only=False)
    ev = baseline_of(EVALS["unoq_evening"][0], group_only=True)
    sv = baseline_of(EVALS["unoq_survey"][0], group_only=True)
    e_ev = fit_warp(full_spec_db(s6), full_spec_db(ev), LO_BIN, HI_BIN)
    e_sv = fit_warp(full_spec_db(s6), full_spec_db(sv), LO_BIN, HI_BIN)
    print(f"  fitted drift session6→evening ε={e_ev:+.4f} (UNO Q report -0.0215), →survey ε={e_sv:+.4f} (-0.0105)")
    # stale-baseline 効果: evening s00000 vs session6 baseline の diff レベルを IR warp が再現するか
    ev_frames = [load_wav(p)[:32_000] for p, b in list_frames(EVALS["unoq_evening"][0], exclude_baseline_group=True) if b == [0,0,0,0,0]]
    real = np.mean([crop(diff_spec(f, s6)).mean() for f in ev_frames])
    s6_frames = [load_wav(p)[:32_000] for p, b in list_frames(run)[:0]]  # placeholder
    s6_frames = [load_wav(p)[:32_000] for p, b in list_frames(SOURCES["unoq"]["runs"][5]) if b == [0,0,0,0,0]][:10]
    same = np.mean([crop(diff_spec(f, s6)).mean() for f in s6_frames])
    sim = np.mean([crop(diff_spec(w.warp_audio(f, e_ev), s6)).mean() for f in s6_frames])
    print(f"  mean diff level (dB, 400-3000Hz) of all-closed vs session6 baseline: same-session={same:.1f}, "
          f"real evening={real:.1f}, session6 IR-warped by ε={e_ev:+.4f}: {sim:.1f}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="unoq")
    ap.add_argument("--warp", default="ir", choices=("ir", "feat", "none"))
    ap.add_argument("--n-warp", type=int, default=2)
    ap.add_argument("--warp-max", type=float, default=0.035)
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baselines", default="", help="'all' = 全 run を cross-baseline に使う (既定は SOURCES の3本)")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    srcs = args.sources.split(",")
    if args.baselines == "all":
        for s_ in srcs:
            SOURCES[s_]["baselines"] = list(range(len(SOURCES[s_]["runs"])))
    variant = f"{'+'.join(srcs)}_{args.warp}{args.n_warp if args.warp!='none' else ''}" + ("_sa" if args.specaug else "") + ("_blall" if args.baselines == "all" else "")
    CACHE.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    L = []
    def log(s): print(s, flush=True); L.append(s)
    log(f"[build] variant={variant} D={D} (bins {LO_BIN}..{HI_BIN-1})")
    parts = []
    for si, s in enumerate(srcs):
        p = build_train(SOURCES[s], args.warp, args.n_warp, args.warp_max, args.seed + si, log)
        p["domain"] = np.full(len(p["y14"]), si); parts.append(p)
    X = np.concatenate([p["X"] for p in parts]); y14 = np.concatenate([p["y14"] for p in parts])
    y32 = np.concatenate([p["y32"] for p in parts]); eps = np.concatenate([p["eps"] for p in parts])
    dom = np.concatenate([p["domain"] for p in parts]); run_id = np.concatenate([p["run_id"] for p in parts])
    bl_id = np.concatenate([p["bl_id"] for p in parts])
    if args.specaug:
        rng = np.random.default_rng(args.seed + 100)
        X = X + rng.normal(0, 0.6, X.shape).astype(np.float32) + rng.uniform(-2, 2, (len(X), 1)).astype(np.float32)
    log(f"  train X={X.shape}  warped rows={(eps!=0).sum()}  classes14={np.bincount(y14, minlength=14).tolist()}")
    evals = {n: build_eval(n, r, d, log) for n, (r, d) in EVALS.items() if r.exists()}
    mu = X.mean(0); sd = X.std(0) + 1e-6
    np.savez_compressed(CACHE / f"solist_ds_{variant}.npz", X=X, y14=y14, y32=y32, eps=eps, domain=dom,
                        run_id=run_id, bl_id=bl_id, mu=mu, sd=sd,
                        **{f"eval_{n}_{k}": v for n, e in evals.items() for k, v in e.items()})
    np.savez(OUT / f"norm_{variant}.npz", mu=mu, sd=sd)
    manifest = dict(variant=variant, sources=srcs, warp=args.warp, n_warp=args.n_warp, warp_max=args.warp_max,
                    specaug=args.specaug, seed=args.seed, D=D, bins=[LO_BIN + 1, HI_BIN], band_hz=[LO_HZ, HI_HZ],
                    feature_schema_id="tdiff-rfft1024-h512-symhann-magmean-log20-floor80-bin26-192-f32-v1",
                    train_rows=int(len(X)), train_runs={s: [str(r) for r in SOURCES[s]["runs"]] for s in srcs},
                    cross_baselines={s: SOURCES[s]["baselines"] for s in srcs},
                    evals={n: dict(path=str(EVALS[n][0]), rows=int(len(e["y14"]))) for n, e in evals.items()},
                    csv={})
    if not args.no_csv:
        for C, ycol in ((14, y14), (32, y32)):
            n_rows = MAX_CELLS // (D + C)
            idx = stratified_subsample(ycol, n_rows, args.seed)
            p = OUT / f"train_{variant}_{C}cls_5k.csv"
            shape = write_csv(p, X[idx], ycol[idx], C, mu, sd); manifest["csv"][p.name] = shape
            log(f"  {p.name}: {shape[0]} x {shape[1]} = {shape[0]*shape[1]} cells")
            for n, e in evals.items():
                p = OUT / f"test_{n}_{variant}_{C}cls.csv"
                shape = write_csv(p, e["X"], e["y14"] if C == 14 else e["y32"], C, mu, sd); manifest["csv"][p.name] = shape
    (OUT / f"MANIFEST_{variant}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / f"BUILD_{variant}.log").write_text("\n".join(L), encoding="utf-8")
    log(f"[done] {OUT}")


if __name__ == "__main__":
    main()
