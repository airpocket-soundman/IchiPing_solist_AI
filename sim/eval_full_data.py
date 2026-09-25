"""全データ (FRDM 全 run + UNO Q) で大型前段 + Solist ELM ヘッドを学習し、現地校正込みで評価する。

データ
  FRDM : full_32_train_v2..v12, v21..v25, v5/v5_part2, eval_quiet/noise_low/noise_high, full_32_eval_v1 (7 日)
  UNO Q: train session1..8 + eval gray/evening/survey/crowd (2026-09-12)
特徴 : N333 = 1024-bin noise_diff_norm (2048 点 Welch log-PSD − 同 run baseline log-PSD, frame 毎 z 正規化)
        の 400–3000 Hz。baseline frame は評価・学習サンプルから除外する
        (FRDM: s00000 先頭 min(10, n/2) frame, UNO Q: meta group=="baseline")。
評価
  * FRDM fold: 評価日 = FRDM の 1 日、学習 = 残り FRDM 日 + UNO Q 全部 (評価日は学習/標準化/early stop に不使用)
  * UNO Q fold: 評価 = UNO Q eval 4 セット、学習 = FRDM 全部 + UNO Q session1..8
  * early stopping は学習側の run 単位 hold-out (frame 分割ではない)
  * 学習 aug: 1024-bin 上で周波数シフト ε~U(−2%, +2%) (比例シフト, 1 サンプル毎)
校正 (主指標, ユーザー方針: 運用で現地校正する)
  評価日の別 run / 別 eval セットから状態毎に校正サンプルを取り、factory と混合して ELM β を再計算、
  校正に使わない run で評価。
  * win5: 連続 3 frame を 6 秒連続音とみなし 2 秒窓・1 秒ずらしで 5 窓 (ユーザー指定の運用手順)
  * ind5 / ind10: 独立 frame 5 / 10 個 (比較用)
モデル (速度無視・Flash ≲180KB int8 を目安に最大化)
  B1: Conv 16-32-64-64 + FC64 (≈173KB) / B2: Conv 16-32-64 + FC64 (≈166KB) / E3: T8-16-32 FC32 × 3 (≈125KB, 埋め込み連結 96)
  ヘッド: ELM m32 / m64 (乱数α; 実機αは未取得) と PC 側の CNN ヘッド (参考)

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_full_data.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_improve_lodo import Elm, rand_alpha, to14, macro_f1  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
CACHE = ROOT / "sim" / "_cache"
FRDM = Path(r"D:/GitHub/IchiPing/pc/captures")
UNOQ = Path(r"D:/GitHub/IchiPing-UNO-Q/pc/captures")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NFFT, HOP = 2048, 1024
BIN_HZ = 16_000 / NFFT
BLO, BHI = int(round(400 / BIN_HZ)) - 1, int(round(3000 / BIN_HZ))     # 1024-bin 上の 400–3000 Hz (333 bin)
C = 32
SHIFT_MAX = 0.02

FRDM_RUNS = ([f"full_32_train_v{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25)]
             + ["full_32_train_v5_part2", "eval_quiet", "eval_noise_low", "eval_noise_high", "full_32_eval_v1"])
UNOQ_TRAIN = [f"uno_q_train_20260912_session{i}_wav" for i in range(1, 7)] + \
             [f"uno_q_train_20260912_session{i}_loud_wav" for i in (7, 8)]
UNOQ_EVAL = [f"uno_q_eval_20260912_{n}_wav" for n in ("gray", "evening", "survey", "crowd")]
# UNO Q eval の校正ペア (校正セット → 評価セット): 時刻の近い別セッション。gray は 3 frame/状態のため評価のみ
UNOQ_CAL_PAIRS = [("evening", "survey"), ("survey", "evening"), ("survey", "crowd"), ("crowd", "survey")]


# ------------------------------------------------------------------ features
def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


_WIN = np.hanning(NFFT)


def seg_power(a: np.ndarray) -> np.ndarray:
    """(n_seg, 1024) の区間パワー (DC 除外)。"""
    seg = np.stack([a[s:s + NFFT] for s in range(0, len(a) - NFFT + 1, HOP)]) * _WIN
    return (np.abs(np.fft.rfft(seg, axis=1)) ** 2)[:, 1:]


def to_db(P: np.ndarray) -> np.ndarray:
    return np.maximum(10 * np.log10(P + 1e-12), -80.0)


def norm_diff(db: np.ndarray, base: np.ndarray) -> np.ndarray:
    d = db - base
    return ((d - d.mean(-1, keepdims=True)) / (d.std(-1, keepdims=True) + 1e-6)).astype(np.float32)


def run_path(name: str) -> Path:
    return (UNOQ if name.startswith("uno_q") else FRDM) / name


def state_frames(run: Path):
    """[(state_idx, [wav...], [is_baseline...])], 時間順。"""
    out = []
    for d in sorted(run.iterdir()):
        n = d.name
        if not (d.is_dir() and n.startswith("s") and len(n) == 6 and set(n[1:]) <= {"0", "1"}):
            continue
        c = sum(int(b) << k for k, b in enumerate(n[1:]))
        wavs = sorted(d.glob("frame_*.wav"))
        groups = {}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            groups = {f["wav"]: f.get("group") for f in meta.get("frames", []) if "wav" in f}
        if c == 0:
            if groups:
                isb = [groups.get(w.name) == "baseline" for w in wavs]
            else:
                nb = min(10, len(wavs) // 2)
                isb = [i < nb for i in range(len(wavs))]
        else:
            isb = [False] * len(wavs)
        out.append((c, wavs, isb))
    return out


def build_run(name: str) -> dict:
    """run 毎キャッシュ: 各 frame の N1024 と、校正窓用に状態毎先頭 3 frame の区間パワー。"""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / f"full_{name}.npz"
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    sf = state_frames(run_path(name))
    base_db = np.mean([to_db(seg_power(load_wav(w)).mean(0)) for c, ws, bs in sf for w, b in zip(ws, bs) if b], axis=0)
    X, y, order, win = [], [], [], []
    for c, ws, bs in sf:
        k = 0
        segs = []
        for w, b in zip(ws, bs):
            if b:
                continue
            P = seg_power(load_wav(w))
            X.append(norm_diff(to_db(P.mean(0)), base_db)); y.append(c); order.append(k); k += 1
            if len(segs) < 3:
                segs.append(P)
        if len(segs) == 3:                                  # 6 秒連続音近似: 2 秒窓・1 秒ずらし 5 窓
            h = len(segs[0]) // 2
            parts = [segs[0], np.r_[segs[0][h:], segs[1][:h]], segs[1], np.r_[segs[1][h:], segs[2][:h]], segs[2]]
            win.append(np.stack([norm_diff(to_db(p.mean(0)), base_db) for p in parts]))
        else:
            win.append(np.full((5, 1024), np.nan, np.float32))
    d = {"X": np.stack(X).astype(np.float16), "y": np.array(y), "order": np.array(order),
         "win5": np.stack(win).astype(np.float16)}
    np.savez_compressed(cp, **d)
    print(f"  cached {name}: {d['X'].shape}", flush=True)
    return d


def run_day(name: str) -> str:
    if name.startswith("uno_q"):
        return "2026-09-12"
    return json.loads((FRDM / name / "s00000" / "meta.json").read_text(encoding="utf-8"))["started_at"][:10]


# ------------------------------------------------------------------ models
def conv_stack(spec, fc):
    layers, ch, L = [], 1, BHI - BLO
    for oc, k, s in spec:
        layers += [nn.Conv1d(ch, oc, k, stride=s), nn.BatchNorm1d(oc), nn.ReLU()]
        ch, L = oc, (L - k) // s + 1
    params = sum(ci * co * k + co for (ci, (co, k, _)) in zip([1] + [s[0] for s in spec[:-1]], spec))
    layers += [nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch * L, fc), nn.ReLU()]
    return nn.Sequential(*layers), params + ch * L * fc + fc


class Net(nn.Module):
    def __init__(self, spec, fc):
        super().__init__()
        self.front, self.params = conv_stack(spec, fc)
        self.head = nn.Linear(fc, C)

    def forward(self, x):
        return self.head(self.front(x))


ARCH = {
    "B1 Conv16-32-64-64 FC64": [([(16, 9, 2), (32, 7, 2), (64, 5, 2), (64, 3, 1)], 64)],
    "B2 Conv16-32-64 FC64": [([(16, 9, 2), (32, 7, 2), (64, 5, 2)], 64)],
    "E3 (Conv8-16-32 FC32)×3": [([(8, 9, 2), (16, 7, 2), (32, 5, 2)], 32)] * 3,
}


def shift_batch(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """1024-bin 上の比例周波数シフト (bin k=(k+1)Δf, 端は保持)。"""
    n = x.shape[1]
    src = (torch.arange(1, n + 1, device=x.device)[None, :] / (1 + eps[:, None])) - 1
    src = src.clamp(0, n - 1)
    lo = src.floor().long(); hi = (lo + 1).clamp(max=n - 1); w = src - lo
    return torch.gather(x, 1, lo) * (1 - w) + torch.gather(x, 1, hi) * w


def train_net(spec, fc, Xtr, ytr, Xva, yva, seed, epochs=200, patience=20):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(DEV); yt = torch.from_numpy(ytr).to(DEV)
    band = Xt[:, BLO:BHI]
    mu, sd = band.mean(0), band.std(0) + 1e-6
    prep = lambda A: ((A[:, BLO:BHI] - mu) / sd)[:, None, :]
    Xv = prep(torch.from_numpy(Xva.astype(np.float32)).to(DEV)); yv = torch.from_numpy(yva).to(DEV)
    net = Net(spec, fc).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_loss, stale = None, 1e9, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen).to(DEV)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            eps = (torch.rand(len(b), device=DEV) * 2 - 1) * SHIFT_MAX
            opt.zero_grad(set_to_none=True)
            lossf(net(prep(shift_batch(Xt[b], eps))), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(nn.functional.cross_entropy(net(Xv), yv))
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss, stale = vl, 0
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best); net.eval()

    def apply(A, fn):
        out = []
        with torch.no_grad():
            for i in range(0, len(A), 4096):
                out.append(fn(prep(torch.from_numpy(A[i:i + 4096].astype(np.float32)).to(DEV))).cpu())
        return torch.cat(out).numpy().astype(np.float64)
    return net, (lambda A: apply(A, net.front)), (lambda A: apply(A, net))


# ------------------------------------------------------------------ evaluation
def evaluate(S, y, g):
    pred = S.argmax(1)
    votes = [int(S[(g == r) & (y == c)].sum(0).argmax() == c)
             for r in np.unique(g) for c in range(C) if np.any((g == r) & (y == c))]
    return {"frame": float(np.mean(pred == y)), "macro_f1": macro_f1(pred, y, C),
            "vote": float(np.mean(votes)), "as14": float(np.mean(to14(pred) == to14(y)))}


# win5 は状態ディレクトリ名のソート順 (s00000, s00001, ...) で格納されている。クラス番号は bit k → 2^k。
WIN_LABELS = np.array([sum(int(b) << k for k, b in enumerate(f"{i:05b}")) for i in range(C)])


def cal_set(d, kind):
    """校正サンプル (特徴 1024, ラベル)。"""
    if kind == "win5":
        ok = ~np.isnan(d["win5"].astype(np.float32)).any(axis=(1, 2))
        W = d["win5"][ok].astype(np.float32)
        return W.reshape(-1, 1024), np.repeat(WIN_LABELS[ok], 5)
    n = int(kind[3:])
    m = d["order"] < n
    return d["X"][m].astype(np.float32), d["y"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B2 のみ・seed 1・FRDM 2 fold で動作確認")
    args = ap.parse_args()
    seeds = (0,) if args.quick else (0, 1)
    runs = FRDM_RUNS + UNOQ_TRAIN + UNOQ_EVAL
    data = {r: build_run(r) for r in runs}
    day = {r: run_day(r) for r in runs}
    frdm_days = sorted({day[r] for r in FRDM_RUNS})
    print("FRDM days:", {d: [r for r in FRDM_RUNS if day[r] == d] for d in frdm_days}, flush=True)

    folds = [("FRDM " + d, [r for r in FRDM_RUNS if day[r] == d]) for d in frdm_days] + [("UNO Q eval", UNOQ_EVAL)]
    archs = dict(ARCH)
    if args.quick:
        folds = folds[:2]; archs = {k: v for k, v in archs.items() if k.startswith("B2")}
    # 学習データ条件: 主 = FRDM+UNO Q。ablation = FRDM fold は UNO Q 無し、UNO Q fold は FRDM 無し (B1 のみ)
    conds = [("FRDM+UNO Q", None)] + ([] if args.quick else [("単一ドメイン", "B1 Conv16-32-64-64 FC64")])
    res = {"folds": {}, "params": {}}
    rng = np.random.default_rng(0)
    for fold, test_runs in folds:
        is_unoq = fold.startswith("UNO Q")
        pool = [r for r in runs if r not in test_runs and not (is_unoq and r in UNOQ_EVAL)]
        for cond, only_arch in conds:
            train_runs = pool if cond == "FRDM+UNO Q" else \
                [r for r in pool if r.startswith("uno_q") == is_unoq]
            # early stop 用: 各ドメインから学習 run を 1 つずつ hold-out
            val_runs = []
            for dom in ("uno_q", "full_32", "eval_"):
                cand = [r for r in train_runs if r.startswith(dom)]
                if len(cand) > 2:
                    val_runs.append(cand[rng.integers(len(cand))])
            fit_runs = [r for r in train_runs if r not in val_runs]
            cat = lambda rs, k: np.concatenate([data[r][k] for r in rs])
            Xf, yf = cat(fit_runs, "X"), cat(fit_runs, "y")
            Xv, yv = cat(val_runs, "X"), cat(val_runs, "y")
            Xall, yall = cat(train_runs, "X"), cat(train_runs, "y")
            for aname, members in archs.items():
                if only_arch and aname != only_arch:
                    continue
                for seed in seeds:
                    embeds, logits_fns = [], []
                    for j, (spec, fc) in enumerate(members):
                        net, emb, lg = train_net(spec, fc, Xf, yf, Xv, yv, seed * 10 + j)
                        embeds.append(emb); logits_fns.append(lg)
                        res["params"][aname] = res["params"].get(aname, 0) if j else 0
                        res["params"][aname] += net.params
                    E = lambda A: np.concatenate([e(A) for e in embeds], axis=1)
                    Z = E(Xall)
                    heads = {"CNN ヘッド (PC 参考)": None}
                    for m in (32, 64):
                        heads[f"ELM m{m}"] = Elm(rand_alpha(Z.shape[1], m, seed + 1), 0.5, 1.0).fit(Z, yall, C)
                    tests = {r: (E(data[r]["X"].astype(np.float32)), data[r]["y"]) for r in test_runs}
                    for hname, h in heads.items():
                        key = f"{fold}|{cond}|{aname}|{hname}"
                        rec = res["folds"].setdefault(key, {"factory": [], "win5": [], "ind5": [], "ind10": []})
                        for r, (Ze, ye) in tests.items():
                            S = sum(lg(data[r]["X"].astype(np.float32)) for lg in logits_fns) if h is None else h.scores(Ze)
                            rec["factory"].append(evaluate(S, ye, np.zeros(len(ye))))
                        if h is None:
                            continue
                        if is_unoq:
                            pairs = [(f"uno_q_eval_20260912_{a}_wav", f"uno_q_eval_20260912_{b}_wav") for a, b in UNOQ_CAL_PAIRS]
                        else:
                            pairs = [(a, b) for a in test_runs for b in test_runs if a != b]
                        for kind in ("win5", "ind5", "ind10"):
                            for rc, rt in pairs:
                                Xc, yc = cal_set(data[rc], kind)
                                hm = h.recal(E(Xc), yc, "mix")
                                Ze, ye = tests[rt]
                                rec[kind].append(evaluate(hm.scores(Ze), ye, np.zeros(len(ye))))
                    f = res["folds"][f"{fold}|{cond}|{aname}|ELM m32"]
                    mean = lambda k: np.mean([q["frame"] for q in f[k]]) if f[k] else float("nan")
                    print(f"{fold} | {cond} | {aname} | seed{seed} ELM m32 factory={mean('factory'):.3f} "
                          f"win5={mean('win5'):.3f} ind10={mean('ind10'):.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FULL_DATA.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res, frdm_days)


def write_md(res, frdm_days):
    p = lambda v: "—" if v != v else f"{v*100:.1f}%"
    keys = res["folds"]
    conds = sorted({k.split("|")[1] for k in keys}, key=lambda c: c != "FRDM+UNO Q")
    archs = list(dict.fromkeys(k.split("|")[2] for k in keys))
    heads = list(dict.fromkeys(k.split("|")[3] for k in keys))

    def agg(fold_filter, cond, a, h, kind, metric="frame"):
        vals = []
        for k, rec in keys.items():
            f, c, aa, hh = k.split("|")
            if c == cond and aa == a and hh == h and fold_filter(f) and rec[kind]:
                vals.append(np.mean([q[metric] for q in rec[kind]]))     # fold 内平均 → fold 平均
        return float(np.mean(vals)) if vals else float("nan")

    L = ["# 全データ (FRDM 7 日 + UNO Q) × 大型前段 + Solist ELM ヘッド", "",
         "FRDM は日単位 leave-one-day-out (学習 = 残り FRDM + UNO Q 全部)、UNO Q は eval 4 セット (学習 = FRDM 全部 + UNO Q session1–8)。",
         "early stop は学習側 run の hold-out。学習 aug = 周波数シフト ±2%。baseline frame は評価から除外。",
         "**校正あり (主指標)** = 評価日の別 run (UNO Q は時刻の近い別 eval セット) から状態毎に校正サンプルを取り factory と混合して β 再計算。",
         "win5 = 6 秒連続音の 2 秒窓・1 秒ずらし 5 窓 (運用手順), ind5/ind10 = 独立 frame 5/10 個。値は frame 精度 (fold 平均)。", "",
         f"パラメータ (int8 Flash 目安): " + ", ".join(f"{a} {n/1024:.0f}KB" for a, n in res["params"].items()), ""]
    for scope, filt in (("FRDM 日単位 LODO", lambda f: f.startswith("FRDM")), ("UNO Q eval", lambda f: f.startswith("UNO Q"))):
        L += [f"## {scope}", "",
              "| 学習データ | 前段 | ヘッド | 校正なし | 校正あり win5 | 校正あり ind5 | 校正あり ind10 | win5 macroF1 | win5 状態投票 | win5 14cls換算 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cond in conds:
            for a in archs:
                for h in heads:
                    fac = agg(filt, cond, a, h, "factory")
                    if fac != fac:
                        continue
                    L.append(f"| {cond} | {a} | {h} | {p(fac)} | {p(agg(filt, cond, a, h, 'win5'))} | "
                             f"{p(agg(filt, cond, a, h, 'ind5'))} | {p(agg(filt, cond, a, h, 'ind10'))} | "
                             f"{p(agg(filt, cond, a, h, 'win5', 'macro_f1'))} | {p(agg(filt, cond, a, h, 'win5', 'vote'))} | "
                             f"{p(agg(filt, cond, a, h, 'win5', 'as14'))} |")
        L.append("")
    L += ["## fold 別 (FRDM+UNO Q, ELM m64, 校正なし → win5 校正)", "", "| fold | " + " | ".join(archs) + " |",
          "|---|" + "---:|" * len(archs)]
    for fold in [f"FRDM {d}" for d in frdm_days] + ["UNO Q eval"]:
        cells = []
        for a in archs:
            rec = keys.get(f"{fold}|FRDM+UNO Q|{a}|ELM m64")
            cells.append("—" if not rec else f"{p(np.mean([q['frame'] for q in rec['factory']]))} → "
                         f"{p(np.mean([q['frame'] for q in rec['win5']]) if rec['win5'] else float('nan'))}")
        L.append(f"| {fold} | " + " | ".join(cells) + " |")
    L += ["", "注: ELM は一様乱数α (実機/公式Sim の α は未取得)。m64 の AI RAM は公式 Sim で要確認。",
          "UNO Q の校正なし値は gray を含む 4 セット、校正あり値は校正ペア (evening/survey/crowd 間) の評価側のみ。"]
    (OUT / "FULL_DATA.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
