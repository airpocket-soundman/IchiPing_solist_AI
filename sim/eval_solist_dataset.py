"""make_solist_dataset.py の出力を MCU 実装枠 (D=167, m=32, Sim α seed1, hard_sigmoid) の ELM で評価。

- 各 variant の学習特徴で β を最小二乗 (ridge) 学習し、UNO Q eval 4 セット (gray/evening/survey/crowd)
  と FRDM eval_v1 を frame 精度 / 状態別投票精度 (14cls, 32cls) で比較する。
- 評価セットの baseline はそのセットの 'baseline' 群 (電源投入時校正の想定)。学習には混ぜない。
- 併せて m=256 (乱数 α) の参照値も出し、m=32 の頭打ちか データの限界かを切り分ける。

usage: python sim/eval_solist_dataset.py [variant ...]   (省略時: sim/_cache/solist_ds_*.npz 全部)
出力: sim_export/solist_ds/EVAL.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "_cache"
OUT = ROOT / "sim_export" / "solist_ds"
ALPHA32 = np.load(ROOT / "sim_export" / "_alpha32_sim.npy")     # (167,32) Solist-AI Sim seed=1


def hard_sigmoid(z):
    return np.clip(0.2 * z + 0.5, 0.0, 1.0)


def onehot(y, C):
    M = np.zeros((len(y), C)); M[np.arange(len(y)), y] = 1.0; return M


def fit(H, Y, ridge):
    return np.linalg.solve(H.T @ H + ridge * np.eye(H.shape[1]), H.T @ Y)


def vote_acc(P, y, y32):
    cor = tot = 0
    for s in np.unique(y32):
        idx = np.where(y32 == s)[0]
        cor += int(P[idx].sum(0).argmax() == y[idx][0]); tot += 1
    return cor / tot


def evaluate(variant: str, scales=(1.0, 0.5, 0.35, 0.25), ridge=0.1):
    z = np.load(CACHE / f"solist_ds_{variant}.npz")
    X, y14, y32, mu, sd = z["X"], z["y14"], z["y32"], z["mu"], z["sd"]
    evals = sorted({k[5:-2] for k in z.files if k.startswith("eval_") and k.endswith("_X")})
    Xn = (X - mu) / sd
    rows = []
    for C, y in ((14, y14), (32, y32)):
        Y = onehot(y, C)
        for tag, alpha in (("m32(Sim α)", ALPHA32),
                           ("m256(rand α)", np.random.default_rng(1).standard_normal((167, 256)) / np.sqrt(167))):
            best = None
            for s in (scales if alpha.shape[1] == 32 else (1.0, 0.5)):
                beta = fit(hard_sigmoid((Xn * s) @ alpha), Y, ridge)
                res = {}
                for e in evals:
                    Xe = (z[f"eval_{e}_X"] - mu) / sd
                    ye = z[f"eval_{e}_y14"] if C == 14 else z[f"eval_{e}_y32"]
                    P = hard_sigmoid((Xe * s) @ alpha) @ beta
                    res[e] = (float((P.argmax(1) == ye).mean()), vote_acc(P, ye, z[f"eval_{e}_y32"]))
                mean_frame = np.mean([r[0] for k, r in res.items() if k.startswith("unoq")])
                if best is None or mean_frame > best[0]:
                    best = (mean_frame, s, res)
            rows.append((C, tag, best[1], best[2]))
    return evals, rows, len(X), int((z["eps"] != 0).sum())


def main():
    variants = sys.argv[1:] or sorted(p.stem[len("solist_ds_"):] for p in CACHE.glob("solist_ds_*.npz"))
    lines = ["# Solist-AI 向け学習データ評価 (ELM D=167, hard_sigmoid, ridge=0.1)", "",
             "評価 = 別時間帯/別環境の eval セット。各セットの baseline は自身の 'baseline' 群 (電源投入時校正)。",
             "値 = frame 精度 / 状態別投票精度。scale s は UNO Q eval 平均 frame 精度で選択。", ""]
    for v in variants:
        evals, rows, n, nw = evaluate(v)
        lines += [f"## variant `{v}`  (train rows={n}, warped rows={nw})", "",
                  "| cls | model | s | " + " | ".join(evals) + " |",
                  "|---|---|---|" + "---|" * len(evals)]
        for C, tag, s, res in rows:
            cells = [f"{res[e][0]:.1%} / {res[e][1]:.1%}" for e in evals]
            lines.append(f"| {C} | {tag} | {s} | " + " | ".join(cells) + " |")
        lines.append("")
        print("\n".join(lines[-len(rows) - 5:]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT / 'EVAL.md'}")


if __name__ == "__main__":
    main()
