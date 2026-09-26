"""このハード (Stamp-S3A 採取, captures/stamp_*) のセッションを PC 上の工場モデルで評価する。

特徴は学習と同じ float 経路 (eval_full_data.build_run: N1024, 同セッション baseline 差) →
int8 CNN 前段 (board_model_frontend_32cls.npz) → ELM (AxlCORE と同じ bf16 参照計算)。
実機ファームが PC と同じ特徴を出していれば、実機サーベイの結果はこの PC 評価と揃うはず。

usage: python sim/eval_stamp_session.py stamp_20260926_quick1_wav [...] [--model sim_export/solist_ds/board_model_frontend_32cls.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_data import BLO, BHI, CACHE, build_run  # noqa: E402
from emit_frontend_model import ALPHA, elm_input, int_forward, mcu_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_model(p):
    z = np.load(p)
    n = sum(1 for k in z.files if k.endswith("_wq") and k.startswith("conv"))
    qm = [dict(wq=z[f"conv{i}_wq"], M=z[f"conv{i}_M"], B=z[f"conv{i}_B"], k=int(z[f"conv{i}_k"]), s=int(z[f"conv{i}_s"]))
          for i in range(n)]
    fq = dict(wq=z["fc_wq"], M=z["fc_M"], B=z["fc_B"])
    return z, qm, fq


def predict(X, z, qm, fq, beta=None):
    xs = ((X[:, BLO:BHI].astype(np.float32) - z["in_mu"]) / z["in_sd"]).astype(np.float32)
    xq = np.clip(np.round(xs / float(z["s_in"])), -127, 127).astype(np.int8)
    E = int_forward(xq, qm, fq)
    Zp = elm_input(E, z["emb_mul"], z["emb_add"])
    return mcu_reference(Zp, z["beta"] if beta is None else beta), Zp


def label(c):
    return "s" + "".join(str((c >> k) & 1) for k in range(5))


def class14(c):
    a, b, cc, ab, bc = [(c >> k) & 1 for k in range(5)]
    if ab == 0:
        return 0 if a == 0 else 1
    if bc == 0:
        return 2 + a + 2 * b
    return 6 + a + 2 * b + 4 * cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "sim_export" / "solist_ds" / "board_model_frontend_32cls.npz"))
    ap.add_argument("--fresh", action="store_true", help="特徴キャッシュを作り直す")
    a = ap.parse_args()
    z, qm, fq = load_model(a.model)
    c14 = np.array([class14(c) for c in range(32)])
    for r in a.runs:
        if a.fresh:
            (CACHE / f"full_{r}.npz").unlink(missing_ok=True)
        d = build_run(r)
        X, y = d["X"].astype(np.float32), d["y"]
        P, _ = predict(X, z, qm, fq)
        pr = P.argmax(1)
        print(f"{r}: frames={len(y)}  32cls={np.mean(pr == y):.3f}  14cls={np.mean(c14[pr] == c14[y]):.3f}")
        for c in range(32):
            m = y == c
            if m.any():
                preds = " ".join(label(p) for p in pr[m])
                print(f"  {label(c)} {np.mean(pr[m] == c):.2f}  {preds}")


if __name__ == "__main__":
    main()
