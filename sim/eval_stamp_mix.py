"""このハード (Stamp-S3A 採取) のセッションを学習に加えたときの効果を、セッション単位 hold-out で比べる。

評価: Stamp セッションを 1 本ずつ評価に回し (leave-one-session-out)、残りを学習側に使う。
      指標は実機と同じ int8 前段 + bf16 ELM 参照計算 (mcu_reference) の 32 クラス / 14 クラス換算。
構成:
  factory       工場モデル (board_model_frontend_32cls.npz) のまま
  beta_mix      前段は工場のまま、ELM β を「UNO Q 学習 + Stamp (評価以外)」で解き直す
                (Stamp 側の重み w = UNO Q frame 数 / Stamp frame 数 × --stamp-weight)。実機は β 差し替えのみ
  beta_stamp    前段は工場のまま、β を Stamp (評価以外) だけで解く
  full_mix      前段 CNN (--arch) から「UNO Q 学習 + Stamp (評価以外)」で学習し直す
                (early stop / ハイパラ選択 = 評価以外の Stamp セッション 1 本)
  full_stamp    前段 CNN から Stamp (評価以外) だけで学習

usage: python sim/eval_stamp_mix.py stamp_20260926_s1_wav stamp_20260926_s2_wav stamp_20260926_s3_wav
       [--configs factory,beta_mix,beta_stamp,full_mix,full_stamp] [--arch small] [--stamp-weight 1.0]
結果: sim_export/solist_ds/STAMP_MIX.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_data import UNOQ_TRAIN, build_run  # noqa: E402
from emit_frontend_model import (ARCHS, ELM_HIDDEN, C, build_candidate, elm_input, emb_norm,  # noqa: E402
                                 mcu_reference, hard_sigmoid, ALPHA)
from eval_stamp_session import load_model, class14  # noqa: E402
from eval_full_data import BLO, BHI  # noqa: E402
from emit_frontend_model import int_forward  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
C14 = np.array([class14(c) for c in range(32)])


def score(P, y):
    p = P.argmax(1)
    return float(np.mean(p == y)), float(np.mean(C14[p] == C14[y]))


def weighted_beta(Zs, ys, ws, lam):
    """β = (Σ w H^T H + λI)^-1 Σ w H^T Y (群毎の重み付き ridge)。"""
    G = lam * np.eye(ELM_HIDDEN)
    R = np.zeros((ELM_HIDDEN, C))
    for Z, y, w in zip(Zs, ys, ws):
        H = hard_sigmoid(Z @ ALPHA).astype(np.float64)
        Y = np.zeros((len(y), C)); Y[np.arange(len(y)), y] = 1
        G += w * H.T @ H; R += w * H.T @ Y
    return np.linalg.solve(G, R).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp", nargs="+")
    ap.add_argument("--configs", default="factory,beta_mix,beta_stamp,full_mix,full_stamp")
    ap.add_argument("--arch", default="small", choices=tuple(ARCHS))
    ap.add_argument("--stamp-weight", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    cfgs = a.configs.split(",")
    z, qm, fq = load_model(ROOT / "sim_export" / "solist_ds" / "board_model_frontend_32cls.npz")

    def factory_Z(X):
        xs = ((X[:, BLO:BHI].astype(np.float32) - z["in_mu"]) / z["in_sd"]).astype(np.float32)
        xq = np.clip(np.round(xs / float(z["s_in"])), -127, 127).astype(np.int8)
        return elm_input(int_forward(xq, qm, fq), z["emb_mul"], z["emb_add"])

    data = {r: build_run(r) for r in UNOQ_TRAIN + a.stamp}
    get = lambda rs, k: np.concatenate([data[r][k] for r in rs])
    Xu, yu = get(UNOQ_TRAIN, "X").astype(np.float32), get(UNOQ_TRAIN, "y")
    Zu = factory_Z(Xu) if any(c.startswith("beta") for c in cfgs) else None
    res = {c: {} for c in cfgs}
    for k, held in enumerate(a.stamp):
        rest = [r for r in a.stamp if r != held]
        Xe, ye = data[held]["X"].astype(np.float32), data[held]["y"]
        Ze = factory_Z(Xe)
        Xs, ys = (get(rest, "X").astype(np.float32), get(rest, "y")) if rest else (None, None)
        for c in cfgs:
            if c == "factory":
                s = score(mcu_reference(Ze, z["beta"]), ye)
            elif c.startswith("beta"):
                if not rest:
                    continue
                Zs = factory_Z(Xs)
                w = len(yu) / len(ys) * a.stamp_weight
                best = None
                for lam in (0.1, 1.0, 10.0):            # ridge は学習側の Stamp で選ぶ (評価セッションは見ない)
                    groups = ([Zu, Zs], [yu, ys], [1.0, w]) if c == "beta_mix" else ([Zs], [ys], [1.0])
                    beta = weighted_beta(*groups, lam)
                    acc = score(mcu_reference(Zs, beta), ys)[0]
                    if best is None or acc > best[0]:
                        best = (acc, beta)
                s = score(mcu_reference(Ze, best[1]), ye)
            else:                                        # full_*: 前段から学習
                if len(rest) < 2 and c == "full_stamp":
                    continue
                val = rest[0]
                fit_stamp = [r for r in rest if r != val]
                tr_runs = (UNOQ_TRAIN if c == "full_mix" else []) + rest
                fit_runs = (UNOQ_TRAIN if c == "full_mix" else []) + fit_stamp
                if not fit_runs:
                    continue
                X, y = get(tr_runs, "X").astype(np.float32), get(tr_runs, "y")
                Xf, yf = get(fit_runs, "X"), get(fit_runs, "y")
                Xv, yv = data[val]["X"], data[val]["y"]
                spec, emb_dim = ARCHS[a.arch]
                cands = [build_candidate(spec, emb_dim, X, Xf, yf, Xv, yv, sd) for sd in range(a.seeds)]
                m = max(cands, key=lambda q: q["val_acc"])
                E = m["embed"](X)
                mul, add = emb_norm(E, m["s_elm"])
                beta = weighted_beta([elm_input(E, mul, add)], [y], [1.0], m["lam"])
                s = score(mcu_reference(elm_input(m["embed"](Xe), mul, add), beta), ye)
            res[c][held] = s
            print(f"  hold-out {held}: {c:10s} 32cls={s[0]:.3f} 14cls={s[1]:.3f}", flush=True)

    lines = ["# このハード (Stamp-S3A) のセッションを学習に加えた効果", "",
             f"Stamp セッション {len(a.stamp)} 本を 1 本ずつ評価 (leave-one-session-out)。int8 前段 + bf16 ELM (実機と同じ参照計算)。",
             f"前段 arch={a.arch}, Stamp 重み×{a.stamp_weight}", "",
             "| 構成 | " + " | ".join(r.replace("_wav", "") for r in a.stamp) + " | 平均 32cls | 平均 14cls |",
             "|---|" + "---:|" * (len(a.stamp) + 2)]
    for c in cfgs:
        v = res[c]
        if not v:
            continue
        cells = [f"{v[r][0] * 100:.1f} / {v[r][1] * 100:.1f}" if r in v else "-" for r in a.stamp]
        m32 = np.mean([x[0] for x in v.values()]); m14 = np.mean([x[1] for x in v.values()])
        lines.append(f"| {c} | " + " | ".join(cells) + f" | **{m32 * 100:.1f}%** | {m14 * 100:.1f}% |")
    lines += ["", "セルは 32 クラス / 14 クラス換算 (%)。"]
    (OUT / "STAMP_MIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "STAMP_MIX.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
