"""現地校正 (Solist-AI の ODL) が「事前学習で見ていない条件」をどこまで取り戻せるかを、手持ちのデータで調べる。

設定: 事前学習は夜ブロック (2026-09-26 s10/s11, 室温低下 ε ≈ -0.15%) を除いた UNO Q + Stamp
      (シフト aug なし, クロスベースライン 2 本, 前段 small; sim/eval_best_model.py の最良構成)。
      現地校正 = 夜 s10 の 1 状態 5 frame (実機と同じ 160 サンプル, Gray code 順)、評価 = 夜 s11 と s10 の残り。
比べる方法:
  none        校正なし
  odl_bf16    AxlCORE と同じ bf16 の OS-ELM (P0 = w (G + λI)^-1, G = 事前学習の Gram)。λ, w を振る
  odl_float   同じ更新を float64 で (参考)
  eps         ラベル不要の温度補正: 起動時 baseline の周波数シフト ε を事前学習の基準 baseline から推定し、
              特徴 (1024 bin) を逆に伸縮してから前段へ
  recenter    ラベル不要: 校正 frame の埋め込み平均を事前学習の平均に合わせる (emb_add の補正)
  + 組み合わせ、校正する状態を減らした場合 (8 / 16 状態)
結果: sim_export/solist_ds/ODL_CALIBRATION.md
usage: python sim/eval_odl_calibration.py [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_full_data as efd  # noqa: E402
import estimate_run_shift as ers  # noqa: E402
from eval_full_data import UNOQ_TRAIN  # noqa: E402
from emit_frontend_model import ALPHA, ARCHS, ELM_HIDDEN, build_candidate, elm_input, emb_norm, mcu_reference  # noqa: E402
from emit_board_model import q  # noqa: E402
from eval_best_model import BLOCKS, STAMP_ALL, build_train, feats, run_db, C14  # noqa: E402
from eval_stamp_mix import weighted_beta  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
CAL, EVAL = "stamp_20260926_s10_wav", "stamp_20260926_s11_wav"


def hs(Z):
    return np.clip(0.2 * (Z @ q(ALPHA)) + 0.5, 0, 1).astype(np.float64)


def acc(P, y):
    p = P.argmax(1)
    return float(np.mean(p == y)), float(np.mean(C14[p] == C14[y]))


def inv_shift(X, eps):
    """(1+ε) 倍にずれた特徴を元に戻す: X'(k) = X(k (1+ε))。"""
    bins = np.arange(1, X.shape[1] + 1, dtype=np.float64)
    return np.stack([np.interp(bins * (1 + eps), bins, x) for x in X.astype(np.float64)]).astype(np.float32)


def base_spectrum(run):
    return ers.run_spectra(ROOT / "captures" / run)["s00000"][0]


def odl(H_seq, y_seq, beta0, P0, bf16):
    beta, P = beta0.astype(np.float64).copy(), P0.astype(np.float64).copy()
    rd = (lambda v: q(v).astype(np.float64)) if bf16 else (lambda v: v)
    P, beta = rd(P), rd(beta)
    for h, c in zip(H_seq, y_seq):
        h = rd(h)
        t = np.zeros(beta.shape[1]); t[c] = 1
        Ph = P @ h
        P = rd(P - np.outer(Ph, Ph) / (1.0 + h @ Ph))
        beta = rd(beta + np.outer(P @ h, t - h @ beta))
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    efd.SHIFT_MAX = 0.0
    spec, emb_dim = ARCHS["small"]
    train_stamp = [r for r in STAMP_ALL if r not in BLOCKS["night"]]
    val = train_stamp[-1]
    tr_runs = UNOQ_TRAIN + train_stamp
    X, y = build_train(tr_runs, 2, np.random.default_rng(0))
    Xf, yf = build_train([r for r in tr_runs if r != val], 2, np.random.default_rng(0))
    dv = run_db(val)
    m = max((build_candidate(spec, emb_dim, X, Xf, yf, feats(dv, dv["base"]), dv["y"], sd)
             for sd in (a.seed, a.seed + 1)), key=lambda c: c["val_acc"])
    E = m["embed"](X)
    mul, add = emb_norm(E, m["s_elm"])
    lam = m["lam"]
    Htr = hs(elm_input(E, mul, add))
    G = Htr.T @ Htr
    Y = np.eye(32)[y]
    beta0 = np.linalg.solve(G + lam * np.eye(ELM_HIDDEN), Htr.T @ Y).astype(np.float32)
    emb_mean = (E * mul + add).mean(0)
    print(f"pretrained (no night): val {m['val_acc']:.3f}, train frames {len(y)}, lambda {lam}", flush=True)

    # 温度補正の基準 = 事前学習に使った Stamp セッションの baseline 平均スペクトル (8192 点)
    ref = np.mean([base_spectrum(r) for r in train_stamp], axis=0)
    dc, de = run_db(CAL), run_db(EVAL)
    eps_cal, _ = ers.fit_eps(ref, base_spectrum(CAL))
    eps_eval, _ = ers.fit_eps(ref, base_spectrum(EVAL))
    print(f"estimated eps: s10 {eps_cal * 100:+.3f}%  s11 {eps_eval * 100:+.3f}%", flush=True)

    Xc_all, yc_all = feats(dc, dc["base"]).astype(np.float32), dc["y"]
    Xe, ye = feats(de, de["base"]).astype(np.float32), de["y"]
    order = [p ^ (p >> 1) for p in range(32)]

    def cal_idx(states):
        return np.concatenate([np.flatnonzero(yc_all == c)[:5] for c in states])

    rest = np.setdiff1d(np.arange(len(yc_all)), cal_idx(order))

    def evaluate(tag, beta, eps=None, recenter=False, states=order):
        ci = cal_idx(states)
        Xc = Xc_all[ci]
        Xr, Xev = Xc_all[rest], Xe
        if eps is not None:
            Xc, Xr = inv_shift(Xc, eps[0]), inv_shift(Xr, eps[0])
            Xev = inv_shift(Xev, eps[1])
        add2 = add
        if recenter:                                    # ラベル不要: 校正 frame の埋め込み平均を事前学習の平均へ
            Ec = m["embed"](Xc) * mul + add
            add2 = add + (emb_mean - Ec.mean(0))
        Z = lambda A: elm_input(m["embed"](A), mul, add2)
        b = beta(Z(Xc), yc_all[ci]) if callable(beta) else beta
        r = {"s11": acc(mcu_reference(Z(Xev), b), ye), "s10 rest": acc(mcu_reference(Z(Xr), b), yc_all[rest])}
        rows.append((tag, r))
        print(f"  {tag:48s} s11 {r['s11'][0]:.3f}  s10-rest {r['s10 rest'][0]:.3f}", flush=True)

    rows = []
    both_eps = (eps_cal, eps_eval)
    evaluate("none (事前学習のまま)", beta0)
    evaluate("eps (温度補正のみ, ラベル不要)", beta0, eps=both_eps)
    evaluate("recenter (埋め込み平均の補正, ラベル不要)", beta0, recenter=True)
    evaluate("eps + recenter", beta0, eps=both_eps, recenter=True)
    n = len(Htr)
    for lam_p, ws in ((10, 0.1), (10, 1.0), (1, 0.1), (1, 1.0), (100, 1.0), (100, 10.0)):
        w = n / 160 * ws
        P0 = w * np.linalg.inv(G + lam_p * np.eye(ELM_HIDDEN))
        for bf in (True, False):
            f = (lambda P0=P0, bf=bf: (lambda Zc, yc: odl(hs(Zc), yc, beta0, P0, bf).astype(np.float32)))()
            evaluate(f"odl {'bf16' if bf else 'float'} λ={lam_p} w={w:.0f}", f)
    # 組み合わせと状態数 (bf16, λ=10, w=n/160: 事前学習と校正を等重み)
    P0 = n / 160 * np.linalg.inv(G + 10 * np.eye(ELM_HIDDEN))
    fb = lambda Zc, yc: odl(hs(Zc), yc, beta0, P0 * len(order) * 5 / len(yc), True).astype(np.float32)
    evaluate("eps + odl bf16 λ=10 w=等重み", fb, eps=both_eps)
    evaluate("eps + recenter + odl bf16 λ=10 w=等重み", fb, eps=both_eps, recenter=True)
    for k in (8, 16):
        evaluate(f"odl bf16 λ=10 w=等重み, {k} 状態だけ校正", fb, states=order[::32 // k])
        evaluate(f"eps + odl bf16 λ=10 w=等重み, {k} 状態だけ校正", fb, eps=both_eps, states=order[::32 // k])

    L = ["# 現地校正 (ODL) の効果: 事前学習で見ていない夜の条件", "",
         "事前学習 = UNO Q + Stamp (夜 s10/s11 を除く), シフト aug なし, クロスベースライン 2, 前段 small。",
         f"現地校正 = 夜 s10 の 1 状態 5 frame (Gray code 順), 評価 = 夜 s11 / s10 の残り。推定 ε: s10 {eps_cal * 100:+.3f}%, s11 {eps_eval * 100:+.3f}%。",
         "", "| 方法 | s11 32cls | s11 14cls | s10 残り 32cls |", "|---|---:|---:|---:|"]
    for tag, r in rows:
        L.append(f"| {tag} | {r['s11'][0] * 100:.1f}% | {r['s11'][1] * 100:.1f}% | {r['s10 rest'][0] * 100:.1f}% |")
    (OUT / "ODL_CALIBRATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
