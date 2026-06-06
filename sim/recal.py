"""オンデバイス β 再計算 — AxlCORE-ODL の現場校正デモを検証。

Solist-AI のキラー機能: α(ランダム射影)は工場固定のまま、β(出力重み)だけを現場で
最小二乗で計算し直す。これがオンデバイス学習。

検証: ある環境(=1 run)で、各状態あたり N フレームだけを「現場校正データ」として
β を学習し、同 run の残りフレームで精度を測る。N をふって「何ショットで高精度に
届くか」を見る。少数で届けば「30秒録音→β再計算→稼働」デモが成立する。

α は工場固定 (別 run で初期化した seed と同じ乱数=環境非依存) を共有。
"""
from __future__ import annotations

import numpy as np

import feats_diff
import common as C
from solist_elm import _act, accuracy
from aebank import Std, crop, vote_eval


def fit_betas(alpha, bias, act, X, y, classes, ridge):
    H = _act(act, X @ alpha + bias)
    betas = {}
    for c in classes:
        Hc = H[y == c]
        Xc = X[y == c]
        A = Hc.T @ Hc + ridge * np.eye(alpha.shape[1])
        betas[c] = np.linalg.solve(A, Hc.T @ Xc)
    return betas


def scores(alpha, bias, act, betas, classes, X):
    H = _act(act, X @ alpha + bias)
    S = np.empty((len(X), len(classes)))
    for j, c in enumerate(classes):
        S[:, j] = ((X - H @ betas[c]) ** 2).mean(axis=1)
    return S


def main():
    # 全 run ロードし、現場校正は「最後の run」(=工場学習に使っていない新環境) で行う
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    K = 48; act = "sigmoid"; ridge = 1e-1
    classes = list(range(14))

    field = rid.max()                       # 新環境とみなす run
    fmask = rid == field
    Xf, yf, sf = X[fmask], y14[fmask], y32[fmask]

    # α は工場固定 (seed 0 の乱数)。環境に依存しない。
    rng = np.random.default_rng(0)
    D_full = Xf.shape[1]

    print(f"現場校正デモ: 新環境=run#{field}, 各状態の先頭 N フレームで β 再計算 → 残りで評価")
    print(f"  {'N/state':>8} {'calib計':>7} {'test計':>7}  {'frame':>7} {'vote5':>7} {'voteAll':>8}")

    for N in (1, 2, 3, 5, 10, 20):
        # 各 (state) で先頭 N を calib、残りを test に (intra-run split)
        calib = np.zeros(len(Xf), dtype=bool)
        for s in np.unique(sf):
            idx = np.where(sf == s)[0]
            calib[idx[:N]] = True
        test = ~calib
        if test.sum() == 0:
            continue

        # 帯域クロップ + 標準化 (現場 calib 統計で)
        Xc_full = Xf[calib]
        st = Std()
        Xc, Dd = crop(Xc_full, 400, 3000); st.fit(Xc)
        Xc = st(Xc)
        Xt = st(crop(Xf[test], 400, 3000)[0])

        alpha = rng2 = np.random.default_rng(0).standard_normal((Dd, K)) / np.sqrt(Dd)
        bias = np.random.default_rng(1).standard_normal(K) * 0.1
        betas = fit_betas(alpha, bias, act, Xc, yf[calib], classes, ridge)

        S = scores(alpha, bias, act, betas, classes, Xt)
        yt, st_lbl = yf[test], sf[test]
        rt = np.zeros(test.sum(), dtype=int)   # 単一 run なので run_id は 0
        a1 = accuracy(yt, np.array(classes)[S.argmin(1)])
        a5 = vote_eval(S, yt, rt, st_lbl, classes, 5)
        aall = vote_eval(S, yt, rt, st_lbl, classes, 0)
        star = " ★" if aall >= 0.999 else ""
        print(f"  {N:>8} {calib.sum():>7} {test.sum():>7}  "
              f"{a1:>6.1%} {a5:>6.1%} {aall:>7.1%}{star}")

    print("\n  (各状態 N ショットで β のみ再計算。少数で高精度なら現場校正デモ成立)")


if __name__ == "__main__":
    main()
