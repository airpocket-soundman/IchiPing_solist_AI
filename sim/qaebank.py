"""固定小数エミュ — AE-bank を量子化して精度劣化と Flash/RAM を測る。

M0+ は FPU 無なので推論は固定小数。ここでは float で学習した α/β を推論経路で
量子化し、cross-run 14cls 精度がどこまで保たれるかを bit 幅別に測る。
ブロック固定小数 (per-tensor スケール) を仮定:
  q(W,b): scale = max|W|/(2^(b-1)-1); Wq = round(W/scale)*scale
推論: xq @ αq → act → hq → @ βq → 再構成誤差 → argmin。
"""
from __future__ import annotations

import numpy as np

import feats_diff
import common as C
from solist_elm import _act, accuracy
from aebank import AEBank, Std, crop, vote_eval


def quant(W, bits):
    if bits >= 32:
        return W.astype(np.float64)
    q = (1 << (bits - 1)) - 1
    scale = (np.abs(W).max() + 1e-12) / q
    return np.clip(np.round(W / scale), -q - 1, q) * scale


def score_quant(bank: AEBank, X, xbits, abits, hbits, bbits):
    Xq = quant(X, xbits)
    aq = quant(bank.alpha, abits)
    bq = quant(bank.bias, abits)
    H = _act(bank.act, Xq @ aq + bq)
    Hq = quant(H, hbits)
    S = np.empty((len(X), len(bank.classes)))
    for j, c in enumerate(bank.classes):
        recon = Hq @ quant(bank.betas[c], bbits)
        S[:, j] = ((Xq - recon) ** 2).mean(axis=1)
    return S


def footprint(K, D, n_cls, bbits, abits):
    """β=K*D*n_cls, α=D*K(共有) のバイト数。"""
    bbytes = (bbits + 7) // 8
    abytes = (abits + 7) // 8
    beta = K * D * n_cls * bbytes
    alpha = D * K * abytes
    return beta, alpha


def main():
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    n = rid.max() + 1; te = rid >= n - 3; tr = ~te
    Xr, D = crop(X, 400, 3000)
    st = Std().fit(Xr[tr]); Xtr, Xte = st(Xr[tr]), st(Xr[te])
    yte, r_te, s_te = y14[te], rid[te], y32[te]
    classes = list(range(14))

    for K in (48, 96):
        bank = AEBank(n_hidden=K, ridge=1e-1).fit(Xtr, y14[tr], classes)
        print(f"\n=== AE-bank K={K} D={D} 量子化スイープ (cross-run, vote-all) ===")
        print(f"  {'x':>4}{'α':>4}{'h':>4}{'β':>4}  {'frame':>7}{'vote5':>7}{'voteAll':>8}  {'Flash':>10}{'RAM/infer':>10}")
        cfgs = [
            (32, 32, 32, 32),   # float 基準
            (16, 16, 16, 16),   # 全 q15
            (12, 8, 16, 16),    # 入力12bit(ADC相当)/α int8/h,β int16
            (12, 8, 8, 8),      # 全 int8 級
            (12, 8, 16, 8),     # β だけ int8 (Flash 半減狙い)
        ]
        for xb, ab, hb, bb in cfgs:
            S = score_quant(bank, Xte, xb, ab, hb, bb)
            a1 = accuracy(yte, np.array(classes)[S.argmin(1)])
            a5 = vote_eval(S, yte, r_te, s_te, classes, 5)
            aall = vote_eval(S, yte, r_te, s_te, classes, 0)
            beta, alpha = footprint(K, D, 14, bb, ab)
            ram = D * ((xb+7)//8) + K * ((hb+7)//8) + D * ((bb+7)//8)  # xq+hq+recon
            star = " ★" if aall >= 0.999 else ""
            print(f"  {xb:>4}{ab:>4}{hb:>4}{bb:>4}  {a1:>6.1%}{a5:>6.1%}{aall:>7.1%}{star}"
                  f"  {(beta+alpha)/1024:>8.1f}KB{ram:>8d}B")


if __name__ == "__main__":
    main()
