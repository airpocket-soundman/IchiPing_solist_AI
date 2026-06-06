"""14cls を 100% に詰める — マルチフレーム投票 + 帯域微調整 + 2モデルアンサンブル。

根拠: デバイスは 1 状態で複数チャープ(フレーム)を撃てる。v1 の live 運用も vote5。
同一 (run, 物理状態) の k フレームの ELM decision を平均してから argmax すると、
フレーム単位のばらつきが平均化され精度が上がる。cross-run (末尾3run hold-out) で評価。
"""
from __future__ import annotations

import numpy as np

from solist_elm import SolistELM, accuracy
import feats_diff
import common as C


class Std:
    def fit(self, X): self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; return self
    def __call__(self, X): return (X - self.mu) / self.sd


def crop(X, lo_hz, hi_hz):
    b = feats_diff.BIN_HZ_DIFF
    lo = max(0, int(round(lo_hz / b)) - 1); hi = min(X.shape[1], int(round(hi_hz / b)))
    return X[:, lo:hi], hi - lo


def vote_eval(dec, y, rid, y32, k):
    """同一 (run, 32状態) を k フレーム窓で soft-vote (decision 平均→argmax)。
    k=0 は全フレーム集約 (その状態の全ショット)。返り値は 14cls accuracy。"""
    correct = tot = 0
    for r in np.unique(rid):
        for s in np.unique(y32[rid == r]):
            m = (rid == r) & (y32 == s)
            idx = np.where(m)[0]
            D = dec[idx]; Y = y[idx]
            if k <= 0:
                groups = [np.arange(len(idx))]
            else:
                groups = [np.arange(i, min(i + k, len(idx)))
                          for i in range(0, len(idx), k)]
            for g in groups:
                pred = D[g].mean(0).argmax()
                correct += int(pred == Y[g][0]); tot += 1
    return correct / tot


def fit_decision(Xtr, ytr, Xte, K, ridge, seeds):
    """seeds アンサンブルの test decision (平均) を返す。"""
    dec = None
    for s in seeds:
        elm = SolistELM(K, "sigmoid", ridge, seed=s).fit(Xtr, ytr, 14)
        d = elm.decision(Xte)
        dec = d if dec is None else dec + d
    return dec / len(seeds)


def main():
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    n = rid.max() + 1
    te = rid >= n - 3; tr = ~te
    print(f"diff_fft X={X.shape}  train={tr.sum()} test={te.sum()} (cross-run, test=last3 runs)")

    for lo, hi in ((400, 3000), (350, 3200), (400, 2800)):
        Xr, D = crop(X, lo, hi)
        st = Std().fit(Xr[tr]); Xtr, Xte = st(Xr[tr]), st(Xr[te])
        print(f"\n=== 帯域 {lo}-{hi}Hz D={D} ===")
        for K, ridge, seeds, tag in ((128, 1.0, (0,), "single"),
                                     (167, 1.0, (0, 1), "2-model(HW可)"),
                                     (128, 1.0, (0, 1, 2, 3), "4-model")):
            Kk = 128 if tag != "2-model(HW可)" else 256
            dec = fit_decision(Xtr, y14[tr], Xte, Kk, ridge, seeds)
            yte, r_te, s_te = y14[te], rid[te], y32[te]
            a1 = accuracy(yte, dec.argmax(1))
            a5 = vote_eval(dec, yte, r_te, s_te, 5)
            a10 = vote_eval(dec, yte, r_te, s_te, 10)
            aall = vote_eval(dec, yte, r_te, s_te, 0)
            print(f"  {tag:>13} K={Kk:>3}: frame={a1:6.1%}  vote5={a5:6.1%}  "
                  f"vote10={a10:6.1%}  vote-all={aall:6.1%}"
                  f"{'  ★100%' if aall>=0.999 else ''}")


if __name__ == "__main__":
    main()
