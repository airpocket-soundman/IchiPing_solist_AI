"""32cls を AE-bank (32 検出器) で検証 + 観測等価限界の分析。

32 状態それぞれに AE を持たせ argmin。32cls は「扉が閉だと内側が観測不能」なので
原理的に頭打ち。誤りが観測等価クラス内に収まる(=14cls に collapse すれば正解)なら
実害は無い。そこで次を測る:
  - 32cls 正解率 (厳密)
  - 32cls 予測を 14 等価クラスに collapse した時の正解率 (= 実質の判別力)
"""
from __future__ import annotations

import numpy as np

import feats_diff
import common as C
from solist_elm import accuracy
from aebank import AEBank, Std, crop, vote_eval


def collapse_32_to_14(idx32):
    """32状態 index → 5bit → 14等価クラス index。"""
    bits = np.array([(idx32 >> k) & 1 for k in range(5)])
    return C.class_idx_14(bits)


def main():
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    n = rid.max() + 1; te = rid >= n - 3; tr = ~te
    Xr, D = crop(X, 400, 3000)
    st = Std().fit(Xr[tr]); Xtr, Xte = st(Xr[tr]), st(Xr[te])
    classes32 = sorted(np.unique(y32).tolist())

    print(f"32cls AE-bank D={D} cross-run (test=last3runs), {len(classes32)} states")
    print(f"  {'K':>4}{'ridge':>7}  {'32cls_frame':>11}{'32cls_voteAll':>13}  "
          f"{'→14collapse_frame':>17}{'→14_voteAll':>12}")

    for K in (48, 96):
        for ridge in (1e-1,):
            bank = AEBank(n_hidden=K, ridge=ridge).fit(Xtr, y32[tr], classes32)
            S = bank.scores(Xte)                       # (N,32) 再構成誤差
            pred32 = np.array(classes32)[S.argmin(1)]
            yte, r_te, s_te = y32[te], rid[te], y32[te]

            a32 = accuracy(yte, pred32)
            a32v = vote_eval(S, yte, r_te, s_te, classes32, 0)

            # 14 collapse: 予測 32 と真 32 を共に 14 へ落として比較 (frame)
            pred14 = np.array([collapse_32_to_14(p) for p in pred32])
            true14 = np.array([collapse_32_to_14(t) for t in yte])
            a14 = accuracy(true14, pred14)
            # vote: スコアを 14 クラスに集約 (各14クラス = 属する32状態の最小誤差) してから投票
            # 簡易に: soft-vote は 32 で行い、勝者を 14 へ collapse
            corr = tot = 0
            for r in np.unique(r_te):
                for s in np.unique(s_te[r_te == r]):
                    idx = np.where((r_te == r) & (s_te == s))[0]
                    win32 = np.array(classes32)[S[idx].sum(0).argmin()]
                    corr += int(collapse_32_to_14(win32) == collapse_32_to_14(s)); tot += 1
            a14v = corr / tot

            print(f"  {K:>4}{ridge:>7.0e}  {a32:>10.1%}{a32v:>12.1%}  "
                  f"{a14:>16.1%}{a14v:>11.1%}")

    print("\n  (32cls厳密は観測限界で低い想定。→14collapseが高ければ"
          "「誤りは観測等価クラス内=実害なし」を意味する)")


if __name__ == "__main__":
    main()
