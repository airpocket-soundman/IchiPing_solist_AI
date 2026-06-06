"""eval セットでのオンデバイスβ再校正 — 100%到達の正攻法を検証。

工場βは別セッション(v6-v12)由来でevalに1回校正だと~92%止まり(bench_v612)。
Solist-AIの目玉=現場でβ再計算。α(乱数固定)はそのまま、各eval条件で
各状態N frameだけ使ってβ_cを現場学習→残りで評価。少数で100%なら
「設置時に各扉/窓を一度ずつ→β再計算」で実運用精度が出ることを意味する。
"""
from __future__ import annotations
import numpy as np
from bench_v612 import baseline_of, build_run, crop, EVAL_SETS, _act
from solist_elm import accuracy
from common import class_idx_14


def vote_states(S, y, y32, classes, k):
    classes=np.array(classes); cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]
        groups=[idx] if k<=0 else [idx[i:i+k] for i in range(0,len(idx),k)]
        for g in groups:
            cor+=int(classes[S[g].sum(0).argmin()]==y[g][0]); tot+=1
    return cor/tot


def main():
    LO,HI=400,3000; K=32; ridge=1e-1; ncls=14
    rng=np.random.default_rng(0)
    classes=list(range(ncls))
    print(f"オンデバイスβ再校正 (14cls, K={K}, α乱数固定, 各eval条件で現場校正)")
    print(f"  {'eval set':16} {'N/state':>7} {'calib':>6} {'test':>6}  {'frame':>7} {'vote-all':>8}")
    for es in EVAL_SETS:
        bl = baseline_of(es)                       # 現場で取得した baseline
        X,y14,y32 = build_run(es, bl, "bl_self")   # 自前baselineで diff
        Xc,D = crop(X,LO,HI)
        alpha = rng.standard_normal((D,K))/np.sqrt(D); bias=rng.standard_normal(K)*0.1
        H = _act("hard_sigmoid", Xc@alpha+bias)
        for N in (1,2,3,5):
            calib=np.zeros(len(Xc),bool)
            for s in np.unique(y32):
                idx=np.where(y32==s)[0]; calib[idx[:N]]=True
            test=~calib
            if test.sum()==0: continue
            betas={}
            for c in classes:
                m=calib&(y14==c); Hc=H[m]; Xt=Xc[m]
                if len(Hc)==0: betas[c]=np.zeros((K,D)); continue
                betas[c]=np.linalg.solve(Hc.T@Hc+ridge*np.eye(K), Hc.T@Xt)
            S=np.stack([((Xc[test]-H[test]@betas[c])**2).mean(1) for c in classes],1)
            yt=y14[test]; s32=y32[test]
            af=accuracy(yt, np.array(classes)[S.argmin(1)])
            av=vote_states(S, yt, s32, classes, 0)
            star=" ★" if av>=0.999 else ""
            print(f"  {es:16} {N:>7} {calib.sum():>6} {test.sum():>6}  {af:>6.1%} {av:>7.1%}{star}")


if __name__=="__main__":
    main()
