"""キャリブレーション効果の切り分け (baseline効果 vs β再学習効果)。

A: 工場β + 古いbaseline(v6)        … 校正しない (他環境baselineを流用)
B: 工場β + 現地baseline(eval自身)   … baselineだけ現地で取得
C: B + β再校正(同一eval内N/状態)    … βも現地学習 (※同一セッション=過大評価, 参考)

A→B = 「baselineを撮る効果」(同一評価frameで比較=有効な測定)
B→C = 「β再学習の上積み」(校正と評価が同一セッションなので楽観側, 参考値)
工場βは v6-v12 を cross-baseline で学習 (bench_v612 と同条件, K=32, hard_sigmoid, 標準化なし)。
"""
from __future__ import annotations
import numpy as np
from bench_v612 import (baseline_of, build_run, crop, AEBank, _act,
                        TRAIN_RUNS, BASELINE_RUNS, EVAL_SETS, specaug)
from solist_elm import accuracy
from common import class_idx_14


def vote_states(S, y, y32, classes, k=0):
    classes=np.array(classes); cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]
        groups=[idx] if k<=0 else [idx[i:i+k] for i in range(0,len(idx),k)]
        for g in groups:
            cor+=int(classes[S[g].sum(0).argmin()]==y[g][0]); tot+=1
    return cor/tot


def main():
    LO,HI=400,3000; K=32; ridge=1e-1; ncls=14; classes=list(range(ncls))

    # --- 工場β: v6-v12 cross-baseline (3 baseline) で学習 ---
    bls={r:baseline_of(r) for r in BASELINE_RUNS}
    Xtr=[]; ytr=[]
    for run in TRAIN_RUNS:
        for br,bl in bls.items():
            X,y14,_=build_run(run,bl,f"bl_{br[-3:]}"); Xc,D=crop(X,LO,HI)
            Xtr.append(Xc); ytr.append(y14)
    Xtr=np.concatenate(Xtr); ytr=np.concatenate(ytr)
    Xtr,ytr=specaug(Xtr,ytr,n_extra=1)
    factory=AEBank(K=K,ridge=ridge).fit(Xtr,ytr,classes)
    print(f"工場β学習: X={Xtr.shape} D={D} K={K}\n")

    stale_bl = bls["full_32_train_v6"]   # 古い/他環境 baseline

    print(f"{'eval set':16} {'A:旧baseline':>13} {'B:現地baseline':>15} {'C:+β再校正(N=3)':>17}")
    print(f"{'':16} {'(校正なし)':>13} {'(baselineのみ)':>15} {'(参考/同session)':>17}")
    for es in EVAL_SETS:
        fresh_bl=baseline_of(es)
        Xa,y14a,y32a=build_run(es,stale_bl,"bl_v6stale"); XaC,_=crop(Xa,LO,HI)
        Xb,y14b,y32b=build_run(es,fresh_bl,"bl_self");    XbC,_=crop(Xb,LO,HI)
        # A: factory β + stale baseline
        Sa=factory.scores(XaC); A=vote_states(Sa,y14a,y32a,classes)
        # B: factory β + fresh baseline
        Sb=factory.scores(XbC); B=vote_states(Sb,y14b,y32b,classes)
        # C: fresh baseline + β再校正(同一eval内, N=3/状態, αは工場のまま)
        H=_act("hard_sigmoid", XbC@factory.alpha+factory.bias)
        calib=np.zeros(len(XbC),bool)
        for s in np.unique(y32b):
            idx=np.where(y32b==s)[0]; calib[idx[:3]]=True
        test=~calib; betas={}
        for c in classes:
            m=calib&(y14b==c)
            betas[c]=np.linalg.solve(H[m].T@H[m]+ridge*np.eye(K), H[m].T@XbC[m])
        Sc=np.stack([((XbC[test]-H[test]@betas[c])**2).mean(1) for c in classes],1)
        C=vote_states(Sc,y14b[test],y32b[test],classes)
        print(f"{es:16} {A:>12.1%} {B:>14.1%} {C:>16.1%}")

    print("\n  A→B = baselineを現地で撮る効果 (同一評価frame比較=有効)")
    print("  B→C = β再学習の上積み (校正と評価が同一session=楽観側, 参考)")
    print("  ※経時ドリフト(設置→後日運用)への頑健性は、別時刻の同環境データが無いため測定不能")


if __name__=="__main__":
    main()
