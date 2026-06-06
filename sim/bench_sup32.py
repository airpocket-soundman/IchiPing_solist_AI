"""32cls 識別的ルート: 教師あり単一ELM (βを分類に学習) で精度向上を狙う。

AE-bank(生成的)は strict ~50-62% で頭打ち(微小サブ状態差を拾えない)。
識別的=βを「32クラス one-hot 回帰」に最小二乗学習し argmax。同じ α 乱数固定。
Solist-AI 教師あり版ライブラリ(出力≠入力)に対応。容量無視(SD前提)で大K。
比較: IchiPing CNN(識別的) 32cls ~88%。
"""
from __future__ import annotations
import numpy as np
from bench_max32 import build_train, feat_aug, Std, cropb, vote, collapse14
from bench_v612 import baseline_of, build_run, EVAL_SETS, _act
from solist_elm import accuracy


def main():
    classes=list(range(32)); ridge=1e-1
    for lo,hi in ((200,6000),(16,8000)):
        Xtr,ytr,D=build_train(lo,hi)
        Xtr=feat_aug(Xtr,1,7); ytr=np.concatenate([ytr,ytr])
        st=Std().fit(Xtr); Xz=st(Xtr).astype(np.float32)
        Y=np.zeros((len(ytr),32),np.float32); Y[np.arange(len(ytr)),ytr]=1.0
        print(f"\n###### 教師あり32cls band {lo}-{hi} D={D} train={Xz.shape[0]} ######", flush=True)
        for K in (128,256,512):
            rng=np.random.default_rng(0)
            alpha=(rng.standard_normal((D,K))/np.sqrt(D)).astype(np.float32)
            bias=(rng.standard_normal(K)*0.1).astype(np.float32)
            H=_act("hard_sigmoid", Xz@alpha+bias).astype(np.float64)
            beta=np.linalg.solve(H.T@H+ridge*np.eye(K), H.T@Y.astype(np.float64))  # (K,32)
            def pred(Xq):
                Hq=_act("hard_sigmoid", Xq@alpha+bias)
                return (Hq@beta)
            line=f"  K={K:>3}: "
            for es in EVAL_SETS:
                z=build_run(es,baseline_of(es),"bl_self"); Xe,_=cropb(z[0],lo,hi); Xe=st(Xe).astype(np.float32)
                y32e=z[2]; dec=pred(Xe)
                # frame strict
                strict=accuracy(y32e, dec.argmax(1))
                # vote (soft, per state)
                cor=tot=0
                for s in np.unique(y32e):
                    idx=np.where(y32e==s)[0]; cor+=int(dec[idx].sum(0).argmax()==s); tot+=1
                sv=cor/tot
                p14=collapse14(dec.argmax(1)); t14=collapse14(y32e)
                line+=f"{es.split('_')[-1]}:f{strict:.0%}/vote{sv:.0%}/14c{accuracy(t14,p14):.0%}  "
            print(line, flush=True)


if __name__=="__main__":
    main()
