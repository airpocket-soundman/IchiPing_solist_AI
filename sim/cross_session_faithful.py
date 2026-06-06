"""faithful Solist-AIモデルで「別セッション/別条件 + State校正」の実力を測る。

ユーザー指摘: 同一セッション内(5/state校正→残り)は連続録音の使い回しで汎化ゼロ=無意味。
実力 = 別セッション/別条件で、現地でState校正(β学習)したものを、別の録音でテスト。

シナリオ1(別セッション・同setup): β校正=v6-v11(state付), テスト=v12(別セッション, 完全hold-out)
シナリオ2(別条件): β校正=eval_quiet(state付), テスト=eval_noise_low / eval_noise_high
αは乱数固定(faithful), 標準化は校正データ統計。14cls/32cls, K=64/256。
"""
from __future__ import annotations
import numpy as np
from solist_faithful import SolistAI, Norm, collapse14, votescore, load
from rich_feat32 import feat_single, build
from solist_elm import accuracy

def run(name, train_runs, test_runs):
    Xtr,y32tr = load(train_runs)
    print(f"\n######## {name} ########")
    print(f"  校正(β学習)= {train_runs} : {len(Xtr)} frames")
    for ncls,tag in ((14,"14cls"),(32,"32cls")):
        ytr = collapse14(y32tr) if ncls==14 else y32tr
        for K in (64,256):
            nm=Norm().fit(Xtr); Xn=nm(Xtr)
            m=SolistAI(Xtr.shape[1],K,ncls,scale_alpha=0.2,l2=1e-1,seed=1,use_bf16=True)
            m.fit(Xn,ytr)
            outs=[]
            for tr in test_runs:
                Xe,y32e=load([tr]); ye=collapse14(y32e) if ncls==14 else y32e
                dec=m.decision(nm(Xe))
                outs.append(f"{tr.replace('full_32_train_','').replace('eval_','')}:"
                            f"{accuracy(ye,dec.argmax(1)):.0%}/v{votescore(dec,ye,y32e):.0%}")
            print(f"  {tag} K={K:>3}: " + "  ".join(outs))

def main():
    TR611=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
    run("シナリオ1: 別セッション同setup (v6-11校正→v12テスト)", TR611, ["full_32_train_v12"])
    run("シナリオ2: 別条件 (eval_quiet校正→noise_low/highテスト)",
        ["eval_quiet"], ["eval_noise_low","eval_noise_high"])

if __name__=="__main__":
    main()
