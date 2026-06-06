"""学習が最も進みやすい Sim 用データセットを生成 (14cls, クラス均衡, クリーン)。

OS-ELM が学習しやすい条件:
 - 14cls (易/分離良)、cross-baseline(v6/v9/v11)で頑健性は確保、**aug無し(クリーン)**で当て易く
 - **クラス均衡**(各クラス同数) → 逐次RLSが偏らず学習が進む
 - ≤100万セル(行×列)に収める
学習=v6-v11、テスト=v12(自己baseline)。標準化は学習統計。
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_v612 import baseline_of, build_run, crop
from common import class_idx_14
from solist_elm import _act, accuracy

OUT = Path(__file__).resolve().parent.parent / "sim_export"
TR=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
BL=["full_32_train_v6","full_32_train_v9","full_32_train_v11"]
LO,HI=400,3000; PER_CLASS=390   # 14×390=5460 行 (≤1M: ×181=988,260)

def y14_of(y32): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in y32])
def onehot(y,C): M=np.zeros((len(y),C),int); M[np.arange(len(y)),y]=1; return M

def main():
    OUT.mkdir(exist_ok=True)
    # cross-baseline 全特徴
    bls={b:baseline_of(b) for b in BL}
    X=[]; y=[]
    for run in TR:
        for b,bl in bls.items():
            Xr,_,y32=build_run(run,bl,f"bl_{b[-3:]}"); xx,D=crop(Xr,LO,HI)
            X.append(xx); y.append(y14_of(y32))
    X=np.concatenate(X); y=np.concatenate(y)
    # クラス均衡サンプリング
    rng=np.random.default_rng(0); idx=[]
    for c in range(14):
        ci=np.where(y==c)[0]
        idx.append(rng.choice(ci, size=min(PER_CLASS,len(ci)), replace=len(ci)<PER_CLASS))
    idx=np.concatenate(idx); rng.shuffle(idx)
    Xtr=X[idx]; ytr=y[idx]
    print(f"均衡学習: {Xtr.shape} (クラス分布 {np.bincount(ytr)})  cells={Xtr.shape[0]*(D+14)}")
    # テスト v12
    Xv,_,y32v=build_run("full_32_train_v12", baseline_of("full_32_train_v12"), "bl_self_v12")
    Xte,_=crop(Xv,LO,HI); yte=y14_of(y32v)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    def write(p,Xx,yy):
        head=[f"f{i+1}" for i in range(D)]+[f"c{j}" for j in range(14)]
        mat=np.hstack([(Xx-mu)/sd, onehot(yy,14).astype(float)])
        np.savetxt(p,mat,delimiter=",",fmt="%.6f",header=",".join(head),comments="")
        print(f"  {p.name}: {mat.shape[0]}行×{mat.shape[1]}列")
    write(OUT/"train_best_14cls.csv", Xtr, ytr)
    write(OUT/"test_best_14cls.csv",  Xte, yte)
    # 参考: バッチLS(理想)での到達精度 (OS-ELMが収束すればこの近辺)
    Xn=(Xtr-mu)/sd; Xt=(Xte-mu)/sd
    rngk=np.random.default_rng(1); a=rngk.uniform(-0.2,0.2,(D,32))
    H=_act("hard_sigmoid",Xn@a); Y=onehot(ytr,14).astype(float)
    beta=np.linalg.solve(H.T@H+1e-1*np.eye(32),H.T@Y)
    pred=(_act("hard_sigmoid",Xt@a)@beta).argmax(1)
    print(f"\n[参考] バッチLS(K=32 hard_sigmoid)での v12 14cls 正解率: {accuracy(yte,pred):.1%}")
    print(f"出力先: {OUT}")

if __name__=="__main__":
    main()
