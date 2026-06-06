"""32cls 精度追求 (容量無視=SDカード前提)。大K・広帯域・最大データでAE-bankの限界を探る。

観測等価限界の素の上限 ~44% (C状態8可観測, B状態2択, A状態8択)。これを超えるには
閉扉奥の微小音響サブ構造(パネル共振/サーボ位置)を拾う必要。容量制約が無いので
K・D・データを最大化し、AE-bank(生成的再構成)がどこまで届くか実測する。
比較対象: IchiPing CNN(識別的)は強aug込みで32cls ~88%。
"""
from __future__ import annotations
import numpy as np
from bench_v612 import baseline_of, build_run, crop, TRAIN_RUNS, EVAL_SETS, _act
from solist_elm import accuracy
from common import CACHE_DIR, class_idx_14

ALL_BL = TRAIN_RUNS[:]

class Std:
    def fit(self,X): self.mu=X.mean(0); self.sd=X.std(0)+1e-6; return self
    def __call__(self,X): return (X-self.mu)/self.sd

def cropb(X,lo,hi):
    b=16000/1024; l=max(0,int(round(lo/b))-1); h=min(X.shape[1],int(round(hi/b))); return X[:,l:h],h-l

def feat_aug(X,copies,seed,sigma=0.8,fm_w=40,fm_n=3,lvl=4.0):
    rng=np.random.default_rng(seed); out=[X]; D=X.shape[1]
    for _ in range(copies):
        Xa=X+rng.normal(0,sigma,X.shape).astype(X.dtype)
        Xa=Xa+rng.uniform(-lvl,lvl,(len(Xa),1)).astype(X.dtype)
        for _m in range(fm_n):
            w=int(rng.integers(1,fm_w)); s=int(rng.integers(0,max(1,D-w))); Xa[:,s:s+w]=0.0
        out.append(Xa)
    return np.concatenate(out)

def vote(S,y,y32,classes,k=0):
    classes=np.array(classes); cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]
        groups=[idx] if k<=0 else [idx[i:i+k] for i in range(0,len(idx),k)]
        for g in groups: cor+=int(classes[S[g].sum(0).argmin()]==y[g][0]); tot+=1
    return cor/tot

def collapse14(arr): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in arr])

def build_train(lo,hi):
    # clean 49 (y32 available)
    Xs=[]; ys=[]
    y32_tmpl={}
    for run in TRAIN_RUNS:
        for br in ALL_BL:
            z=np.load(CACHE_DIR/f"v612_{run}__bl_{br[-3:]}.npz")
            xx,D=cropb(z["X"],lo,hi); Xs.append(xx); ys.append(z["y32"])
            y32_tmpl.setdefault(run, z["y32"])
    # noisy 49 (y32 を clean テンプレから復元)
    for run in TRAIN_RUNS:
        for br in ALL_BL:
            f=CACHE_DIR/f"max_noisy_{run}_{br[-3:]}.npz"
            if not f.exists(): continue
            z=np.load(f); xx,_=cropb(z["X"],lo,hi); Xs.append(xx); ys.append(y32_tmpl[run])
    X=np.concatenate(Xs).astype(np.float32); y=np.concatenate(ys)
    return X,y,D

def main():
    classes=list(range(32)); ridge=1e-1
    for lo,hi in ((400,3000),(200,6000)):
        Xtr,ytr,D=build_train(lo,hi)
        Xtr=feat_aug(Xtr,1,7); ytr=np.concatenate([ytr,ytr])   # ×2 (省メモリ)
        st=Std().fit(Xtr); Xz=st(Xtr).astype(np.float32)
        print(f"\n########## 32cls band {lo}-{hi}Hz D={D} train={Xz.shape[0]} ##########", flush=True)
        for K in (64,128,256):
            rng=np.random.default_rng(0)
            alpha=(rng.standard_normal((D,K))/np.sqrt(D)).astype(np.float32)
            bias=(rng.standard_normal(K)*0.1).astype(np.float32)
            H=_act("hard_sigmoid", Xz@alpha+bias).astype(np.float32); betas={}
            for c in classes:
                Hc=H[ytr==c]; Xc=Xz[ytr==c]
                betas[c]=np.linalg.solve((Hc.T@Hc+ridge*np.eye(K)).astype(np.float64), (Hc.T@Xc).astype(np.float64))
            def scores(Xq):
                Hq=_act("hard_sigmoid", Xq@alpha+bias)
                return np.stack([((Xq-Hq@betas[c].astype(np.float32))**2).mean(1) for c in classes],1)
            line=f"  K={K:>3}: "
            for es in EVAL_SETS:
                z=build_run(es,baseline_of(es),"bl_self"); Xe,_=cropb(z[0],lo,hi); Xe=st(Xe).astype(np.float32)
                S=scores(Xe); y32e=z[2]
                strict=vote(S,y32e,y32e,classes)
                p14=collapse14(np.array(classes)[S.argmin(1)]); t14=collapse14(y32e)
                line+=f"{es.split('_')[-1]}:strict{strict:.0%}/14c{accuracy(t14,p14):.0%}  "
            print(line, flush=True)

if __name__=="__main__":
    main()
