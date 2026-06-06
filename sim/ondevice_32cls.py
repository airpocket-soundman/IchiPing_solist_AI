"""オンデバイス学習相当: 同一環境(session)で少数フレーム校正→同環境の残りでテスト。
16KB枠に収まる m=32/64 で 32cls 実力を測る。RLS(OS-ELM forget=1.0)とbatch-LS上限を比較。"""
import sys; sys.path.insert(0,'sim')
import numpy as np
from bench_v612 import build_run, baseline_of, crop
from solist_elm import _act, accuracy

LO,HI=400,3000; OUT=[]
def log(s): OUT.append(s); print(s,flush=True)

def onehot(y,C):
    M=np.zeros((len(y),C)); M[np.arange(len(y)),y]=1; return M

def rls(H,Y,m,forget=1.0,lam=0.1):
    beta=np.zeros((m,Y.shape[1])); P=np.eye(m)/lam
    for i in range(len(H)):
        h=H[i]; Ph=P@h; k=Ph/(forget+h@Ph)
        beta=beta+np.outer(k,(Y[i]-beta.T@h)); P=(P-np.outer(k,Ph))/forget
    return beta

def evalacc(Xte,yte,al,beta,s):
    P=_act('hard_sigmoid',(Xte*s)@al)@beta; pred=P.argmax(1)
    fa=accuracy(yte,pred)
    # per-class block vote
    cor=tot=0
    for c in np.unique(yte):
        idx=np.where(yte==c)[0]
        if len(idx)==0: continue
        cor+=int(P[idx].sum(0).argmax()==c); tot+=1
    return fa, cor/tot

for sess in ("full_32_train_v12","full_32_train_v6"):
    X,_,y32=build_run(sess, baseline_of(sess), f"od_{sess[-3:]}")
    X,D=crop(X,LO,HI); y=y32
    cls=np.unique(y); 
    log(f"\n=== {sess}: X{X.shape} D={D} classes={len(cls)} frames/cls~{np.bincount(y).min()}-{np.bincount(y).max()} ===")
    rng=np.random.default_rng(0)
    for m in (32,64):
        al=np.random.default_rng(1).uniform(-0.2,0.2,(D,m))
        for ncal in (10,30):
            # per-class split: ncal calib, rest test (random)
            tri=[]; tei=[]
            for c in cls:
                ci=np.where(y==c)[0]; rng.shuffle(ci)
                tri+=list(ci[:ncal]); tei+=list(ci[ncal:])
            tri=np.array(tri); tei=np.array(tei)
            mu=X[tri].mean(0); sd=X[tri].std(0)+1e-6
            Xtr=(X[tri]-mu)/sd; Xte=(X[tei]-mu)/sd
            Ytr=onehot(y[tri],32)
            for s in (1.0,0.5):
                H=_act('hard_sigmoid',(Xtr*s)@al)
                # batch-LS 上限
                bls=np.linalg.solve(H.T@H+0.1*np.eye(m),H.T@Ytr)
                fb,vb=evalacc(Xte,y[tei],al,bls,s)
                # RLS (OS-ELM)
                bo=rls(H,Ytr,m,forget=1.0,lam=0.1)
                fo,vo=evalacc(Xte,y[tei],al,bo,s)
                log(f" m={m} ncal={ncal:>2}/cls s={s}: batchLS frame={fb:.1%} vote={vb:.1%} | OS-ELM frame={fo:.1%} vote={vo:.1%}")
open('sim_export/_ondevice.txt','w',encoding='utf-8').write("\n".join(OUT))
