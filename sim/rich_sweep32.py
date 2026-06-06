"""32cls: 時間分割W × 帯域の掃引で最大化 (≤512制約内)。best を eval セットで検証。

multiwin が single を大きく上回った(74→83.5%)。W(時間窓数)を増やすと時間分解能↑だが
≤512に収めるため帯域/周波数分解能を下げる必要(トレードオフ)。各設定で cross-run
v6-11→v12 を測り、best を eval(quiet/noise) でも検証。識別的ELM K=256。
"""
from __future__ import annotations
import wave, numpy as np
from pathlib import Path
from common import CAPTURES_ROOT, CACHE_DIR, parse_state_label
from solist_elm import _act, accuracy

RUNS=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11,12)]
EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]

def load(p):
    with wave.open(str(p),'rb') as w: raw=w.readframes(w.getnframes())
    return np.frombuffer(raw,dtype=np.int16).astype(np.float64)/32768.0
def baseline(run):
    bw=sorted((CAPTURES_ROOT/run/"s00000").glob("frame_*.wav")); L=len(load(bw[0]))
    return np.mean(np.stack([load(p)[:L] for p in bw]),0)
def spec(d,nfft,lo,hi):
    win=np.hanning(nfft); segs=[np.abs(np.fft.rfft(d[s:s+nfft]*win)) for s in range(0,len(d)-nfft+1,nfft//2)]
    mag=np.mean(segs,0)[1:]; db=np.maximum(20*np.log10(mag+1e-9),-80.0)
    b=16000/nfft; l=max(0,int(round(lo/b))-1); h=int(round(hi/b)); return db[l:h].astype(np.float32)
def feat(d,W,nfft,lo,hi):
    n=len(d); seg=n//W; return np.concatenate([spec(d[w*seg:(w+1)*seg],nfft,lo,hi) for w in range(W)])

def build(run,cfg):
    W,nfft,lo,hi=cfg; key=f"W{W}_{nfft}_{lo}_{hi}"
    f=CACHE_DIR/f"sw_{run}_{key}.npz"
    if f.exists(): z=np.load(f); return z["X"],z["y32"]
    bl=baseline(run); L=len(bl); X=[];y=[]
    for sd in sorted((CAPTURES_ROOT/run).iterdir()):
        bits=parse_state_label(sd.name)
        if bits is None: continue
        c32=int(sum(int(b)<<k for k,b in enumerate(bits)))
        for wav in sorted(sd.glob("frame_*.wav")):
            a=load(wav); X.append(feat(a[:L]-bl[:L],W,nfft,lo,hi)); y.append(c32)
    X=np.stack(X);y=np.array(y); np.savez_compressed(f,X=X,y32=y); return X,y

class Std:
    def fit(s,X): s.mu=X.mean(0);s.sd=X.std(0)+1e-6;return s
    def __call__(s,X): return (X-s.mu)/s.sd

def train_sup(Xtr,ytr,K):
    st=Std().fit(Xtr); Xz=st(Xtr); D=Xz.shape[1]
    rng=np.random.default_rng(0); a=rng.standard_normal((D,K))/np.sqrt(D); b=rng.standard_normal(K)*0.1
    H=_act("hard_sigmoid",Xz@a+b); Y=np.zeros((len(ytr),32)); Y[np.arange(len(ytr)),ytr]=1
    beta=np.linalg.solve(H.T@H+1e-1*np.eye(K),H.T@Y)
    return (st,a,b,beta)
def predict(model,X):
    st,a,b,beta=model; return _act("hard_sigmoid",st(X)@a+b)@beta
def score(model,X,y):
    dec=predict(model,X); strict=accuracy(y,dec.argmax(1))
    cor=tot=0
    for s in np.unique(y):
        idx=np.where(y==s)[0]; cor+=int(dec[idx].sum(0).argmax()==s); tot+=1
    return strict,cor/tot

def main():
    cfgs=[(1,1024,400,3000),(2,512,300,3500),(4,512,300,3500),
          (6,512,400,2800),(8,512,500,2500),(4,1024,300,3000)]
    best=None
    for cfg in cfgs:
        Xtr=[];ytr=[];Xte=None
        for run in RUNS:
            X,y=build(run,cfg)
            if run=="full_32_train_v12": Xte,yte=X,y
            else: Xtr.append(X);ytr.append(y)
        Xtr=np.concatenate(Xtr);ytr=np.concatenate(ytr); D=Xtr.shape[1]
        m=train_sup(Xtr,ytr,256); st,sv=score(m,Xte,yte)
        flag=""
        if best is None or sv>best[1]: best=(cfg,sv); flag=" *"
        print(f"  cfg W={cfg[0]} nfft={cfg[1]} {cfg[2]}-{cfg[3]}Hz D={D:>3}: v12 strict={st:.1%} vote={sv:.1%}{flag}",flush=True)
    # best cfg を eval セットで検証 (全RUN学習)
    cfg=best[0]; print(f"\nbest cfg={cfg} → eval検証 (train v6-12 全部)")
    Xtr=[];ytr=[]
    for run in RUNS:
        X,y=build(run,cfg); Xtr.append(X);ytr.append(y)
    m=train_sup(np.concatenate(Xtr),np.concatenate(ytr),256)
    for es in EVALS:
        X,y=build(es,cfg); st,sv=score(m,X,y)
        print(f"  {es:16}: strict={st:.1%} vote={sv:.1%}")

if __name__=="__main__":
    main()
