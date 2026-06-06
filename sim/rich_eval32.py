"""時間特徴(multiwin) + cross-baseline + strong-aug で eval セット汎化を検証 (32cls)。

multiwin は within-family で 92% だが eval(別session)で35%に崩壊(過学習)。
正則化(cross-baseline 3種 + 特徴強aug + 標準化)を入れて eval で保てるか確認。
デプロイ可能な W=4 nfft=512 300-3500 D=412 (≤512)。識別的ELM K=256。
"""
from __future__ import annotations
import wave, numpy as np
from common import CAPTURES_ROOT, CACHE_DIR, parse_state_label, class_idx_14
from solist_elm import _act, accuracy

TRAIN=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11,12)]
BASES=["full_32_train_v6","full_32_train_v9","full_32_train_v12"]
EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]
CFG=(4,512,300,3500)

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
def feat(d):
    W,nfft,lo,hi=CFG; n=len(d); seg=n//W
    return np.concatenate([spec(d[w*seg:(w+1)*seg],nfft,lo,hi) for w in range(W)])

def build(run, bl, tag):
    f=CACHE_DIR/f"re_{run}_{tag}.npz"
    if f.exists(): z=np.load(f); return z["X"],z["y14"],z["y32"]
    L=len(bl); X=[];y14=[];y32=[]
    for sd in sorted((CAPTURES_ROOT/run).iterdir()):
        bits=parse_state_label(sd.name)
        if bits is None: continue
        c14=class_idx_14(bits); c32=int(sum(int(b)<<k for k,b in enumerate(bits)))
        for wav in sorted(sd.glob("frame_*.wav")):
            a=load(wav); X.append(feat(a[:L]-bl[:L])); y14.append(c14); y32.append(c32)
    X=np.stack(X);y14=np.array(y14);y32=np.array(y32); np.savez_compressed(f,X=X,y14=y14,y32=y32); return X,y14,y32

class Std:
    def fit(s,X): s.mu=X.mean(0);s.sd=X.std(0)+1e-6;return s
    def __call__(s,X): return (X-s.mu)/s.sd
def aug(X,copies,seed,sigma=0.8,fm_w=40,fm_n=3,lvl=4.0):
    rng=np.random.default_rng(seed);out=[X];D=X.shape[1]
    for _ in range(copies):
        Xa=X+rng.normal(0,sigma,X.shape).astype(X.dtype); Xa=Xa+rng.uniform(-lvl,lvl,(len(Xa),1)).astype(X.dtype)
        for _m in range(fm_n):
            w=int(rng.integers(1,fm_w));s=int(rng.integers(0,max(1,D-w)));Xa[:,s:s+w]=0.0
        out.append(Xa)
    return np.concatenate(out)
def collapse14(arr): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in arr])

def main():
    bls={r:baseline(r) for r in BASES}
    Xtr=[];y32tr=[]
    for run in TRAIN:
        for br,bl in bls.items():
            X,_,y32=build(run,bl,f"bl_{br[-3:]}"); Xtr.append(X); y32tr.append(y32)
    Xtr=np.concatenate(Xtr);y32tr=np.concatenate(y32tr)
    Xtr=aug(Xtr,1,7); y32tr=np.concatenate([y32tr,y32tr])
    st=Std().fit(Xtr); Xz=st(Xtr); D=Xz.shape[1]
    print(f"multiwin+crossbaseline+aug: train={Xz.shape}")
    K=256; rng=np.random.default_rng(0); a=rng.standard_normal((D,K))/np.sqrt(D); b=rng.standard_normal(K)*0.1
    H=_act("hard_sigmoid",Xz@a+b); Y=np.zeros((len(y32tr),32)); Y[np.arange(len(y32tr)),y32tr]=1
    beta=np.linalg.solve(H.T@H+1e-1*np.eye(K),H.T@Y)
    def dec(X): return _act("hard_sigmoid",st(X)@a+b)@beta
    print(f"\n=== 32cls multiwin(D={D}) 正則化版, eval検証 ===")
    for es in EVALS:
        X,_,y32=build(es,baseline(es),"bl_self"); d=dec(X)
        strict=accuracy(y32,d.argmax(1))
        cor=tot=0
        for s in np.unique(y32):
            idx=np.where(y32==s)[0];cor+=int(d[idx].sum(0).argmax()==s);tot+=1
        p14=collapse14(d.argmax(1));t14=collapse14(y32)
        print(f"  {es:16}: strict={strict:.1%} vote={cor/tot:.1%}  14collapse={accuracy(t14,p14):.1%}")

if __name__=="__main__":
    main()
