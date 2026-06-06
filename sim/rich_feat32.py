"""32cls 改善: 入力特徴の作り込み (Solist-AI制約内=前処理側で勝負)。

Solist-AIのαは乱数固定(学習不可)なので、アクセラレータ側で学習特徴は持てない。
→ 漏洩音のサブ状態手がかりを「入力特徴」に明示的に出す。現状は2秒|FFT|平均で
時間構造を潰している。漏洩反射は時間構造に出るはずなので、時間分割(マルチウィンドウ)
特徴を試す。比較: 現状(単一平均) vs マルチウィンドウ。識別的ELM, cross-run v6-11→v12。
"""
from __future__ import annotations
import wave
import numpy as np
from pathlib import Path
from common import CAPTURES_ROOT, CACHE_DIR, parse_state_label
from solist_elm import _act, accuracy

RUNS=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11,12)]

def load(p):
    with wave.open(str(p),'rb') as w: raw=w.readframes(w.getnframes())
    return np.frombuffer(raw,dtype=np.int16).astype(np.float64)/32768.0

def baseline(run):
    bw=sorted((CAPTURES_ROOT/run/"s00000").glob("frame_*.wav")); L=len(load(bw[0]))
    return np.mean(np.stack([load(p)[:L] for p in bw]),0)

def spec(d, nfft, lo, hi):
    win=np.hanning(nfft); segs=[]
    for s in range(0,len(d)-nfft+1,nfft//2):
        segs.append(np.abs(np.fft.rfft(d[s:s+nfft]*win)))
    mag=np.mean(segs,0)[1:]
    db=np.maximum(20*np.log10(mag+1e-9),-80.0)
    b=16000/nfft; l=max(0,int(round(lo/b))-1); h=int(round(hi/b))
    return db[l:h].astype(np.float32)

def feat_single(d):  # 現状相当: NFFT1024, 2秒平均, 400-3000
    return spec(d,1024,400,3000)

def feat_multiwin(d, W=4, lo=300, hi=3500):  # 時間W分割, 各窓 NFFT512 のスペクトルを連結
    n=len(d); seg=n//W; out=[]
    for w in range(W):
        out.append(spec(d[w*seg:(w+1)*seg],512,lo,hi))
    return np.concatenate(out)

def build(run, fn, tag):
    f=CACHE_DIR/f"rich_{run}_{tag}.npz"
    if f.exists(): z=np.load(f); return z["X"],z["y32"]
    bl=baseline(run); X=[]; y=[]
    for sd in sorted((CAPTURES_ROOT/run).iterdir()):
        bits=parse_state_label(sd.name)
        if bits is None: continue
        c32=int(sum(int(b)<<k for k,b in enumerate(bits)))
        L=len(bl)
        for wav in sorted(sd.glob("frame_*.wav")):
            a=load(wav); X.append(fn(a[:L]-bl[:L])); y.append(c32)
    X=np.stack(X); y=np.array(y); np.savez_compressed(f,X=X,y32=y); return X,y

class Std:
    def fit(s,X): s.mu=X.mean(0); s.sd=X.std(0)+1e-6; return s
    def __call__(s,X): return (X-s.mu)/s.sd

def sup_eval(Xtr,ytr,Xte,yte,K,tag):
    st=Std().fit(Xtr); Xtr=st(Xtr); Xte=st(Xte); D=Xtr.shape[1]
    rng=np.random.default_rng(0); a=(rng.standard_normal((D,K))/np.sqrt(D)); b=rng.standard_normal(K)*0.1
    H=_act("hard_sigmoid",Xtr@a+b); Y=np.zeros((len(ytr),32)); Y[np.arange(len(ytr)),ytr]=1
    beta=np.linalg.solve(H.T@H+1e-1*np.eye(K),H.T@Y)
    dec=_act("hard_sigmoid",Xte@a+b)@beta
    strict=accuracy(yte,dec.argmax(1))
    cor=tot=0
    for s in np.unique(yte):
        idx=np.where(yte==s)[0]; cor+=int(dec[idx].sum(0).argmax()==s); tot+=1
    print(f"  {tag:14} D={D:>3} K={K}: strict(frame)={strict:.1%} vote={cor/tot:.1%}")

def main():
    for tag,fn in (("single",feat_single),("multiwin",feat_multiwin)):
        Xtr=[];ytr=[];Xte=None
        for run in RUNS:
            X,y=build(run,fn,tag)
            if run=="full_32_train_v12": Xte,yte=X,y
            else: Xtr.append(X); ytr.append(y)
        Xtr=np.concatenate(Xtr); ytr=np.concatenate(ytr)
        print(f"\n== feature={tag} (train v6-11={Xtr.shape}, test v12) ==")
        for K in (256,512):
            sup_eval(Xtr,ytr,Xte,yte,K,tag)

if __name__=="__main__":
    main()
