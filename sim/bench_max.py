"""最大データ学習: 全baseline×全version の diff (7×7=49) + 強オーグメンテーション多重。

ユーザー指定:
  - v6..v12 の各 baseline で v6..v12 の全データを diff → 49 組合せ (full cross product)
  - それらに強度の強い種々の augmentation を組合せ、学習データ最大数で学習
augmentation (FFT(audio-baseline)特徴に妥当なもの):
  - 音声空間: GaussianHiss (SNR 0-20dB, 強)   ※diff = (x+noise)-base = signal_diff+noise で妥当
  - 特徴空間: SpectralJitter σ0.8dB / FreqMask(幅40×3) / LevelJitter(±4dB 加算)
  ※ TimeShift は time-domain diff のコヒーレンスを壊すので不使用
評価: eval_quiet/noise_low/noise_high を A=旧baseline(v6) と B=現地baseline で測定(工場β)。
標準化(train統計 per-dim z-score)を適用。Flash は β サイズで決まり学習数に依存しない。
"""
from __future__ import annotations
import numpy as np
from bench_v612 import (baseline_of, build_run, build_run_noisy, crop,
                        TRAIN_RUNS, EVAL_SETS, _act)
from solist_elm import accuracy
from common import CACHE_DIR

ALL_BASELINES = TRAIN_RUNS[:]   # v6..v12 全部


class Std:
    def fit(self,X): self.mu=X.mean(0); self.sd=X.std(0)+1e-6; return self
    def __call__(self,X): return (X-self.mu)/self.sd


def strong_feat_aug(X, copies, seed, sigma=0.8, fm_w=40, fm_n=3, lvl_db=4.0):
    rng=np.random.default_rng(seed); out=[X]; D=X.shape[1]
    for _ in range(copies):
        Xa=X + rng.normal(0,sigma,X.shape).astype(X.dtype)        # SpectralJitter
        Xa=Xa + rng.uniform(-lvl_db,lvl_db,(len(Xa),1)).astype(X.dtype)  # LevelJitter(加算dB)
        for _m in range(fm_n):                                     # FreqMask
            w=rng.integers(1,fm_w); s=rng.integers(0,max(1,D-w))
            Xa[:,s:s+w]=0.0
        out.append(Xa)
    return np.concatenate(out)


def vote_states(S,y,y32,classes,k=0):
    classes=np.array(classes); cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]
        groups=[idx] if k<=0 else [idx[i:i+k] for i in range(0,len(idx),k)]
        for g in groups: cor+=int(classes[S[g].sum(0).argmin()]==y[g][0]); tot+=1
    return cor/tot


def main():
    LO,HI=400,3000; ncls=14; classes=list(range(ncls)); ridge=1e-1
    bls={r:baseline_of(r) for r in ALL_BASELINES}

    # --- 49 組合せ clean diff ---
    print("[build] 49 baseline×version clean diffs ...", flush=True)
    Xc=[]; yc=[]
    for run in TRAIN_RUNS:
        for br,bl in bls.items():
            X,y14,_=build_run(run,bl,f"bl_{br[-3:]}"); xx,D=crop(X,LO,HI)
            Xc.append(xx); yc.append(y14)
    Xc=np.concatenate(Xc); yc=np.concatenate(yc)
    print(f"  clean cross-product X={Xc.shape}")

    # --- 音声 GaussianHiss aug (7version × 全baseline, 強SNR) ---
    print("[build] audio GaussianHiss aug (×7 baselines) ...", flush=True)
    Xn=[]; yn=[]
    for i,run in enumerate(TRAIN_RUNS):
        for j,(br,bl) in enumerate(bls.items()):
            f=CACHE_DIR/f"max_noisy_{run}_{br[-3:]}.npz"
            if f.exists(): z=np.load(f); X,y14=z["X"],z["y14"]
            else:
                X,y14,_=build_run_noisy(run,bl,seed=1000+i*7+j,snr_db=(0,20))
                np.savez_compressed(f,X=X,y14=y14)
            xx,_=crop(X,LO,HI); Xn.append(xx); yn.append(y14)
    Xn=np.concatenate(Xn); yn=np.concatenate(yn)
    print(f"  audio-noise X={Xn.shape}")

    # --- 結合 + 特徴空間 強aug 多重 ---
    Xall=np.concatenate([Xc,Xn]); yall=np.concatenate([yc,yn])
    Xaug=strong_feat_aug(Xall,copies=2,seed=7)            # ×3
    yaug=np.concatenate([yall,yall,yall])
    print(f"  TOTAL train (max) X={Xaug.shape}  ({Xaug.shape[0]} frames)")

    st=Std().fit(Xaug); Xtr=st(Xaug)

    # --- 工場β (AE-bank) ---
    for K in (32, 48):
        rng=np.random.default_rng(0)
        alpha=rng.standard_normal((D,K))/np.sqrt(D); bias=rng.standard_normal(K)*0.1
        H=_act("hard_sigmoid", Xtr@alpha+bias); betas={}
        for c in classes:
            Hc=H[yaug==c]; Xt=Xtr[yaug==c]
            betas[c]=np.linalg.solve(Hc.T@Hc+ridge*np.eye(K), Hc.T@Xt)
        def scores(Xz):
            Hz=_act("hard_sigmoid", Xz@alpha+bias)
            return np.stack([((Xz-Hz@betas[c])**2).mean(1) for c in classes],1)
        print(f"\n=== 14cls 最大データ学習 K={K} D={D} (49cross×強aug, 標準化) ===")
        print(f"  {'eval':16} {'A:旧baseline':>12} {'B:現地baseline':>14}")
        for es in EVAL_SETS:
            Xa,ya,sa=build_run(es,bls['full_32_train_v6'],"bl_v6stale"); Xa,_=crop(Xa,LO,HI)
            Xb,yb,sb=build_run(es,baseline_of(es),"bl_self");           Xb,_=crop(Xb,LO,HI)
            A=vote_states(scores(st(Xa)),ya,sa,classes)
            B=vote_states(scores(st(Xb)),yb,sb,classes)
            print(f"  {es:16} {A:>11.1%} {B:>13.1%}")
        beta_i8=D*K*ncls; beta_bf=D*K*2*ncls
        print(f"  Flash β: int8={beta_i8/1024:.0f}KB / bf16={beta_bf/1024:.0f}KB "
              f"(+baseline2s 64KB, +fw~50KB)")


if __name__=="__main__":
    main()
