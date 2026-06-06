"""Solist-AI Sim 用 CSV を cross-baseline + augmentation 付きで生成。

特徴 = FFT(audio − baseline) (時間領域diff→FFTマグニチュード, 400-3000Hz, D=167)。
学習データ拡張:
  - cross-baseline: v6-v11 各frameを 複数baseline(v6,v9,v11) で diff → baseline選択に頑健化
  - augmentation: SpectralJitter σ0.8 + LevelJitter ±4dB + FreqMask(幅40×3) を +1コピー
テスト(v12)は自己baselineのみ・aug無し(評価データは拡張しない)。
標準化は学習データ統計。別セッション検証(学習v6-11 → テストv12)。
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_v612 import baseline_of, build_run, crop
from common import class_idx_14

OUT = Path(__file__).resolve().parent.parent / "sim_export"
TR=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
BL=["full_32_train_v6","full_32_train_v9","full_32_train_v11"]  # cross-baseline (学習セッション由来)
LO,HI=400,3000

def lab(y32,C): return (np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in y32]) if C==14 else y32)
def onehot(y,C): M=np.zeros((len(y),C),int); M[np.arange(len(y)),y]=1; return M

def feat_aug(X, seed, sigma=0.8, fm_w=40, fm_n=3, lvl=4.0):
    rng=np.random.default_rng(seed); D=X.shape[1]
    Xa=X+rng.normal(0,sigma,X.shape).astype(X.dtype)
    Xa=Xa+rng.uniform(-lvl,lvl,(len(Xa),1)).astype(X.dtype)
    for _ in range(fm_n):
        w=int(rng.integers(1,fm_w)); s=int(rng.integers(0,max(1,D-w))); Xa[:,s:s+w]=0.0
    return Xa

def write(path,X,y32,C,mu,sd):
    Xs=(X-mu)/sd
    head=[f"f{i+1}" for i in range(X.shape[1])]+[f"c{j}" for j in range(C)]
    mat=np.hstack([Xs, onehot(lab(y32,C),C).astype(float)])
    np.savetxt(path, mat, delimiter=",", fmt="%.6f", header=",".join(head), comments="")
    print(f"  {path.name}: {mat.shape[0]}行 × {mat.shape[1]}列 (入力1-{X.shape[1]}, Expected {X.shape[1]+1}-{X.shape[1]+C})")

def main():
    OUT.mkdir(exist_ok=True)
    # 学習: cross-baseline (v6-11 × {v6,v9,v11} baseline)
    bls={b:baseline_of(b) for b in BL}
    Xc=[]; yc=[]
    for run in TR:
        for b,bl in bls.items():
            X,_,y32=build_run(run,bl,f"bl_{b[-3:]}"); xx,D=crop(X,LO,HI)
            Xc.append(xx); yc.append(y32)
    Xc=np.concatenate(Xc); yc=np.concatenate(yc)
    # augmentation: +1コピー
    Xa=feat_aug(Xc,seed=7); Xtr=np.concatenate([Xc,Xa]); ytr=np.concatenate([yc,yc])
    print(f"学習(cross-baseline×{len(BL)} + aug): {Xtr.shape[0]}行 (元{len(Xc)}×2)")
    # テスト: v12 自己baseline, aug無し
    Xv,_,yv=build_run("full_32_train_v12", baseline_of("full_32_train_v12"), "bl_self_v12")
    Xte,_=crop(Xv,LO,HI)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    for C in (14,32):
        print(f"=== {C}cls ===")
        write(OUT/f"train_xbl_aug_v6-11_{C}cls.csv", Xtr, ytr, C, mu, sd)
        write(OUT/f"test_v12_{C}cls.csv",            Xte, yv,  C, mu, sd)
    print(f"\n出力: {OUT}")
    print("Sim設定: Input First col=1/rows=1/cols=167 | Expected First col=168/rows=1/cols=14(or32)")

if __name__=="__main__":
    main()
