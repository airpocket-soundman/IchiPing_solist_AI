"""Solist-AI Sim (教師あり版) 用 学習/テスト CSV を生成。

Sim形式: 1枚の表, 各行=1チャンク(1サンプル)。1行目ヘッダはSimが無視。
  列1..167  = 入力特徴 (FFT(audio-baseline) single D=167, 標準化済)
  列168..   = Expected one-hot (14 or 32列)
Sim設定: Input data First col=1 / rows=1 / cols=167,  Expected First col=168 / rows=1 / cols=C
シナリオ(実力検証=別セッション): 学習=v6-v11, テスト=v12。
標準化は学習(v6-11)統計で実施しテストにも同じ統計を適用。
"""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich_feat32 import feat_single, build
from common import class_idx_14

OUT = Path(__file__).resolve().parent.parent / "sim_export"
TR=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
TE=["full_32_train_v12"]

def lab(y32,C): return (np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in y32]) if C==14 else y32)
def onehot(y,C): M=np.zeros((len(y),C),int); M[np.arange(len(y)),y]=1; return M

def load(runs):
    X=[];y=[]
    for r in runs: a,b=build(r,feat_single,"single"); X.append(a);y.append(b)
    return np.concatenate(X), np.concatenate(y)

def write(path,X,y32,C,mu,sd):
    Xs=(X-mu)/sd
    head=[f"f{i+1}" for i in range(X.shape[1])]+[f"c{j}" for j in range(C)]
    mat=np.hstack([Xs, onehot(lab(y32,C),C).astype(float)])
    np.savetxt(path, mat, delimiter=",", fmt="%.6f", header=",".join(head), comments="")
    print(f"  {path.name}: {mat.shape[0]}行 × {mat.shape[1]}列 "
          f"(入力 列1-{X.shape[1]}, Expected 列{X.shape[1]+1}-{X.shape[1]+C})")

def main():
    OUT.mkdir(exist_ok=True)
    Xtr,y32tr=load(TR); Xte,y32te=load(TE)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    for C in (14,32):
        print(f"=== {C}cls (別セッション: 学習v6-11 / テストv12) ===")
        write(OUT/f"train_v6-11_{C}cls.csv", Xtr, y32tr, C, mu, sd)
        write(OUT/f"test_v12_{C}cls.csv",    Xte, y32te, C, mu, sd)
    print(f"\n出力先: {OUT}")
    print("Sim設定: [Input] First col=1, rows=1, cols=167 | [Expected] First col=168, rows=1, cols=14または32")

if __name__=="__main__":
    main()
