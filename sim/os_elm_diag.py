"""Sim再現診断: OS-ELM(逐次RLS+忘却) vs batch-LS、隠れ層collapse、事前学習β推論。

目的:
 1. SimのSolist-AIは OS-ELM(逐次RLS) で学習。これをPython実装し、私のbatch-LSと比較。
    → Simで学習しない原因が「逐次RLSの不安定/未収束」か「隠れ層の潰れ(αスケール)」か切り分け。
 2. 事前学習β(batch-LSで計算)をそのまま推論に使った場合の精度
    = 「factory β をMCUにロードして推論」する運用の到達精度。
データ: train_best_14cls.csv / test_best_14cls.csv (14cls均衡クリーン)。
"""
from __future__ import annotations
import numpy as np
from solist_elm import _act, accuracy

def load(p):
    a=np.loadtxt(p,delimiter=",",skiprows=1); return a[:,:167].astype(np.float64), a[:,167:]
Xtr,Ytr=load("sim_export/train_best_14cls.csv"); ytr=Ytr.argmax(1)
Xte,Yte=load("sim_export/test_best_14cls.csv"); yte=Yte.argmax(1)
D=Xtr.shape[1]; M=Ytr.shape[1]; N=len(Xtr)
print(f"data: train{Xtr.shape} test{Xte.shape} D={D} M={M}")

def make_alpha(scale,seed=1): return np.random.default_rng(seed).uniform(-scale,scale,(D,scale and 32 or 32)) # placeholder
def alpha_K(scale,K,seed=1): return np.random.default_rng(seed).uniform(-scale,scale,(D,K))

# ---------- 1. 隠れ層collapse 診断 (scaleAlpha sweep) ----------
print("\n=== 隠れ層collapse診断: scaleAlpha と hidden(hard_sigmoid)の分散 ===")
K=32
for sa in (0.01,0.05,0.1,0.2,0.5,1.0,2.0):
    a=alpha_K(sa,K); H=_act("hard_sigmoid",Xtr@a)
    print(f"  scaleAlpha={sa:>4}: hidden mean={H.mean():.3f} std={H.std():.4f} "
          f"(0/1飽和率={np.mean((H<=0)|(H>=1))*100:4.1f}%)  {'←潰れ' if H.std()<0.05 else ''}")

# ---------- 2. batch-LS 基準 (各scaleAlphaでの精度) ----------
print("\n=== batch-LS β でのv12精度 (scaleAlpha別, K=32) ===")
best=None
for sa in (0.05,0.1,0.2,0.5,1.0):
    a=alpha_K(sa,K); H=_act("hard_sigmoid",Xtr@a)
    beta=np.linalg.solve(H.T@H+0.1*np.eye(K),H.T@Ytr)
    acc=accuracy(yte,(_act("hard_sigmoid",Xte@a)@beta).argmax(1))
    print(f"  scaleAlpha={sa}: batch-LS v12 14cls acc={acc:.1%}")
    if best is None or acc>best[0]: best=(acc,sa)
SA=best[1]; print(f"  → 最良 scaleAlpha={SA} (acc={best[0]:.1%})")

# ---------- 3. OS-ELM (逐次RLS+忘却) 実装 ----------
def os_elm(X,Y,a,lam_init=0.1,forget=1.0,reps=1):
    K=a.shape[1]; H=_act("hard_sigmoid",X@a)
    beta=np.zeros((K,Y.shape[1])); P=np.eye(K)/lam_init   # P0=(λI)^-1
    maxb=[]
    for r in range(reps):
        for i in range(len(H)):
            h=H[i]; t=Y[i]
            Ph=P@h; denom=forget+h@Ph; k=Ph/denom
            beta=beta+np.outer(k,(t-beta.T@h))
            P=(P-np.outer(k,Ph))/forget
            if i%500==0: maxb.append(abs(beta).max())
    return beta,H,np.array(maxb)

print(f"\n=== OS-ELM(逐次RLS) vs batch-LS (scaleAlpha={SA}, K={K}) ===")
a=alpha_K(SA,K)
Hb=_act("hard_sigmoid",Xtr@a); beta_b=np.linalg.solve(Hb.T@Hb+0.1*np.eye(K),Hb.T@Ytr)
acc_b=accuracy(yte,(_act("hard_sigmoid",Xte@a)@beta_b).argmax(1))
print(f"  batch-LS:        v12 acc={acc_b:.1%}  max|β|={abs(beta_b).max():.2f}")
for forget,reps in ((1.0,1),(1.02,1),(1.0,5),(1.02,10),(0.999,1)):
    beta_o,_,maxb=os_elm(Xtr,Ytr,a,lam_init=0.1,forget=forget,reps=reps)
    acc_o=accuracy(yte,(_act("hard_sigmoid",Xte@a)@beta_o).argmax(1))
    div="発散" if (maxb.max()>1e3 or not np.isfinite(maxb.max())) else "安定"
    print(f"  OS-ELM forget={forget} reps={reps}: v12 acc={acc_o:.1%}  "
          f"max|β|={abs(beta_o).max():.2e} ({div})")

# ---------- 4. 事前学習β→推論 (factory β load 運用の到達点) ----------
print(f"\n=== 事前学習β(batch-LS)をロードして推論する運用 ===")
print(f"  scaleAlpha={SA}, K={K}: v12 14cls 推論精度 = {acc_b:.1%}")
print(f"  → αを同一(同seed/scale)に揃えれば、このβをODL_SetWeightBeta相当でロードし推論可能")
