"""Solist-AI と同一構成の忠実モデル + 14cls/32cls シミュレーション。

doc 準拠 (AnomalyDetection_an §3.1.2/§3.3, Sim教師あり版QS):
  - 3層NN: input → hidden → output
  - α: 一様乱数 uniform[-scaleAlpha, scaleAlpha] (ODL_GenerateRandomNumber, seed)
  - バイアス無し: h = G(x·α)   (ESN式 γ=0,δ=1 の通常3層)
  - 活性化 G = hard sigmoid: max(0, min(1, 0.2x+0.5))  (ODL_ACTV_SIGMOID, default)
  - β: ridge 最小二乗 (l2Param)。教師あり分類は target=one-hot, 推論 argmax(h·β)
  - bfloat16 演算 (実機精度エミュ)
  - 前処理: Normalize (per-dim z-score)
構成: 出力=入力(異常)でなく、出力=クラス数(教師あり分類)。input=single特徴 D=167。
評価: factory(train v6-12) cross-env eval + 現場β再校正。
"""
from __future__ import annotations
import numpy as np
from rich_feat32 import feat_single, build
from common import class_idx_14
from solist_elm import accuracy

TRAIN=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11,12)]
EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]

def bf16(x):
    """float32 → bfloat16 丸め → float32 (実機精度エミュ)。"""
    x=np.asarray(x,dtype=np.float32); u=x.view(np.uint32)
    u=((u + 0x8000) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)

def hard_sigmoid(z): return np.clip(0.2*z+0.5, 0.0, 1.0)

class SolistAI:
    """Solist-AI 忠実モデル (3層, α一様乱数固定/バイアス無/hard sigmoid/β=ridge LS)。"""
    def __init__(self, D, K, M, scale_alpha=0.2, l2=1e-1, seed=1, use_bf16=True):
        self.D,self.K,self.M,self.l2,self.bf=D,K,M,l2,use_bf16
        rng=np.random.default_rng(seed)
        a=rng.uniform(-scale_alpha, scale_alpha, (D,K)).astype(np.float32)
        self.alpha=bf16(a) if use_bf16 else a
        self.beta=None
    def hidden(self, X):
        X=bf16(X) if self.bf else X
        z=X@self.alpha
        if self.bf: z=bf16(z)
        h=hard_sigmoid(z)
        return bf16(h) if self.bf else h
    def fit(self, X, y):
        H=self.hidden(X).astype(np.float64)
        Y=np.zeros((len(y),self.M)); Y[np.arange(len(y)),y]=1.0
        beta=np.linalg.solve(H.T@H+self.l2*np.eye(self.K), H.T@Y)
        self.beta=bf16(beta) if self.bf else beta.astype(np.float32)
    def decision(self, X):
        H=self.hidden(X)
        d=H@(self.beta.astype(np.float32))
        return bf16(d) if self.bf else d
    def predict(self, X): return self.decision(X).argmax(1)

class Norm:
    def fit(s,X): s.mu=X.mean(0);s.sd=X.std(0)+1e-6;return s
    def __call__(s,X): return ((X-s.mu)/s.sd).astype(np.float32)

def collapse14(a): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in a])

def votescore(dec, y, y32):
    cor=tot=0
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]; cor+=int(dec[idx].sum(0).argmax()==(y[idx[0]])); tot+=1
    return cor/tot

def load(runs, tag="single"):
    X=[];y32=[]
    for r in runs: a,b=build(r,feat_single,tag); X.append(a);y32.append(b)
    return np.concatenate(X), np.concatenate(y32)

def main():
    Xtr,y32tr=load(TRAIN);
    evals={es:load([es]) for es in EVALS}
    D=Xtr.shape[1]
    print(f"Solist-AI忠実モデル: 入力single D={D}, α=一様乱数, hard_sigmoid, β=ridge LS, bfloat16")

    for ncls,tag in ((14,"14cls"),(32,"32cls")):
        ytr = collapse14(y32tr) if ncls==14 else y32tr
        print(f"\n########## {tag} ##########")
        for K in (64,128,256):
            node=D+K+ncls
            nm=Norm().fit(Xtr); Xn=nm(Xtr)
            m=SolistAI(D,K,ncls,scale_alpha=0.2,l2=1e-1,seed=1,use_bf16=True)
            m.fit(Xn,ytr)
            print(f"  --- K={K} (node和={node}{' 超過' if node>570 else ''}) ---")
            for es,(Xe,y32e) in evals.items():
                ye = collapse14(y32e) if ncls==14 else y32e
                Xen=nm(Xe)
                # factory
                dec=m.decision(Xen); f_s=accuracy(ye,dec.argmax(1)); f_v=votescore(dec,ye,y32e)
                # 現場β再校正 N=5 (α固定, β再学習, within-env)
                Hh=m.hidden(Xen).astype(np.float64)
                calib=np.zeros(len(Xen),bool)
                for s in np.unique(y32e):
                    idx=np.where(y32e==s)[0]; calib[idx[:5]]=True
                test=~calib
                Y=np.zeros((calib.sum(),ncls)); Y[np.arange(calib.sum()),ye[calib]]=1
                bb=np.linalg.solve(Hh[calib].T@Hh[calib]+0.1*np.eye(K),Hh[calib].T@Y)
                rd=Hh[test]@bb; r_s=accuracy(ye[test],rd.argmax(1)); r_v=votescore(rd,ye[test],y32e[test])
                print(f"    {es:16}: factory {f_s:.0%}/v{f_v:.0%}   recalN5 {r_s:.0%}/v{r_v:.0%}")

if __name__=="__main__":
    main()
