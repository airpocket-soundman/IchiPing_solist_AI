"""IchiPing 精度向上の2原理を実IchiPingデータで検証。

検証する原理(Solist-AI移植で得た知見の逆輸入):
  A. 時間構造の特徴化: 2秒平均(single)を時間分割(multiwin)に → in-distribution向上
  B. 環境ごとの最終層 再校正(現場校正): backbone凍結 + headをN-shotで再学習 → cross-env回復

モデル: 学習特徴を持つ代表として2層MLP(ELMと違いαも学習)。実IchiPingデータ。
注: IchiPing本番CNN(neutron)そのものでなく代表モデルでの原理検証。本番への適用は次段。

protocol:
  A: train v6-11 → test v12 (in-family cross-run), 32cls strict/vote/14collapse, single vs multiwin
  B: train v6-12 → eval_quiet/noise_low/noise_high (cross-env). factory head vs N-shot 再校正head。
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from rich_feat32 import feat_single, feat_multiwin, build
from common import class_idx_14
from solist_elm import accuracy

TRAIN=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
ALL=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11,12)]
EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]
torch.manual_seed(0); np.random.seed(0)

def collapse14(a): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in a])

def get(run,tag):
    fn=feat_single if tag=="single" else feat_multiwin
    return build(run,fn,tag)

class MLP(nn.Module):
    def __init__(s,D,H=256,C=32):
        super().__init__(); s.f1=nn.Linear(D,H); s.f2=nn.Linear(H,H); s.head=nn.Linear(H,C); s.dp=nn.Dropout(0.3); s.act=nn.ReLU()
    def embed(s,x): return s.act(s.f2(s.dp(s.act(s.f1(x)))))
    def forward(s,x): return s.head(s.dp(s.embed(x)))

def train(Xtr,ytr,D,epochs=60):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xn=(Xtr-mu)/sd
    m=MLP(D); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss()
    Xt=torch.tensor(Xn,dtype=torch.float32); yt=torch.tensor(ytr)
    n=len(Xt)
    for ep in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,128):
            idx=perm[i:i+128]; opt.zero_grad(); loss=lossf(m(Xt[idx]),yt[idx]); loss.backward(); opt.step()
    m.eval(); return m,mu,sd

def evalacc(m,mu,sd,X,y):
    with torch.no_grad():
        logit=m(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()
    strict=accuracy(y,logit.argmax(1))
    cor=tot=0
    for s in np.unique(y):
        idx=np.where(y==s)[0]; cor+=int(logit[idx].sum(0).argmax()==s); tot+=1
    p14=collapse14(logit.argmax(1)); t14=collapse14(y)
    return strict,cor/tot,accuracy(t14,p14)

def recal_head(m,mu,sd,X,y,N):
    """backbone凍結, head(線形)を N-shot/state でLS再学習し残りで評価。"""
    with torch.no_grad():
        E=m.embed(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()  # penultimate
    calib=np.zeros(len(X),bool)
    for s in np.unique(y):
        idx=np.where(y==s)[0]; calib[idx[:N]]=True
    test=~calib
    Y=np.zeros((calib.sum(),32)); Y[np.arange(calib.sum()),y[calib]]=1
    Hc=E[calib]; W=np.linalg.solve(Hc.T@Hc+1e-1*np.eye(Hc.shape[1]), Hc.T@Y)
    dec=E[test]@W; yt=y[test]
    strict=accuracy(yt,dec.argmax(1))
    cor=tot=0
    for s in np.unique(yt):
        idx=np.where(yt==s)[0]; cor+=int(dec[idx].sum(0).argmax()==s); tot+=1
    return strict,cor/tot,accuracy(collapse14(dec.argmax(1)),collapse14(yt))

def main():
    print("######## 実験A: 時間特徴(single vs multiwin), in-family v6-11→v12 ########")
    for tag in ("single","multiwin"):
        Xtr=[];ytr=[]
        for r in TRAIN: X,y=get(r,tag); Xtr.append(X);ytr.append(y)
        Xtr=np.concatenate(Xtr);ytr=np.concatenate(ytr); D=Xtr.shape[1]
        Xte,yte=get("full_32_train_v12",tag)
        m,mu,sd=train(Xtr,ytr,D)
        st,sv,c14=evalacc(m,mu,sd,Xte,yte)
        print(f"  {tag:9} D={D:>3}: 32cls strict={st:.1%} vote={sv:.1%}  14collapse={c14:.1%}")

    print("\n######## 実験B: 再校正head (train v6-12, cross-env eval) ########")
    for tag in ("single","multiwin"):
        Xtr=[];ytr=[]
        for r in ALL: X,y=get(r,tag); Xtr.append(X);ytr.append(y)
        Xtr=np.concatenate(Xtr);ytr=np.concatenate(ytr); D=Xtr.shape[1]
        m,mu,sd=train(Xtr,ytr,D)
        print(f"  --- feature={tag} (D={D}) ---")
        for es in EVALS:
            Xe,ye=get(es,tag)
            f_st,f_sv,f_c=evalacc(m,mu,sd,Xe,ye)               # factory head
            r3=recal_head(m,mu,sd,Xe,ye,3)                      # 再校正 N=3
            r5=recal_head(m,mu,sd,Xe,ye,5)
            print(f"    {es:16}: factory strict={f_st:.1%}/vote{f_sv:.1%}/14c{f_c:.1%}  "
                  f"recalN3 strict={r3[0]:.1%}/vote{r3[1]:.1%}  recalN5 vote={r5[1]:.1%}/14c{r5[2]:.1%}")

if __name__=="__main__":
    main()
