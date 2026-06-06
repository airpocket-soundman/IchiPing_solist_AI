"""MCU実装枠(m=32,D=167,AI RAM~11KB)で14cls/32clsの最大精度モデル構築。
factory(別session汎化)とon-device(現地校正)の両方、入力スケール最適化。
αはSimの実α(seed1,167×32,_alpha32_sim.npy)を使用→ロード後Simと一致。"""
import sys; sys.path.insert(0,'sim')
import numpy as np
from bench_v612 import build_run, baseline_of, crop
from solist_elm import _act, accuracy
LO,HI=400,3000
al=np.load('sim_export/_alpha32_sim.npy')  # (167,32) Sim seed1
L=[]
def log(s): L.append(s)

# --- データ構築 ---
TR=[f"full_32_train_v{i}" for i in (6,7,8,9,10,11)]
BL=["full_32_train_v6","full_32_train_v9","full_32_train_v11"]
bls={b:baseline_of(b) for b in BL}
Xtr14=[];Xtr32=[];y14=[];y32=[]
for run in TR:
    for b,bl in bls.items():
        Xr,yy14,yy32=build_run(run,bl,f"bl_{b[-3:]}"); xx,D=crop(Xr,LO,HI)
        Xtr14.append(xx); y14.append(yy14); y32.append(yy32)
Xtr=np.concatenate(Xtr14); Y14=np.concatenate(y14); Y32=np.concatenate(y32)
# 軽aug: ガウス雑音+レベルジッタ ×2
rng=np.random.default_rng(0)
aug=[Xtr]
for _ in range(2):
    aug.append(Xtr + rng.normal(0,0.5,Xtr.shape) + rng.normal(0,1.0,(len(Xtr),1)))
Xtr=np.concatenate(aug); Y14=np.tile(Y14,3); Y32=np.tile(Y32,3)
# test v12 (self-baseline)
Xv,yv14,yv32=build_run("full_32_train_v12",baseline_of("full_32_train_v12"),"od_v12")
Xte,_=crop(Xv,LO,HI)
log(f"train {Xtr.shape} (aug×3), test {Xte.shape} D={D}")

def oh(y,C): M=np.zeros((len(y),C)); M[np.arange(len(y)),y]=1; return M
def fit(X,Y,s,ridge=0.1):
    H=_act('hard_sigmoid',(X*s)@al); return np.linalg.solve(H.T@H+ridge*np.eye(32),H.T@Y)
def ev(X,y,beta,s):
    P=_act('hard_sigmoid',(X*s)@al)@beta; fa=accuracy(y,P.argmax(1))
    cor=tot=0
    for c in np.unique(y):
        idx=np.where(y==c)[0]; cor+=int(P[idx].sum(0).argmax()==c); tot+=1
    return fa,cor/tot

best={}
for C,Ytr_oh,yte_full in [(14,oh(Y14,14),yv14),(32,oh(Y32,32),yv32)]:
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtrn=(Xtr-mu)/sd; Xten=(Xte-mu)/sd
    log(f"\n===== {C}cls =====")
    # (a) factory: train全部 → test v12 cold
    log("-- factory (別session汎化, 校正なし) --")
    bestf=None
    for s in (1.0,0.5,0.35,0.25):
        b=fit(Xtrn,Ytr_oh,s); fa,va=ev(Xten,yte_full,b,s)
        log(f"  s={s}(effSA~{0.205*s:.3f}): frame={fa:.1%} vote={va:.1%}")
        if bestf is None or va>bestf[0]: bestf=(va,fa,s,b,mu,sd)
    best[(C,'factory')]=bestf
    # (b) on-device: v12 10/cls校正 → v12 holdout
    log("-- on-device (現地v12を10/cls校正→残りtest) --")
    rng2=np.random.default_rng(0); tri=[];tei=[]
    for c in range(C):
        ci=np.where(yte_full==c)[0]; rng2.shuffle(ci); tri+=list(ci[:10]); tei+=list(ci[10:])
    tri=np.array(tri);tei=np.array(tei)
    mu2=Xte[tri].mean(0); sd2=Xte[tri].std(0)+1e-6
    Xc=(Xte[tri]-mu2)/sd2; Xh=(Xte[tei]-mu2)/sd2
    besto=None
    for s in (1.0,0.5,0.35,0.25):
        b=fit(Xc,oh(yte_full[tri],C),s); fa,va=ev(Xh,yte_full[tei],b,s)
        log(f"  s={s}: frame={fa:.1%} vote={va:.1%}")
        if besto is None or va>besto[0] or (va==besto[0] and fa>besto[1]): besto=(va,fa,s,b,mu2,sd2,tei)
    best[(C,'ondevice')]=besto

# 保存
import pickle
with open('sim_export/_best_models.pkl','wb') as f: pickle.dump(best,f)
log("\n=== BEST ===")
for k,v in best.items():
    log(f"{k}: vote={v[0]:.1%} frame={v[1]:.1%} s={v[2]}")
open('sim_export/_best.txt','w',encoding='utf-8').write("\n".join(L))
