"""32cls multiwin + 環境内オンデバイスβ再校正。

時間特徴は環境固有だが環境内では一貫 → deployment環境で β を現場学習すれば
32clsも回復するはず(14clsの工場92%→現場100%と同理屈)。各eval条件内で N/状態だけ
βを学習(α乱数固定)→残りで評価。multiwin D=412(キャッシュ済 re_*_bl_self を利用)。
"""
from __future__ import annotations
import numpy as np
from common import CACHE_DIR, class_idx_14
from solist_elm import _act, accuracy

EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]
def collapse14(arr): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in arr])

def main():
    K=256; ridge=1e-1; rng=np.random.default_rng(0)
    print(f"32cls multiwin(D=412) 環境内β再校正 (α乱数固定, 識別的)")
    print(f"  {'eval':16} {'N/state':>7}  {'strict':>7} {'vote':>6} {'14collapse':>10}")
    for es in EVALS:
        z=np.load(CACHE_DIR/f"re_{es}_bl_self.npz"); X=z["X"].astype(np.float64); y32=z["y32"]
        mu=X.mean(0); sd=X.std(0)+1e-6; Xz=(X-mu)/sd; D=Xz.shape[1]
        a=rng.standard_normal((D,K))/np.sqrt(D); b=rng.standard_normal(K)*0.1
        H=_act("hard_sigmoid",Xz@a+b)
        for N in (1,2,3,5):
            calib=np.zeros(len(Xz),bool)
            for s in np.unique(y32):
                idx=np.where(y32==s)[0]; calib[idx[:N]]=True
            test=~calib
            if test.sum()==0: continue
            ytr=y32[calib]; Y=np.zeros((calib.sum(),32)); Y[np.arange(calib.sum()),ytr]=1
            beta=np.linalg.solve(H[calib].T@H[calib]+ridge*np.eye(K), H[calib].T@Y)
            dec=H[test]@beta; yt=y32[test]
            strict=accuracy(yt,dec.argmax(1))
            cor=tot=0
            for s in np.unique(yt):
                idx=np.where(yt==s)[0]; cor+=int(dec[idx].sum(0).argmax()==s); tot+=1
            p14=collapse14(dec.argmax(1)); t14=collapse14(yt)
            print(f"  {es:16} {N:>7}  {strict:>6.1%} {cor/tot:>5.1%} {accuracy(t14,p14):>9.1%}")

if __name__=="__main__":
    main()
