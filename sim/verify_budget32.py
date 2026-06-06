"""RAMノード予算(入力+隠れ+出力≤570)内での32cls再校正精度を検証 + パラメータ数比較。

deployable Solist-AI構成: 識別的単一モデル(出力32 one-hot), α乱数固定, β現場再校正。
予算: D+K+32 ≤ 570。
  single  D=167 → K≤371 (K=256 OK)
  multiwin D=412 → K≤126 (K=256は予算超過)
各eval条件内で N-shot/状態でβ再校正→残り評価 (within-env, リークなし)。
"""
from __future__ import annotations
import numpy as np
from rich_feat32 import feat_single, feat_multiwin, build
from common import class_idx_14
from solist_elm import _act, accuracy

EVALS=["eval_quiet","eval_noise_low","eval_noise_high"]
def collapse14(a): return np.array([class_idx_14(np.array([(v>>k)&1 for k in range(5)])) for v in a])

def recal(X,y32,D,K,N,seed=0):
    mu=X.mean(0);sd=X.std(0)+1e-6;Xz=(X-mu)/sd
    rng=np.random.default_rng(seed); a=rng.standard_normal((D,K))/np.sqrt(D); b=rng.standard_normal(K)*0.1
    H=_act("hard_sigmoid",Xz@a+b)
    calib=np.zeros(len(Xz),bool)
    for s in np.unique(y32):
        idx=np.where(y32==s)[0]; calib[idx[:N]]=True
    test=~calib
    Y=np.zeros((calib.sum(),32)); Y[np.arange(calib.sum()),y32[calib]]=1
    beta=np.linalg.solve(H[calib].T@H[calib]+1e-1*np.eye(K),H[calib].T@Y)
    dec=H[test]@beta; yt=y32[test]
    strict=accuracy(yt,dec.argmax(1)); cor=tot=0
    for s in np.unique(yt):
        idx=np.where(yt==s)[0];cor+=int(dec[idx].sum(0).argmax()==s);tot+=1
    return strict,cor/tot,accuracy(collapse14(dec.argmax(1)),collapse14(yt))

def main():
    configs=[("single",feat_single,167,256),
             ("multiwin",feat_multiwin,412,126),
             ("multiwin(参考:予算超)",feat_multiwin,412,256)]
    for name,fn,D,K in configs:
        node=D+K+32; ok="OK" if node<=570 else "超過"
        print(f"\n=== {name} D={D} K={K}  ノード和={node} (≤570:{ok}) ===")
        print(f"  {'eval':16} {'N=3 strict/vote/14c':>22} {'N=5 strict/vote/14c':>22}")
        for es in EVALS:
            tag="single" if name.startswith("single") else "multiwin"
            X,y32=build(es,fn,tag)
            r3=recal(X,y32,D,K,3); r5=recal(X,y32,D,K,5)
            print(f"  {es:16} {r3[0]:>6.1%}/{r3[1]:.0%}/{r3[2]:.0%}      {r5[0]:>6.1%}/{r5[1]:.0%}/{r5[2]:.0%}")

if __name__=="__main__":
    main()
