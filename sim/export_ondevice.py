"""公式Simでオンデバイス学習を実証するCSV生成: v12を校正(10/cls)とtest(残り)に分割、m=32想定。"""
import sys; sys.path.insert(0,'sim')
import numpy as np
from bench_v612 import build_run, baseline_of, crop
OUT="sim_export/"; LO,HI=400,3000
X,_,y=build_run("full_32_train_v12", baseline_of("full_32_train_v12"), "od_v12")
X,D=crop(X,LO,HI)
rng=np.random.default_rng(0); tri=[]; tei=[]
for c in range(32):
    ci=np.where(y==c)[0]; rng.shuffle(ci); tri+=list(ci[:10]); tei+=list(ci[10:])
tri=np.array(tri); tei=np.array(tei)
mu=X[tri].mean(0); sd=X[tri].std(0)+1e-6
def oh(yy): M=np.zeros((len(yy),32)); M[np.arange(len(yy)),yy]=1; return M
def w(p,idx):
    mat=np.hstack([(X[idx]-mu)/sd, oh(y[idx])])
    head=[f"f{i+1}" for i in range(D)]+[f"c{j}" for j in range(32)]
    np.savetxt(p,mat,delimiter=",",fmt="%.6f",header=",".join(head),comments="")
    return mat.shape
s1=w(OUT+"train_ondevice_v12_32cls.csv",tri)
s2=w(OUT+"test_ondevice_v12_32cls.csv",tei)
open(OUT+"_od.txt","w").write(f"train{s1} test{s2} D={D}")
