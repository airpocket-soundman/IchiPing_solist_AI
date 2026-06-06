"""best modelsからロード用model1.xlsx + スケール済みtest CSVを生成(推論のみsim用)。"""
import sys; sys.path.insert(0,'sim')
import numpy as np, pickle, shutil, os, openpyxl
from bench_v612 import build_run, baseline_of, crop
LO,HI=400,3000
best=pickle.load(open('sim_export/_best_models.pkl','rb'))
Xv,yv14,yv32=build_run("full_32_train_v12",baseline_of("full_32_train_v12"),"od_v12")
Xte,D=crop(Xv,LO,HI)
labels={14:yv14,32:yv32}
TEMPLATE='sim_export/model1_260606_2005_41/model1.xlsx'
def oh(y,C):M=np.zeros((len(y),C));M[np.arange(len(y)),y]=1;return M
def emit_model(path,beta,no):
    os.makedirs(os.path.dirname(path),exist_ok=True); shutil.copy(TEMPLATE,path)
    wb=openpyxl.load_workbook(path)
    wb['Sd_ni_m_no'].cell(1,4).value=no   # no列
    ws=wb['beta']
    for i in range(32):
        for j in range(32):
            ws.cell(i+1,j+1).value=float(beta[i,j]) if j<no else 0.0
    wb.save(path)
def emit_test(path,X,y,mu,sd,s,no):
    Xn=((X-mu)/sd)*s
    mat=np.hstack([Xn, oh(y,no)])
    head=[f"f{i+1}" for i in range(D)]+[f"c{j}" for j in range(no)]
    np.savetxt(path,mat,delimiter=",",fmt="%.6f",header=",".join(head),comments="")
    return mat.shape
R=[]
# 14cls factory (s=0.35相当=best) : test=全v12
va,fa,s,b,mu,sd=best[(14,'factory')]
emit_model('sim_export/model1_BEST_14cls/model1.xlsx',b,14)
sh=emit_test('sim_export/test_BEST_14cls.csv',Xte,yv14,mu,sd,s,14)
R.append(f"14cls factory: s={s} model=model1_BEST_14cls/ test=test_BEST_14cls.csv {sh} 期待 frame{fa:.1%}/vote{va:.1%}")
# 32cls factory (校正前) : test=全v12
va,fa,s,b,mu,sd=best[(32,'factory')]
emit_model('sim_export/model1_BEST_32cls_factory/model1.xlsx',b,32)
sh=emit_test('sim_export/test_BEST_32cls_factory.csv',Xte,yv32,mu,sd,s,32)
R.append(f"32cls factory: s={s} model=model1_BEST_32cls_factory/ test=test_BEST_32cls_factory.csv {sh} 期待 frame{fa:.1%}/vote{va:.1%}")
# 32cls on-device (現地校正) : test=holdout, calib統計で標準化
va,fa,s,b,mu2,sd2,tei=best[(32,'ondevice')]
emit_model('sim_export/model1_BEST_32cls_ondevice/model1.xlsx',b,32)
sh=emit_test('sim_export/test_BEST_32cls_ondevice.csv',Xte[tei],yv32[tei],mu2,sd2,s,32)
R.append(f"32cls ondevice: s={s} model=model1_BEST_32cls_ondevice/ test=test_BEST_32cls_ondevice.csv {sh} 期待 frame{fa:.1%}/vote{va:.1%}")
open('sim_export/_emit.txt','w',encoding='utf-8').write("\n".join(R))
