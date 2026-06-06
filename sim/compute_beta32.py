"""Sim m=256 α(seed1)で広範学習データから最適32cls βを計算し、ロード用モデル生成。"""
import sys; sys.path.insert(0, 'sim')
import numpy as np, shutil, os, openpyxl
from solist_elm import _act, accuracy

al = np.load('sim_export/_alpha256.npy')   # (167,256), Sim seed=1
def load(p):
    a = np.loadtxt(p, delimiter=',', skiprows=1).astype(np.float32)
    return a[:, :167], a[:, 167:]
Xtr, Ytr = load('sim_export/train_xbl_aug_v6-11_32cls_24k.csv')  # 24000, cross-baseline+aug
Xte, Yte = load('sim_export/test_v12_32cls.csv'); yte = Yte.argmax(1)
print(f'train {Xtr.shape}, test {Xte.shape}', flush=True)

results = {}
for s in (1.0, 0.5):
    H = _act('hard_sigmoid', (Xtr*s) @ al).astype(np.float64)
    b = np.linalg.solve(H.T@H + 0.1*np.eye(256), H.T @ Ytr.astype(np.float64))
    pred = (_act('hard_sigmoid', (Xte*s) @ al) @ b).argmax(1)
    acc = accuracy(yte, pred)
    # 50-frame vote
    cor=tot=0
    for k in range(0, len(yte), 50):
        cor += int((_act('hard_sigmoid',(Xte[k:k+50]*s)@al)@b).sum(0).argmax()==yte[k]); tot+=1
    print(f'  input x{s} (eff~{0.2*s}): 32cls frame={acc:.1%} vote={cor/tot:.1%}', flush=True)
    results[s] = (acc, b)
    del H

best_s = max(results, key=lambda k: results[k][0])
acc, beta = results[best_s]
print(f'=> best input x{best_s}: {acc:.1%}', flush=True)
np.save('sim_export/_beta32_best.npy', beta)
np.save('sim_export/_scale32.npy', np.array([best_s]))

# ロード用モデル生成 (m=256, no=32, βを差し替え)
src = 'sim_export/model1_260606_1832_41/model1.xlsx'
dstdir = 'sim_export/model1_LOADABLE_32cls'; os.makedirs(dstdir, exist_ok=True)
dst = dstdir + '/model1.xlsx'; shutil.copy(src, dst)
wb = openpyxl.load_workbook(dst); ws = wb['beta']
for i in range(256):
    for j in range(32):
        ws.cell(i+1, j+1).value = float(beta[i, j])
wb.save(dst)
print(f'saved loadable model: {dst} (beta {beta.shape})', flush=True)

# スケール済みテストCSV (best_s != 1.0 のとき)
if best_s != 1.0:
    a = np.loadtxt('sim_export/test_v12_32cls.csv', delimiter=',', skiprows=1)
    a[:, :167] *= best_s
    head = [f'f{i+1}' for i in range(167)] + [f'c{j}' for j in range(32)]
    np.savetxt(f'sim_export/test_v12_32cls_x{best_s}.csv', a, delimiter=',', fmt='%.6f', header=','.join(head), comments='')
    print(f'saved scaled test: test_v12_32cls_x{best_s}.csv (これをタブ2に使う)', flush=True)
else:
    print('scale=1.0 なので既存 test_v12_32cls.csv をそのまま使用', flush=True)
