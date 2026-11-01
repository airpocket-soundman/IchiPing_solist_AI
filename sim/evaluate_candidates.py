"""Score saved candidates on the frozen, training-excluded audio sessions."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
from eval_full_data import build_run
from emit_frontend_model import BLO, BHI, int_forward, elm_input, mcu_reference
from eval_stamp_session import class14

root = Path(__file__).resolve().parent.parent
folder = root/'sim_export/solist_ds/candidates_20261031'
rows=[]
parser=argparse.ArgumentParser()
parser.add_argument('--extra',nargs='*',default=[])
parser.add_argument('--sessions',nargs='+',default=[f'stamp_20261031_holdout_eval_20261031_s{i}_wav' for i in (1,2)])
parser.add_argument('--only-extra',action='store_true')
parser.add_argument('--report-name',default='holdout_evaluation')
args=parser.parse_args()
sessions=args.sessions
data={s:build_run(s) for s in sessions}
extra=[Path(p) for p in args.extra]
for directory in ([] if args.only_extra else sorted(folder.iterdir()))+extra:
    if not (directory/'complete.json').exists() and directory not in extra: continue
    config_path=directory/'complete.json'
    if not config_path.exists(): config_path=folder/'stamp_shift0.01_base4_noise0/complete.json'
    config=json.loads(config_path.read_text())
    assert not set(sessions).intersection(config['sessions'])
    model=next(directory.glob('*.npz')); z=np.load(model)
    layers=[dict(wq=z[f'conv{i}_wq'],M=z[f'conv{i}_M'],B=z[f'conv{i}_B'],k=int(z[f'conv{i}_k']),s=int(z[f'conv{i}_s'])) for i in range(3)]
    fc=dict(wq=z['fc_wq'],M=z['fc_M'],B=z['fc_B'])
    all_y=[]; all_p=[]; per={}
    for name,d in data.items():
        X=d['X'].astype(np.float32); y=d['y']
        lo=int(z['bin_lo']) if 'bin_lo' in z else BLO
        hi=int(z['bin_hi']) if 'bin_hi' in z else BHI
        q=np.clip(np.round(((X[:,lo:hi]-z['in_mu'])/z['in_sd'])/float(z['s_in'])),-127,127).astype(np.int8)
        E=int_forward(q,layers,fc)
        pred=mcu_reference(elm_input(E,z['emb_mul'],z['emb_add']),z['beta']).argmax(1)
        per[name]=dict(n=len(y),accuracy32=float(np.mean(pred==y)),accuracy14=float(np.mean([class14(int(a))==class14(int(b)) for a,b in zip(y,pred)])))
        all_y.extend(y.tolist()); all_p.extend(pred.tolist())
    matrix=np.zeros((32,32),dtype=int)
    for y,p in zip(all_y,all_p): matrix[y,p]+=1
    row=dict(name=directory.name,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),n=len(all_y),accuracy32=float(np.trace(matrix)/matrix.sum()),worst_session=min(v['accuracy32'] for v in per.values()),sessions=per,confusion=matrix.tolist(),config=config)
    rows.append(row)
    print(directory.name, f"{np.trace(matrix)}/{matrix.sum()} = {row['accuracy32']:.3%}",flush=True)
    (directory/(args.report_name+'.json')).write_text(json.dumps(row,indent=2))
rows.sort(key=lambda r:(r['accuracy32'],r['worst_session']),reverse=True)
(root/'docs/results'/('candidate_'+args.report_name+'_20261031.json')).write_text(json.dumps(rows,indent=2))
print('BEST',rows[0]['name'],rows[0]['accuracy32'],flush=True)
