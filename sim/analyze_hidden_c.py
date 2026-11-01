"""Diagnose C-pair spectral separability using session-held-out statistics."""
import json
from pathlib import Path
import numpy as np
from eval_best_model import run_db, feats

root=Path(__file__).resolve().parent.parent
config=json.loads((root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json').read_text())
runs=config['sessions']
data={r:run_db(r) for r in runs}
hz=np.arange(1,1025)*16000/2048
bands={'below100':(7.8,100),'low_30_100':(30,100),'low_50_100':(50,100),'low_100_400':(100,400),'band_30_2700':(30,2700),'band_50_2700':(50,2700),'band_100_2700':(100,2700)}
label=lambda c:'h'+''.join(str((c>>k)&1) for k in (2,4,1,3,0))
report=[]
for closed in (3,10,17,19):
    opened=closed|4
    pair=dict(closed=label(closed),opened=label(opened),bands={})
    for representation in ('raw_db','normalized'):
        arrays={r:(d['db']-d['base'] if representation=='raw_db' else feats(d,d['base']).astype(np.float32)) for r,d in data.items()}
        A=np.concatenate([arrays[r][data[r]['y']==closed] for r in runs])
        B=np.concatenate([arrays[r][data[r]['y']==opened] for r in runs])
        effect=(B.mean(0)-A.mean(0))/np.sqrt((A.var(0)+B.var(0))*.5+1e-6)
        top=np.argsort(-np.abs(effect))[:12]
        pair[representation+'_top_bins']=[dict(hz=float(hz[k]),effect=float(effect[k])) for k in top]
        for band,(lo,hi) in bands.items():
            mask=(hz>=lo)&(hz<=hi)
            per=[]
            for held in runs:
                train=[r for r in runs if r!=held]
                a=np.concatenate([arrays[r][data[r]['y']==closed][:,mask] for r in train])
                b=np.concatenate([arrays[r][data[r]['y']==opened][:,mask] for r in train])
                mid=(a.mean(0)+b.mean(0))*.5
                w=(b.mean(0)-a.mean(0))/(.5*(a.var(0)+b.var(0))+1e-4)
                d=data[held]; idx=(d['y']==closed)|(d['y']==opened)
                pred=(arrays[held][idx][:,mask]-mid)@w>0
                per.append(dict(session=held,n=int(idx.sum()),accuracy=float(np.mean(pred==(d['y'][idx]==opened)))))
            pair['bands'][representation+'_'+band]=dict(mean=float(np.mean([v['accuracy'] for v in per])),worst=float(min(v['accuracy'] for v in per)),sessions=per)
    report.append(pair)
    print(pair['closed'],pair['opened'],[(k,round(v['mean'],3),round(v['worst'],3)) for k,v in pair['bands'].items()],flush=True)
out=root/'docs/results/hidden_c_separability_20261031.json'
out.write_text(json.dumps(dict(method='C-only diagnostic classifier; not 32-class model accuracy; leave-one-session-out',pairs=report),indent=2))
print('SAVED',out,flush=True)
