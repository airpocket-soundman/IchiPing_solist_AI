"""Noise ablation for the best Stamp-only condition, with unchanged data."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent.parent
base=root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json'
config=json.loads(base.read_text())
for noise in (.05,.15):
    name=f'stamp_shift0.01_base4_noise{noise}'
    out=base.parent.parent/name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): continue
    print('TRAINING',name,flush=True)
    command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),
        '--arch','small','--train','stamp','--stamp',*config['sessions'],
        '--val',config['validation'],'--seeds','3','--shift','0.01',
        '--xbase','4','--feature-noise-std',str(noise),'--tag',name,'--output-dir',str(out)]
    with (out/'training.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    updated=dict(config,name=name,feature_noise_std=noise)
    (out/'complete.json').write_text(json.dumps(updated,indent=2))
    print('SAVED',name,flush=True)
subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py')],cwd=root,check=True)
