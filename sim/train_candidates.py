"""Train and retain candidates on a fixed allowlist, before holdout capture completes."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
sessions = ([f'stamp_20260926_s{i}_wav' for i in range(1,12)] +
            ['stamp_20260927_s1_wav','stamp_20260927_s2_wav',
             'stamp_20261030_speed500_s1_wav','stamp_20261030_speed500_s2_wav',
             'stamp_20261030_speed500_s3_retry_wav'])
conditions = ((.005,2,0),(.01,2,0),(.02,2,0),(.01,4,0),(.005,2,.05),(.005,2,.15),(.01,2,.05))
for data in ('unoq','stamp'):
    for shift,cross,noise in conditions:
        name=f'{data}_shift{shift}_base{cross}_noise{noise}'
        out=root/'sim_export/solist_ds/candidates_20261031'/name
        out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists(): continue
        print('TRAINING',name,flush=True)
        command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),
                 '--arch','small','--train',data,'--stamp',*sessions,
                 '--val',sessions[-1],'--seeds','3','--shift',str(shift),
                 '--xbase',str(cross),'--feature-noise-std',str(noise),
                 '--tag',name,'--output-dir',str(out)]
        with (out/'training.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        (out/'complete.json').write_text(json.dumps(dict(name=name,sessions=sessions,validation=sessions[-1],
            shift=shift,cross_baseline=cross,feature_noise_std=noise,data=data,seeds=3),indent=2))
        print('SAVED',name,flush=True)
