"""Controlled addition of full sessions and hidden-C data to low-band multival."""
import json
from pathlib import Path
import subprocess
import sys
import argparse
import time

parser=argparse.ArgumentParser()
parser.add_argument('--fixed-baselines',action='store_true')
parser.add_argument('--wait-for-current',action='store_true')
args=parser.parse_args()

root=Path(__file__).resolve().parent.parent
base=root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json'
config=json.loads(base.read_text())
if args.wait_for_current:
    completion=base.parent.parent/'lowband_multi_plus2_hidden_c/holdout_evaluation.json'
    deadline=time.monotonic()+7200
    while not completion.exists():
        if time.monotonic()>deadline: raise TimeoutError('Current experiment did not finish')
        time.sleep(10)
    time.sleep(5)
validation=['stamp_20260927_s2_wav','stamp_20261030_speed500_s2_wav','stamp_20261030_speed500_s3_retry_wav']
full=['stamp_20261031_current_train_20261031_s1_wav','stamp_20261031_current_train_20261031_s2_wav']
hidden='stamp_20261031_hidden_c_targeted_20261031_s1_wav'
for name, added in [('lowband_multi_plus2',full),('lowband_multi_plus2_hidden_c',full+[hidden])]:
    if args.fixed_baselines: name += '_fixedbase'
    sessions=config['sessions']+added
    for session in added:
        assert (root/'captures'/session/'session.json').exists(), session
    out=root/'sim_export/solist_ds/candidates_20261031'/name
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'complete.json').exists():
        command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),
            '--arch','small','--train','stamp','--stamp',*sessions,
            '--val-sessions',*validation,'--seeds','3','--shift','0.01','--xbase','4',
            '--input-bin-lo','12','--feature-noise-std','0','--tag',name,'--output-dir',str(out)]
        if args.fixed_baselines:
            command += ['--baseline-reference',*config['sessions']]
        print('TRAINING',name,flush=True)
        with (out/'training.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        (out/'complete.json').write_text(json.dumps(dict(config,name=name,sessions=sessions,validation=validation,bin_lo=12,bin_hi=346),indent=2))
        print('SAVED',name,flush=True)
    subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),
        '--extra',str(root/'sim_export/solist_ds/lowband_multival_20261031')],cwd=root,check=True)
print('COMPLETE: baseline, plus two full sessions, plus two full and hidden-C',flush=True)
