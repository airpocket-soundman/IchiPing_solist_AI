"""Compare wider temperature-related shifts with the low-band best model."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent.parent
base=root/'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json'
config=json.loads(base.read_text())
validation=['stamp_20260927_s2_wav','stamp_20261030_speed500_s2_wav','stamp_20261030_speed500_s3_retry_wav']
for shift in (.02,.03):
 name=f'lowband_multi_shift{shift}'
 out=base.parent.parent/name
 out.mkdir(parents=True,exist_ok=True)
 if not (out/'complete.json').exists():
  cmd=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),'--arch','small','--train','stamp',
       '--stamp',*config['sessions'],'--val-sessions',*validation,'--seeds','3','--shift',str(shift),
       '--xbase','4','--input-bin-lo','12','--feature-noise-std','0','--tag',name,'--output-dir',str(out)]
  print('TRAINING',name,flush=True)
  with (out/'training.log').open('w',encoding='utf-8') as log:
   subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  (out/'complete.json').write_text(json.dumps(dict(config,name=name,shift=shift,validation=validation,bin_lo=12,bin_hi=346),indent=2))
  print('SAVED',name,flush=True)
 subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),'--extra',str(root/'sim_export/solist_ds/lowband_multival_20261031')],cwd=root,check=True)
print('COMPLETE: shift 1/2/3 percent comparison',flush=True)
