"""Remove only the targeted C capture; compare xbase 4/8/16 on five holdouts."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent.parent
folder=root/'sim_export/solist_ds/candidates_20261101'
source=folder/'lowband_all22_fixedbase_xbase8'
base=json.loads((source/'complete.json').read_text())
removed='stamp_20261031_hidden_c_targeted_20261031_s1_wav'
sessions=[s for s in base['sessions'] if s!=removed]
assert removed in base['sessions'] and len(sessions)==len(set(sessions))==21
evaluation=json.loads((folder/'cross_baseline_plan.json').read_text())['evaluation_sessions']+['stamp_20261101_holdout_eval_20261101_s1_wav']
assert not set(sessions).intersection(evaluation)
plan=folder/'full21_no_targeted_plan.json'
plan.write_text(json.dumps(dict(sessions=sessions,excluded_session=removed,baseline_reference=base['baseline_reference'],
                                cross_counts=[4,8,16],evaluation_sessions=evaluation),indent=2))
outputs=[]
try:
    for cross in (4,8,16):
        name=f'lowband_full21_no_targeted_fixedbase_xbase{cross}'
        target=folder/name
        target.mkdir(exist_ok=True)
        config=dict(base,name=name,sessions=sessions,cross_baseline=cross,source_model=str(source),excluded_session=removed)
        for field in ('completed','elapsed_seconds'):config.pop(field,None)
        (target/'plan.json').write_text(json.dumps(config,indent=2))
        if not (target/'complete.json').exists():
            print('TRAINING',name,flush=True)
            start=time.monotonic()
            command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),'--arch','small','--train','stamp',
                     '--stamp',*sessions,'--val-sessions',*config['validation'],'--seeds','3','--shift','0.01',
                     '--xbase',str(cross),'--input-bin-lo','12','--feature-noise-std','0',
                     '--baseline-reference',*config['baseline_reference'],'--tag',name,'--output-dir',str(target)]
            with (target/'training.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            config.update(elapsed_seconds=time.monotonic()-start,completed=dt.datetime.now().isoformat())
            (target/'complete.json').write_text(json.dumps(config,indent=2))
        outputs.append(target)
        comparisons=[folder/'lowband_multi_plus2_hidden_c_fixedbase_xbase16',folder/'lowband_all20_fixedbase_xbase8',source]+outputs
        subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),'--only-extra','--extra',*map(str,comparisons),
                        '--sessions',*evaluation,'--report-name','full21_no_targeted_comparison'],cwd=root,check=True)
        print('EVALUATED',name,flush=True)
    (folder/'full21_no_targeted_complete.json').write_text(json.dumps(dict(completed=dt.datetime.now().isoformat())))
except Exception as exc:
    (folder/'full21_no_targeted_failure.json').write_text(json.dumps(dict(error=str(exc),time=dt.datetime.now().isoformat())))
    raise
