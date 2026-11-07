"""Add three new-house full sessions to the deployed 21-session cross16 model."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent.parent
source=root/'sim_export/solist_ds/candidates_20261101/lowband_full21_no_targeted_fixedbase_xbase16'
target=root/'sim_export/solist_ds/candidates_20261107/lowband_full24_house_revision_fixedbase_xbase16'
target.mkdir(parents=True,exist_ok=True)
new=[f'stamp_20261107_house_revision_train_20261107_s{i}_wav' for i in (1,2,3)]
eval_new='stamp_20261107_holdout_eval_house_revision_20261107_s1_wav'
evaluation=[
    'stamp_20261031_holdout_eval_20261031_s1_wav','stamp_20261031_holdout_eval_20261031_s2_wav',
    'stamp_20261031_holdout_eval_fresh_20261031_wav','stamp_20261031_holdout_eval_post_sweep_20261031_s1_wav',
    'stamp_20261101_holdout_eval_20261101_s1_wav',eval_new]

def wait_for(path):
    deadline=time.monotonic()+6*3600
    while not path.exists():
        capture=json.loads((root/'.local/house_revision_20261107_status.json').read_text(encoding='utf-8'))
        if capture['state']=='failed':raise RuntimeError(capture.get('error','Capture failed'))
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(10)
try:
    for i,session in enumerate(new,1):
        wait_for(root/'docs/results'/f'house_revision_train_20261107_s{i}_manifest.json')
        assert (root/'captures'/session/'session.json').exists()
    config=json.loads((source/'complete.json').read_text())
    sessions=config['sessions']+new
    assert len(sessions)==len(set(sessions))==24
    assert not set(sessions).intersection(evaluation)
    assert not any('hidden_c_targeted' in s for s in sessions)
    config.update(name=target.name,sessions=sessions,source_model=str(source),cross_baseline=16,
                  hardware_revision='house_revision_20261107',added_sessions=new)
    for field in ('elapsed_seconds','completed'):config.pop(field,None)
    (target/'plan.json').write_text(json.dumps(config,indent=2))
    if not (target/'complete.json').exists():
        start=time.monotonic()
        print('TRAINING',target.name,flush=True)
        command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),'--arch','small','--train','stamp',
                 '--stamp',*sessions,'--val-sessions',*config['validation'],'--seeds','3','--shift','0.01',
                 '--xbase','16','--input-bin-lo','12','--feature-noise-std','0',
                 '--baseline-reference',*config['baseline_reference'],'--tag',target.name,'--output-dir',str(target)]
        with (target/'training.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        config.update(completed=dt.datetime.now().isoformat(),elapsed_seconds=time.monotonic()-start)
        (target/'complete.json').write_text(json.dumps(config,indent=2))
    wait_for(root/'docs/results/holdout_eval_house_revision_20261107_s1_manifest.json')
    subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),'--only-extra','--extra',str(source),str(target),
                    '--sessions',*evaluation,'--report-name','house_revision_comparison','--report-date','20261107'],cwd=root,check=True)
    print('COMPLETE: additive 24-session model evaluated on six isolated holdout sessions',flush=True)
except Exception as exc:
    (target/'failure.json').write_text(json.dumps(dict(error=str(exc),time=dt.datetime.now().isoformat())))
    raise
