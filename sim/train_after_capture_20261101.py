"""Train as soon as two new training captures are verified; evaluate after holdout capture."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent.parent
folder=root/'sim_export/solist_ds/candidates_20261101'
source=folder/'lowband_all20_fixedbase_xbase8'
target=folder/'lowband_all22_fixedbase_xbase8'
target.mkdir(exist_ok=True)
latest=[f'stamp_20261101_additional_train_20261101_s{i}_wav' for i in (1,2)]
eval_new='stamp_20261101_holdout_eval_20261101_s1_wav'
evaluation=json.loads((folder/'cross_baseline_plan.json').read_text())['evaluation_sessions']+[eval_new]
def wait_for(path):
    deadline=time.monotonic()+14400
    while not path.exists():
        capture=json.loads((root/'.local/collect_three_20261101_status.json').read_text())
        if capture['state']=='failed':raise RuntimeError(capture.get('error','Capture failed'))
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(10)
try:
    for i,session in enumerate(latest,1):
        wait_for(root/'docs/results'/f'additional_train_20261101_s{i}_manifest.json')
        assert (root/'captures'/session/'session.json').exists()
    config=json.loads((source/'complete.json').read_text())
    sessions=config['sessions']+latest
    assert len(sessions)==len(set(sessions))==22
    assert not set(evaluation).intersection(sessions)
    config.update(name=target.name,sessions=sessions,source_model=str(source),cross_baseline=8)
    for field in ('elapsed_seconds','completed'):config.pop(field,None)
    (target/'plan.json').write_text(json.dumps(config,indent=2))
    if not (target/'complete.json').exists():
        print('TRAINING',target.name,flush=True)
        start=time.monotonic()
        command=[sys.executable,'-u',str(root/'sim/emit_frontend_model.py'),'--arch','small',
                 '--train','stamp','--stamp',*sessions,'--val-sessions',*config['validation'],
                 '--seeds','3','--shift','0.01','--xbase','8','--input-bin-lo','12',
                 '--feature-noise-std','0','--baseline-reference',*config['baseline_reference'],
                 '--tag',target.name,'--output-dir',str(target)]
        with (target/'training.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        config.update(elapsed_seconds=time.monotonic()-start,completed=dt.datetime.now().isoformat())
        (target/'complete.json').write_text(json.dumps(config,indent=2))
    wait_for(root/'docs/results/holdout_eval_20261101_s1_manifest.json')
    comparisons=[folder/'lowband_multi_plus2_hidden_c_fixedbase_xbase16',source,target]
    subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),'--only-extra',
                    '--extra',*map(str,comparisons),'--sessions',*evaluation,
                    '--report-name','all22_five_holdout_comparison'],cwd=root,check=True)
    print('COMPLETE: 22-session model compared on five isolated holdout sessions',flush=True)
except Exception as exc:
    (target/'failure.json').write_text(json.dumps(dict(error=str(exc),time=dt.datetime.now().isoformat())))
    raise
