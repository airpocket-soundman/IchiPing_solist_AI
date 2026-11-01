"""Controlled ablations, with seed selection on rotating training-only holdouts."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent.parent
folder=root/'sim_export/solist_ds/robust_training_20261101'
folder.mkdir(exist_ok=True)
previous=root/'sim_export/solist_ds/candidates_20261101'
source=previous/'lowband_all20_fixedbase_xbase16'
base=json.loads((source/'complete.json').read_text())
evaluation=json.loads((previous/'cross_baseline_plan.json').read_text())['evaluation_sessions']
folds=[base['validation'],
       ['stamp_20260926_s2_wav','stamp_20260926_s8_wav','stamp_20261031_current_train_20261031_s1_wav'],
       ['stamp_20260926_s11_wav','stamp_20261030_speed500_s1_wav','stamp_20261031_current_train_post_sweep_20261031_s1_wav']]
assert not set(evaluation).intersection(base['sessions'])
for fold in folds:
    assert set(fold)<=set(base['sessions'])
plan=dict(sessions=base['sessions'],baseline_reference=base['baseline_reference'],cross_baseline=4,
          folds=folds,evaluation_sessions=evaluation,
          variants=['balanced','rotating_validation','c_pair_margin','combined'],
          balance='Equal total weight per class; within class equal weight per session containing it, in CNN and ELM.',
          pair='All 32 classes: softplus(logit_C_flipped - logit_correct + 0.5), coefficient 0.2.',
          selection='Three disjoint-from-fit validation unions; choose shared seed by minimum session accuracy across folds, then mean; refit fixed original split. No holdout-evaluation audio used for selection.')
(folder/'plan.json').write_text(json.dumps(plan,indent=2))


def train(name,flags,validation,seed=None):
    out=folder/name
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists(): return out
    config=dict(base,name=name,validation=validation,cross_baseline=4,experimental_flags=flags,forced_seed=seed)
    config.pop('elapsed_seconds',None);config.pop('completed',None)
    (out/'plan.json').write_text(json.dumps(config,indent=2))
    command=[sys.executable,'-u',str(root/'sim/train_robust_variant.py'),*flags,
             '--arch','small','--train','stamp','--stamp',*base['sessions'],
             '--val-sessions',*validation,'--seeds',str(3 if seed is None else 1),
             '--shift','0.01','--xbase','4','--input-bin-lo','12','--feature-noise-std','0',
             '--baseline-reference',*base['baseline_reference'],'--tag',name,'--output-dir',str(out)]
    if seed is not None:command+=['--forced-seed',str(seed)]
    print('TRAINING',name,flush=True)
    start=time.monotonic()
    with (out/'training.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    config.update(elapsed_seconds=time.monotonic()-start,completed=dt.datetime.now().isoformat())
    (out/'complete.json').write_text(json.dumps(config,indent=2))
    return out


try:
    deadline=time.monotonic()+7200
    pending=previous/'lowband_all20_fixedbase_xbase8'
    while not (pending/'all20_comparison.json').exists():
        if (pending/'failure.json').exists():raise RuntimeError('Preceding xbase8 training failed')
        if time.monotonic()>deadline:raise TimeoutError('Waiting for preceding xbase4/8 experiments')
        time.sleep(10)
    variants=[('balanced',['--balance-session-class'],False),
              ('rotating_validation',['--robust-validation'],True),
              ('c_pair_margin',['--c-pair-weight','0.2'],False),
              ('combined',['--balance-session-class','--robust-validation','--c-pair-weight','0.2'],True)]
    outputs=[]
    for name,flags,rotate in variants:
        selected=None
        if rotate:
            reports=[]
            for index,fold in enumerate(folds):
                fold_dir=train(f'{name}_fold{index}',flags,fold)
                reports.append(json.loads((fold_dir/'validation_selection.json').read_text()))
            rankings=[]
            for seed in range(3):
                acc=[acc for report in reports for row in report if row['seed']==seed for acc in row['sessions'].values()]
                rankings.append(dict(seed=seed,worst=min(acc),mean=sum(acc)/len(acc)))
            selected=max(rankings,key=lambda r:(r['worst'],r['mean']))['seed']
            (folder/f'{name}_seed_selection.json').write_text(json.dumps(dict(folds=folds,rankings=rankings,selected_seed=selected),indent=2))
        out=train(name,flags,folds[0],selected)
        outputs.append(out)
        comparisons=[previous/'lowband_all20_fixedbase_xbase4',previous/'lowband_multi_plus2_hidden_c_fixedbase_xbase16']+outputs
        subprocess.run([sys.executable,'-u',str(root/'sim/evaluate_candidates.py'),'--only-extra','--extra',*map(str,comparisons),
                        '--sessions',*evaluation,'--report-name','robust_training_comparison'],cwd=root,check=True)
        print('EVALUATED',name,flush=True)
    (folder/'complete.json').write_text(json.dumps(dict(completed=dt.datetime.now().isoformat(),models=list(map(str,outputs)))))
except Exception as exc:
    (folder/'failure.json').write_text(json.dumps(dict(error=str(exc),time=dt.datetime.now().isoformat())))
    raise
