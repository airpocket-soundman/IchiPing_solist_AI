"""Add the latest training capture to the best saved xbase-16 configuration."""
import datetime as dt
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--xbase', type=int, choices=(4, 8, 16), default=16)
args = parser.parse_args()

root = Path(__file__).resolve().parent.parent
folder = root/'sim_export/solist_ds/candidates_20261101'
source = folder/'lowband_multi_plus2_hidden_c_fixedbase_xbase16'
config = json.loads((source/'complete.json').read_text())
latest = 'stamp_20261031_current_train_post_sweep_20261031_s1_wav'
sessions = config['sessions'] + [latest]
assert len(sessions) == len(set(sessions)) == 20
evaluation = json.loads((folder/'cross_baseline_plan.json').read_text())['evaluation_sessions']
assert not set(sessions).intersection(evaluation)
available = {p.name for p in (root/'captures').glob('stamp_*')
             if (p/'session.json').exists() and 'holdout_eval' not in p.name}
assert set(sessions) == available, (available-set(sessions), set(sessions)-available)
name = f'lowband_all20_fixedbase_xbase{args.xbase}'
target = folder/name
target.mkdir(exist_ok=True)
config.update(name=name, sessions=sessions, source_model=str(source), cross_baseline=args.xbase)
for field in ('elapsed_seconds', 'completed'):
    config.pop(field, None)
(target/'plan.json').write_text(json.dumps(config, indent=2))
try:
    if not (target/'complete.json').exists():
        start = time.monotonic()
        print('TRAINING', name, flush=True)
        command = [sys.executable, '-u', str(root/'sim/emit_frontend_model.py'),
                   '--arch', 'small', '--train', 'stamp', '--stamp', *sessions,
                   '--val-sessions', *config['validation'], '--seeds', '3',
                   '--shift', '0.01', '--xbase', str(args.xbase), '--input-bin-lo', '12',
                   '--feature-noise-std', '0', '--baseline-reference', *config['baseline_reference'],
                   '--tag', name, '--output-dir', str(target)]
        with (target/'training.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        config.update(elapsed_seconds=time.monotonic()-start, completed=dt.datetime.now().isoformat())
        (target/'complete.json').write_text(json.dumps(config, indent=2))
    comparisons = [root/'sim_export/solist_ds/candidates_20261031'/n for n in
                   ('lowband_multi_plus2_fixedbase', 'lowband_multi_plus2_hidden_c_fixedbase')]
    comparisons += [folder/n for n in (
        'lowband_multi_plus2_fixedbase_xbase8', 'lowband_multi_plus2_hidden_c_fixedbase_xbase8',
        'lowband_multi_plus2_fixedbase_xbase16', 'lowband_multi_plus2_hidden_c_fixedbase_xbase16')]
    comparisons += [folder/f'lowband_all20_fixedbase_xbase{k}' for k in (4, 8, 16)
                    if (folder/f'lowband_all20_fixedbase_xbase{k}'/'complete.json').exists()]
    subprocess.run([sys.executable, '-u', str(root/'sim/evaluate_candidates.py'),
                    '--only-extra', '--extra', *map(str, comparisons),
                    '--sessions', *evaluation, '--report-name', 'all20_comparison'],
                   cwd=root, check=True)
    print('COMPLETE: all 20 training sessions, four isolated evaluation sessions', flush=True)
except Exception as exc:
    (target/'failure.json').write_text(json.dumps(dict(error=str(exc), time=dt.datetime.now().isoformat())))
    raise
