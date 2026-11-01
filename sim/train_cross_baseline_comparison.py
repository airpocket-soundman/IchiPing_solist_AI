"""Compare xbase 8/16 against the saved 18/19-session xbase-4 models."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'sim_export/solist_ds/candidates_20261101'
OUT.mkdir(parents=True, exist_ok=True)
OLD = ROOT / 'sim_export/solist_ds/candidates_20261031'
reference = json.loads((OLD/'stamp_shift0.01_base4_noise0/complete.json').read_text())['sessions']
assert len(reference) == 16 and len(set(reference)) == 16
evaluation = [
    'stamp_20261031_holdout_eval_20261031_s1_wav',
    'stamp_20261031_holdout_eval_20261031_s2_wav',
    'stamp_20261031_holdout_eval_fresh_20261031_wav',
    'stamp_20261031_holdout_eval_post_sweep_20261031_s1_wav',
]
originals = [OLD/'lowband_multi_plus2_fixedbase', OLD/'lowband_multi_plus2_hidden_c_fixedbase']
baseline_hashes = {
    name: hashlib.sha256((ROOT/'captures'/name/'s00000/meta.json').read_bytes()).hexdigest()
    for name in reference
}
plan = dict(reference_sessions=reference, baseline_metadata_sha256=baseline_hashes,
            evaluation_sessions=evaluation, cross_counts=[8, 16], seeds=3,
            note='Same ordered reference pool and unchanged baseline extraction as xbase-4. '
                 'Sampling excludes own session and validation sessions during CNN fit; '
                 'requested cross count is capped by available reference sessions.')
(OUT/'cross_baseline_plan.json').write_text(json.dumps(plan, indent=2))
candidates = []
try:
    for cross in (8, 16):
        for original in originals:
            config = json.loads((original/'complete.json').read_text())
            assert not set(evaluation).intersection(config['sessions'])
            name = f'{original.name}_xbase{cross}'
            target = OUT/name
            target.mkdir(exist_ok=True)
            candidates.append(target)
            if (target/'complete.json').exists():
                continue
            start = time.monotonic()
            print(f'TRAINING {name}', flush=True)
            command = [sys.executable, '-u', str(ROOT/'sim/emit_frontend_model.py'),
                       '--arch', 'small', '--train', 'stamp', '--stamp', *config['sessions'],
                       '--val-sessions', *config['validation'], '--seeds', '3',
                       '--shift', '0.01', '--xbase', str(cross), '--input-bin-lo', '12',
                       '--feature-noise-std', '0', '--baseline-reference', *reference,
                       '--tag', name, '--output-dir', str(target)]
            with (target/'training.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            config.update(name=name, cross_baseline=cross, baseline_reference=reference,
                          baseline_metadata_sha256=baseline_hashes,
                          elapsed_seconds=time.monotonic()-start,
                          completed=dt.datetime.now().isoformat(), source_model=str(original))
            (target/'complete.json').write_text(json.dumps(config, indent=2))
            subprocess.run([sys.executable, '-u', str(ROOT/'sim/evaluate_candidates.py'),
                            '--only-extra', '--extra', str(target), '--sessions', *evaluation,
                            '--report-name', 'cross_baseline_holdout'], cwd=ROOT, check=True)
            print(f'COMPLETE {name}: {config["elapsed_seconds"]:.0f} seconds', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT/'sim/evaluate_candidates.py'),
                    '--only-extra', '--extra', *map(str, originals+candidates),
                    '--sessions', *evaluation, '--report-name', 'cross_baseline_comparison'],
                   cwd=ROOT, check=True)
    (OUT/'cross_baseline_complete.json').write_text(json.dumps(dict(completed=dt.datetime.now().isoformat())))
except Exception as exc:
    (OUT/'cross_baseline_failure.json').write_text(json.dumps(dict(error=str(exc), time=dt.datetime.now().isoformat())))
    raise
