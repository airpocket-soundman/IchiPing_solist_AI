"""Add only the new full session to the best-condition frozen training set."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
base = root / 'sim_export/solist_ds/candidates_20261031/stamp_shift0.01_base4_noise0/complete.json'
config = json.loads(base.read_text())
target = 'stamp_20261031_current_train_20261031_s1_wav'
session_path = root / 'captures' / target
assert (session_path / 'session.json').exists(), 'Full session incomplete'
assert len(list(session_path.rglob('*.wav'))) == 330, 'Unexpected frame count'
sessions = config['sessions'] + [target]
name = 'stamp_shift0.01_base4_noise0_added_full'
out = base.parent.parent / name
out.mkdir(parents=True, exist_ok=True)
if not (out / 'complete.json').exists():
    command = [sys.executable, '-u', str(root / 'sim/emit_frontend_model.py'),
        '--arch', 'small', '--train', 'stamp', '--stamp', *sessions,
        '--val', config['validation'], '--seeds', '3', '--shift', '0.01',
        '--xbase', '4', '--feature-noise-std', '0', '--tag', name, '--output-dir', str(out)]
    print('TRAINING', name, flush=True)
    with (out / 'training.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / 'complete.json').write_text(json.dumps(dict(config, name=name, sessions=sessions), indent=2))
    print('SAVED', name, flush=True)
subprocess.run([sys.executable, '-u', str(root / 'sim/evaluate_candidates.py')], cwd=root, check=True)
