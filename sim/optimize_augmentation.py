"""Compare augmentation on disjoint session blocks, without emitting firmware."""
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_best_model as evaluation

# Keep all three new captures out together when evaluating the current room.
# This prevents cross-baseline augmentation from using their baselines in fit.
evaluation.BLOCKS['current_room'] = [
    'stamp_20261030_speed500_s1_wav',
    'stamp_20261030_speed500_s2_wav',
    'stamp_20261030_speed500_s3_retry_wav',
]
evaluation.STAMP_ALL = [r for block in evaluation.BLOCKS.values() for r in block]
conditions = [(data, shift, cross, noise)
              for data in ('unoq', 'stamp')
              for shift, cross, noise in ((0.005, 2, 0.0), (0.01, 2, 0.0), (0.02, 2, 0.0), (0.01, 4, 0.0), (0.005, 2, 0.05), (0.005, 2, 0.15), (0.01, 2, 0.05))]
for data, shift, cross, noise in conditions:
    name = f'augmentation_20261031_{data}_s{shift}_x{cross}_noise{noise}'
    print(f'START {name}', flush=True)
    evaluation.run_config(Namespace(
        data=data, shift=shift, xbase=cross, arch='small',
        stamp_rep=1, seeds=3, seed_offset=0, ensemble=1, name=name, feature_noise_std=noise,
    ))
    print(f'COMPLETE {name}', flush=True)
