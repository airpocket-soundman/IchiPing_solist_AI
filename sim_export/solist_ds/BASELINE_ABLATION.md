# Baseline augmentation ablation

評価は常に未知run自身の起動時baseline。学習view数・周波数シフト・モデル・seedを揃え、baseline augmentationだけを変更した。

| 学習baseline | モデル | frame | macro F1 | state vote | self比frame差 |
|---|---|---:|---:|---:|---:|
| self_only | PC CNN XL/1024 | 92.1% | 91.0% | 92.9% | +0.0 pt |
| self_only | Solist ELM m32/shape167 | 57.7% | 52.1% | 58.3% | +0.0 pt |
| within_day | PC CNN XL/1024 | 91.2% | 89.7% | 91.8% | -0.9 pt |
| within_day | Solist ELM m32/shape167 | 61.0% | 55.7% | 62.7% | +3.3 pt |
| cross_day | PC CNN XL/1024 | 90.8% | 88.9% | 91.3% | -1.3 pt |
| cross_day | Solist ELM m32/shape167 | 62.8% | 57.5% | 64.6% | +5.0 pt |

## Fold detail

| 未知日 | 学習日間shift | aug範囲 | baseline | PC CNN frame | ELM frame |
|---|---:|---:|---|---:|---:|
| 2026-05-30 | 0.15% | ±0.30% | self_only | 94.4% | 51.2% |
| 2026-05-30 | 0.15% | ±0.30% | within_day | 89.3% | 58.2% |
| 2026-05-30 | 0.15% | ±0.30% | cross_day | 82.7% | 61.0% |
| 2026-05-31 | 0.30% | ±0.60% | self_only | 93.5% | 69.0% |
| 2026-05-31 | 0.30% | ±0.60% | within_day | 94.8% | 66.9% |
| 2026-05-31 | 0.30% | ±0.60% | cross_day | 99.3% | 68.5% |
| 2026-06-01 | -0.05% | ±0.10% | self_only | 88.3% | 53.0% |
| 2026-06-01 | -0.05% | ±0.10% | within_day | 89.4% | 57.9% |
| 2026-06-01 | -0.05% | ±0.10% | cross_day | 90.3% | 58.8% |

## 結論

- PC CNNはself-onlyが92.1%で最良。cross-dayは90.8%で-1.3 ptとなり、平均では悪化した。
- m=32 ELMはself-only 57.7%に対し、within-day 61.0% (+3.3 pt)、cross-day 62.8% (+5.0 pt)。baseline jitterが低容量モデルの正則化として働いた。
- cross-dayのPC効果は未知日ごとに大きく変動したため、物理的に不一致なcross-day差分をPC教師モデルの標準条件にはしない。ELMでは実運用に近いwithin-day jitterを優先候補とする。
