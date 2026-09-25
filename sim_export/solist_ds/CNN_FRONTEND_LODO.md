# 学習済み CNN 前段 + Solist ELM ヘッド (Original 日単位 LODO)

前段 = CNN XL の Conv 3 層 + ボトルネック 64 次元 (学習日のみで学習・凍結)。3 fold × seed 3 平均。
ヘッドのハイパラは固定 (ELM s=0.5 λ=1, 線形 λ=10; 評価日で選択していない)。校正は評価日の別 run 10 frame/class → 同日別 run。
CNN の early stopping は学習日内の frame 分割 (handoff と同じ) で、やや楽観側。

| クラス | ヘッド | factory frame | factory macroF1 | →14cls換算 | 校正のみ | factory+校正 混合 |
|---|---|---:|---:|---:|---:|---:|
| 32cls | PC CNN head (factory) | 89.3% | 88.0% | 100.0% | — | — |
| 32cls | ELM m32 乱数α on emb | 87.0% | 86.0% | 99.8% | 88.0% | 92.4% |
| 32cls | ELM m64 乱数α on emb | 88.3% | 87.4% | 99.7% | 91.5% | 94.2% |
| 32cls | 線形 ridge on emb | 87.2% | 86.2% | 99.0% | 91.7% | 92.2% |
| 14cls | PC CNN head (factory) | 100.0% | 100.0% | — | — | — |
| 14cls | ELM m32 乱数α on emb | 100.0% | 100.0% | — | 100.0% | 100.0% |
| 14cls | ELM m64 乱数α on emb | 100.0% | 100.0% | — | 100.0% | 100.0% |
| 14cls | 線形 ridge on emb | 100.0% | 100.0% | — | 100.0% | 100.0% |
