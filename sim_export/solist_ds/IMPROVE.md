# 精度向上策の比較まとめ (2026-09-25)

評価: UNO Q eval 4 セット (gray 08:29 / evening 19:36 / survey 20:44 / crowd 21:29、各セット自前 baseline 校正) の
14cls frame 精度。学習 variant は `unoq_ir2` (UNO Q 8 セッション + IR シフト)。全結果は
[IMPROVE_ABCDE.md](IMPROVE_ABCDE.md) / [IMPROVE_FGIJK.md](IMPROVE_FGIJK.md) / [IMPROVE_L.md](IMPROVE_L.md)、
実験コードは `sim/improve_experiments.py`。

## 結論

**律速は ELM の α がランダム射影であること。** 167 次元の特徴を 32 本の乱数射影に落とすと情報が失われ
(hard sigmoid はほぼ線形域なので実質 rank-32 の線形モデル)、hidden を増やしても 64 で 87%、128 で 91% まで。
一方、**PC で学習した線形分類器 (ridge, 167→14) は単体で 93.6%** に達し、その 14 スコアを z-score して
ELM に入れる「線形前段 + ELM」が最良。前段は Stamp-S3A または Solist CPU で計算 (167×14 MAC)、ODL は β だけを持つ。

| 手法 (14cls, m=hidden) | gray | evening | survey | crowd | 平均 |
|---|---|---|---|---|---|
| A. ELM m=32 Simα (現状) | 78–85 | 68–72 | 75–80 | 68–74 | **74** |
| B. ELM m=64 乱数α | 91.0 | 78.1 | 85.7 | 92.0 | **86.7** |
| B. ELM m=128 乱数α (実機不可) | 96.9 | 83.3 | 91.7 | 93.3 | **91.3** |
| C. bin平均 k=3 (D=55) + ELM m=64 | 96.4 | 85.9 | 87.5 | 90.2 | **90.0** |
| E. 線形 ridge D=167 λ=10 (hidden 無し) | 97.9 | 87.2 | 93.8 | 95.6 | **93.6** |
| F2. 線形ridge14 前段 + ELM m=32 Simα s=4 | 100 | 93.8 | 89.1 | 93.8 | **94.1** |
| F2. 線形ridge14 前段 + ELM m=64 s=4 | 100 | 94.7 | 93.8 | 98.8 | **96.8** |
| 上記 + 学習データ `unoq+frdm_ir2` (m=32 Simα) | 100 | 96.9 | 96.9 | 93.8 | **96.9** |
| 上記 + 学習データ `unoq+frdm_ir2` (m=64) | 100 | 100 | 100 | 97.2 | **99.3** (投票 100) |
| I. 現地校正 1 frame/状態 (×20) + 線形 ridge | 98.4 | 96.9 | 93.8 | 96.9 | **96.5** |
| 現地校正 2 frame/状態 + 前段 + ELM m=64 (`unoq+frdm_ir2`) | 100 | 100 | 100 | 99.6 | **99.9** |

効かなかったもの: 入力 scale / ridge の調整 (±2 pt)、frame 正規化 (ELM では悪化)、PCA 前段 (16–48 で 63–79%)、
2 インスタンス集約 (+2 pt)、ELM の β だけの現地校正 (+1 pt)。LDA13 前段は 84–86% で ridge 前段に劣る。

周波数シフト (気温差対策) は線形モデルで効く: evening が `unoq_none` 84.4 → `unoq_ir2` 86.6 → `unoq+frdm_ir2` 93.8%。

## 32cls

| 手法 | gray | evening | survey | crowd | 平均 |
|---|---|---|---|---|---|
| ELM m=32 Simα | 40.6 | 25.0 | 21.6 | 30.0 | **29.3** |
| 線形 ridge D=167 | 68.8 | 41.9 | 50.0 | 65.9 | **56.6** |
| 線形ridge32 前段 + ELM m=64 s=2 (`unoq+frdm_ir2`) | 72.9 | 52.5 | 61.6 | 60.0 | **61.7** |
| 上記 + 現地校正 2 frame/状態 | 68.8 | 74.2 | 73.8 | 68.8 | **71.4** |

32 状態は同一 14 クラス内の区別が音響的に弱く (UNO Q CNN でも 82%)、現地校正が必須。

## 実機検証 (2026-09-25)

実機の α は seed/scaleAlpha だけでなく **inputSize に依存して変わる** (167 入力では Sim 採取値と一致するが、
14 入力では別の行列)。そこで `firmware/IchiPingInference/tools/probe_alpha.py` (β=単位行列ファームに ±c·e_i を
送って α_ij を読む) で実機 α を直接取得した (167×32 は Sim 採取値と最大差 0.003 で一致、方法の妥当性を確認)。

- 構成: 線形 ridge 前段 (PC 学習, 167→14, z-score, ×4) → ELM 14→64→14 (実機プローブ α, β は PC 学習, hidden 63 番は
  実機で出力されないため β から除外)。
- 結果 ([docs/board_inference_test_frontend.md](../../docs/board_inference_test_frontend.md)): evening 320 frame で
  **Board 96.2% (投票 96.9%)**、PC bf16 参照との argmax 一致 100%、スコア誤差 max 0.039、推論 245 µs。
  改善前 (167→32→14, Sim α) の 64.4% から +32 pt。
