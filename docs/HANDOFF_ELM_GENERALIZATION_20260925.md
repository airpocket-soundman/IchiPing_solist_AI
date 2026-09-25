# ELM 32cls 日跨ぎ汎化検証 引継ぎ（2026-09-25）

## 1. 目的と現在の結論

Original IchiPing と Arduino IchiPing / UNO Q の複数セッションデータを使い、温度に伴う周波数シフトをaugmentationへ入れたとき、PCの高容量モデルとSolist-AI互換ELMで未知日32クラス分類がどこまで可能かを検証している。

現時点の厳密なOriginal 3日 leave-one-day-out結果は次の通り。

| モデル | frame accuracy | macro F1 | state vote |
|---|---:|---:|---:|
| PC CNN XL / 1024-bin `noise_diff_norm` | 92.0% | 90.9% | 92.2% |
| PC CNN / 現行167特徴 | 74.2% | 70.8% | 75.9% |
| Solist ELM m=32 / 現行D167 | 57.7% | 54.4% | 58.9% |
| Solist ELM m=32 / 1024→167単純補間 | 57.8% | 52.2% | 58.3% |

PC理想モデルから現行ELMへのframe低下は34.3ポイント。内訳は、1024-bin形状特徴から現行167特徴への変更が約17.8ポイント、高容量CNNから固定ランダムα・m=32 ELMへの変更がさらに約16.4ポイント。単純な1024→167補間では改善しないため、次段階では教師あり射影またはCNN teacherからの蒸留が必要。

## 2. 絶対に維持する評価条件

1. 各runの測定音は、そのrunの起動時に取得したbaselineだけで差分化する。
2. 評価日は学習、シフト推定、標準化、ハイパーパラメータ選択に一切使用しない。
3. 外側分割は日単位のleave-one-day-outとする。
4. 評価データにはaugmentationを適用しない。
5. 日間シフトは外側foldの学習2日だけから推定する。
6. Original世代は励振PRBSを再現できないため、同日baseline差分後のスペクトルを周波数方向へワープする。
7. 未知条件の主指標はmacro F1。frame accuracyとrun/class単位のscore-sum voteも併記する。

Originalデータの日付対応:

- 2026-05-30: `full_32_train_v6`, `v7`, `v8`
- 2026-05-31: `full_32_train_v9`, `v10`
- 2026-06-01: `full_32_train_v11`, `v12`

UNO Qの学習・評価データは複数時刻／条件だが、ディレクトリ上はすべて2026-09-12であり、厳密な日跨ぎデータではない。

## 3. 周波数ワープの意図

ユーザーの意図は、観測された日間周波数シフトを再現するだけではない。学習日間で推定したシフト量を基準に、その約2倍までデータをワープして混ぜ、絶対周波数ではなくスペクトル形状で分類させること。

現在のOriginal 3-foldで学習日間から推定されたglobal shiftは次の通り。

- holdout 2026-05-30: +0.15%、augmentation範囲 ±0.30%
- holdout 2026-05-31: +0.30%、augmentation範囲 ±0.60%
- holdout 2026-06-01: −0.05%、augmentation範囲 ±0.10%

UNO Qでは実測ドリフトがevening −2.15%、survey −1.05%、crowd −0.80%。同条件の直接比較で、周波数ワープなし70.3%から±3%ワープ77.6%へ平均+7.3ポイント、eveningとsurveyでは約+12ポイント改善した。最終8セッションモデルは平均32cls 82.5%。根拠は `D:/GitHub/IchiPing-UNO-Q/docs/uno_q/results.md` と `pc/runs/model_comparison_20260912.md`。

## 4. Baseline augmentationの統制実験

評価は常に未知run自身のbaseline。学習view数、周波数ワープ、モデル、seedを統一し、学習baselineだけを変更した。

| 学習baseline | PC CNN XL/1024 | Solist ELM m32/shape167 |
|---|---:|---:|
| selfのみ | 92.1% | 57.7% |
| self＋同日別run | 91.2%（−0.9pt） | 61.0%（+3.3pt） |
| self＋別日 | 90.8%（−1.3pt） | 62.8%（+5.0pt） |

解釈:

- PC教師モデルはself-onlyが最良。cross-day baselineは平均で悪化し、fold間変動も大きい。
- m=32 ELMではbaseline jitterが強い正則化として働く。
- 運用整合性を優先し、PC teacherはself-only。ELMでjitterを使うなら、まず同日・別起動baselineを候補にする。
- 以前cross-baselineのPC結果が高く見えた主因は、baseline方式ではなく評価分割の違い。

詳細: `sim_export/solist_ds/BASELINE_ABLATION.md`

## 5. 追加されたファイル

- `sim/eval_ideal_vs_solist.py`
  - same-run baseline、日間shift推定、±2倍warp
  - PC CNN XL/1024、PC CNN/167、Solist ELM m32を比較
  - GPU決定論モードを使用
- `sim/eval_baseline_ablation.py`
  - self / within-day / cross-day baseline augmentationの統制比較
  - 学習view数を2に揃えるため、self条件はself viewを2回提示
- `sim/eval_multiday_elm.py`
  - 初期探索。cross-day baselineを含むため運用汎化の採用判定には使用不可
  - レポート冒頭にも無効条件であることを明記済み
- `sim_export/solist_ds/IDEAL_VS_SOLIST.{md,json}`
- `sim_export/solist_ds/BASELINE_ABLATION.{md,json}`
- `sim_export/solist_ds/MULTIDAY_ELM.{md,json}`（旧探索・採用不可）

中間cacheは `sim/_cache/ideal_noise_diff_norm_v*.npz` と `ideal_raw_logpsd_v*.npz`。gitignore対象でありコミットしない。元のIchiPing / UNO-Qリポジトリは変更していない。

## 6. 次に実施する作業（最優先）

周波数ワープ単独ablationを、self-baseline固定で実施する。baseline ablationでは全条件に同じワープを入れていたため、ワープ自体の寄与はまだ分離できていない。

比較条件:

1. warpなし
2. 学習日間global shiftの±1倍まで
3. 学習日間global shiftの±2倍まで
4. UNO Qと同じ固定±3%
5. クラス別shiftのrobust上限（P90程度）の±2倍まで

公平性要件:

- self-baselineのみ
- 各条件の学習view数とoptimizer step数を同じにする
- no-warpは同じサンプルを複製して提示回数を合わせる
- 同一CNN、同一seed、同一validation split
- PC CNN XL/1024とSolist ELM m32/shape167の両方を評価
- 評価日は常に未加工・self-baseline

実装案:

- `eval_ideal_vs_solist.py` の `augment()` を、明示的なepsilon列を受け取れる関数へ一般化する。
- 現状は `(0, -delta, +delta, -2delta, +2delta)` 固定。
- 条件ごとに5 viewへ揃える例:
  - none: `(0, 0, 0, 0, 0)`
  - ±1x: `(0, -d, -d, +d, +d)`
  - ±2x: `(0, -d, +d, -2d, +2d)`
  - ±3%: `(0, -0.015, +0.015, -0.03, +0.03)`
- `estimate_shift()` は現在global shiftとclass IQRだけを返す。クラス別epsilon列または`p90_abs_class_shift`も返すよう拡張する。
- 新規 `sim/eval_warp_ablation.py` と `sim_export/solist_ds/WARP_ABLATION.{md,json}` を推奨。

## 7. 実行環境と再現コマンド

通常のPythonにはPyTorchがない。IchiPing側のGPU対応venvを使う。

```powershell
D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe -m py_compile `
  sim/eval_ideal_vs_solist.py sim/eval_baseline_ablation.py sim/eval_multiday_elm.py

D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_ideal_vs_solist.py --epochs 100
D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_baseline_ablation.py
```

確認済み環境:

- PyTorch 2.11.0+cu128
- CUDA利用可
- PC CNNは`torch.use_deterministic_algorithms(True)`を設定

## 8. 注意事項

- `IDEAL_VS_SOLIST`のPC理想モデルとlegacy ELMは入力frontendも異なる。純粋なモデル差だけではないため、PC CNN/167を中間比較としている。
- shape167は1024-binを167点へ単純補間しただけで、教師あり圧縮ではない。
- 日数は3日だけで信頼区間が広い。最終Stamp-S3Aデータでは最低3日、可能なら5日以上、各日複数時刻・起動baselineを収集する。
- READMEの99.1%はv12内で10 frame/class校正後に同じv12の残りを評価した値で、日跨ぎ汎化値ではない。
- 旧`MULTIDAY_ELM`結果はcross-day baselineを含むため、運用採用値として引用しない。
