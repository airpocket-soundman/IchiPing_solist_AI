# Solist-AI Sim (教師あり版) 用データと設定

別セッション検証（学習 v6-v11 → テスト v12）。
特徴 = **FFT(audio − baseline)**（時間領域でbaseline減算→FFTマグニチュード, 400-3000Hz, D=167, 標準化済）。
学習データは **cross-baseline（v6/v9/v11 の3 baselineで diff）＋ augmentation（SpectralJitter+LevelJitter+FreqMask）** 込み。

## 使うファイル（推奨=5k版, ファイル内データ数≤1,000,000 を満たす）
| ファイル | 行 | 列 | セル数 | Hidden | 用途 |
|---|---|---|---|---|---|
| train_xbl_aug_v6-11_14cls_5k.csv | 5,500 | 181 | 995,500 | 256/128 | 学習(14cls) |
| test_v12_14cls.csv | 1,600 | 181 | 289,600 | — | テスト(14cls) |
| train_xbl_aug_v6-11_32cls_5k.csv | 5,000 | 199 | 995,000 | **128** | 学習(32cls) |
| test_v12_32cls.csv | 1,600 | 199 | 318,400 | — | テスト(32cls) |

**制約: ファイル内データ数(行×列)≤1,000,000** を全ファイルで満たす。
（24k版/フル版(57,600行)も残置するが 1M超なのでSim用には5k版を使う。
 32clsは5,000行ではHidden=256だと過学習→**Hidden=128推奨**。14clsはどちらでも~99.8%。）

各行=1チャンク(1サンプル)。1行目ヘッダはSimが無視。

## Sim 設定
**タブ1/2 (Training/Test Data)**
- Input data: First column=**1**, rows at one chunk=**1**, columns at one chunk=**167**
- Expected data: First column=**168**, rows=**1**, columns=**14**（32cls版は**32**）
- Number of chunks: Automatic
- Check → 入力ノード167 / 出力ノード14(or32)

**タブ3 (AI settings and Sim)**
- Hidden=**256**（AI RAMが収まらなければ128）
- Activation=**Hard sigmoid**, Loss=**MSE**, Forgetting=**1.0**, Seed=**1**
- scaleAlpha≈**0.2**, scaleGamma=**0**(ESN無), leakRate=**0**, l2Param≈**0.1**
- Bfloat16: 最初OFF(double)→後でON

## 注意
- 特徴は標準化済み → Normalizeオプションは OFF。
- ノード和: 167+256+14=437 / +32=455（≤570 目安内）。
- 判定: one-hot分類 → argmax(Actual)==argmax(Expected)。直接の正解率表示が無ければ Save の Actual を書き出して計算。

## 期待結果（Python faithful sim 参照値, 別セッションv12, cross-baseline+aug学習）
- **14cls ≈ 100%**
- **32cls ≈ 92%**（cross-baseline+aug無しだと69%まで低下。aug込みが重要）

生成: sim/export_sim_aug.py
