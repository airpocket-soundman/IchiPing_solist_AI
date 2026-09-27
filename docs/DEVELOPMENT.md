# 開発記録:IchiPing on Solist-AI

IchiPing(1 個のマイクと 1 発の Ping で 3 部屋の窓 3 枚・扉 2 枚の開閉 32 通りを当てる能動音響センシング)を、
ROHM Solist-AI(ML63Q2557, 評価ボード DT-EBML63Q2557)へ移植した記録。設計判断・評価のルール・結果・残課題をまとめる。
作品としての説明は [ProtoPedia](https://protopedia.net/prototype/8556)、PV は [media/](../media/README.md) を参照。

## 1. 最終構成

| 処理 | 担当 | 実装 |
|---|---|---|
| PRBS 再生(2 s)と同時録音、48→16 kHz(63 tap FIR)、相関による開始位置合わせ | Stamp-S3A | `firmware/StampMeasure` |
| トグル 5 個・EXEC の読み取り(20 ms デバウンス) | Stamp-S3A | 同上(I²C STATUS 0x03) |
| PCM(16 kHz, 32,000 サンプル)を I²C で受信 | Solist-AI | `src/ichi_stamp_link.c`(READ 1 回 240 サンプル) |
| 特徴 N333:2048 点 FFT(int16 ブロック浮動小数点)→ Welch → dB → 起動時全閉との差分 → frame 正規化 → bin 50..383(334) → int8 | Solist-AI CPU | `src/ichi_feature.c` |
| int8 CNN 前段 small:Conv1d 8/16/32(k9/7/5, stride 2)→ FC 1216→32(重み 42,528, 約 42 KB) | Solist-AI CPU | `src/ichi_inference.c` |
| ELM ヘッド 167→32→32(埋め込み 32 次元をゼロ埋め, bf16, hard sigmoid, α は seed 1 から再生成) | Solist-AI AxlCORE | 同上 |
| 現地校正(ODL):OS-ELM で β を逐次更新(float32, P・β は FRAM) | Solist-AI CPU → AxlCORE | 同上 |
| TFT(ILI9341, ソフト SPI)、PCA9685 + SG90 ×5、状態管理 | Solist-AI | `src/ichi_tft.c`, `ichi_ui.c`, `ichi_servo.c` |

- RAM 使用 13.3 KB / 16 KB。FFT 作業領域と CNN 活性化で 4 KB を共用。
- 1 回の推論は要求から結果まで約 13.6 s。うち PCM を I²C で読みながら FFT する部分が約 8.4 s、特徴の仕上げ 0.19 s、推論 0.24 s。
- Stamp-S3A を使うのは、評価ボードに I²S が無く、拡張端子の信号ピン 8 本が TFT(5)・I²C(2)・PCA9685 OE(1)で埋まるため。
  端子の割り当ては [io_allocation.md](io_allocation.md)、中間基板は [hardware/stamp_s3a_interposer](../hardware/stamp_s3a_interposer/README.md)。

## 2. 経緯

| 時期 | 内容 | 主な結果 |
|---|---|---|
| 2026-06 | 公式 Solist-AI Sim(SLV1.00.04)で ELM(D167 特徴 = 時間波形 baseline 差分 → 1024 点 FFT → 167 点)を検証 | 14 クラス 99.4%(同一日の別セッション)。報告は [index.html](index.html) |
| 09-24 | 実機で ELM のみを推論(PC から特徴を送る試験ファーム) | UNO Q 夕方データで 14 クラス 64.4%。日をまたぐ汎化が足りない |
| 09-25 | 線形 ridge 前段 + ELM | 14 クラス 96.2% |
| 09-25 | 日単位の厳密評価で ELM だけの限界を確認し、CNN 前段 + ELM へ変更 | 32 クラス 66.3% → 89.7%(PC 評価) |
| 09-25〜26 | Stamp-S3A の I²S / I²C を立ち上げ、Solist 単体で録音 → 特徴 → 推論まで接続 | I²C 応答長を uint8 で持っていた不具合を修正(18.8% → 87.5%) |
| 09-26 | Stamp-S3A で自動採取を開始し、事前学習をこのハードのデータで作り直し | 32/32 に到達。時間帯が変わると 31/32 |
| 09-26 | ODL を AxlCORE 内蔵学習(bf16)で実装 → β が崩壊 | 3.1% → 正則化調整で 90.6%(改善なし) |
| 09-27 | 時間帯ブロック hold-out で事前学習モデルを選定、ODL を CPU float32 に変更 | 実機 32/32 ×2、校正後も 32/32 |

### 実機 32 状態サーベイの推移(サーボで 32 状態を 1 回ずつ作り Solist 単体で推論)

| 回 | 日時 | 内容 | 32 クラス | 14 クラス換算 |
|---:|---|---|---:|---:|
| 1 | 09-25 22:42 | UNO Q データだけで事前学習 | 75.0% | — |
| — | 09-26 10:32 | I²C READ の応答長が uint8 で切れる不具合 | 18.8% | 65.6% |
| 2 | 09-26 11:35 | 不具合修正後 | 87.5% | 96.9% |
| 3 | 09-26 13:41 | このハードの録音で β だけ再学習 | 93.8% | 100% |
| 4 | 09-26 13:50 | CNN 前段も再学習(UNO Q + Stamp s1–s5) | 100% | 100% |
| 5 | 09-26 18:00 | 気温が変わった条件 | 96.9% | 100% |
| 6 | 09-26 22:13 | 条件の違う収録を追加(s10, s11) | 96.9% | 100% |
| 7, 8 | 09-27 10:54 / 11:03 | セッション hold-out で選んだ完成版モデル | 100% / 100% | 100% |
| 9 | 09-27 11:16 | 完成版 + 現地校正(float32 ODL) | 100% | 100% |

最終 3 回の詳細は [results/](results/)。途中の回の生ログは整理時に削除した(値は上表のとおり)。

## 3. 設計判断

### 3.1 特徴:D167 から N333 へ

D167 は時間波形で baseline を引くため baseline 波形 64 KB の保持とサンプル単位の同期が必要で、Solist の SRAM 16 KB に合わない。
UNO Q 版で実績のある noise_diff_norm(log-PSD の差分 + frame 正規化)の 400–3000 Hz 334 bin(N333)に切り替えた。
baseline は dB スペクトル 2 KB で済み、サンプル同期も不要になる。全閉との差分を取ると、スピーカの癖・部屋の響き・定常雑音が打ち消され、
窓や扉が変わった分(数 dB)だけが残る。PC 参照実装は `sim/board_fixed_feature.py`(ファームと同じ整数演算)。

### 3.2 CNN 前段 + ELM の 2 段構成

- ELM(入力側 α は乱数で固定、出力側 β だけ学習)は AxlCORE で速く、β の更新だけで学び直せる。当初は ELM だけで足りると考えた。
- 学習に使っていない日で採点すると、ELM だけ(m=32, D167)は 32 クラス 66.3%([IMPROVE_LODO.md](../sim_export/solist_ds/IMPROVE_LODO.md))。
  PC 上の理想モデル CNN XL(1024 bin 入力, 約 30 万パラメータ)は 92.0%([IDEAL_VS_SOLIST.md](../sim_export/solist_ds/IDEAL_VS_SOLIST.md))。
- 小型 CNN 前段(small, 約 42 KB)で埋め込み 32 次元を作り ELM に渡すと 89.7%、校正込み 93.2%
  ([TINY_FRONTEND_LODO.md](../sim_export/solist_ds/TINY_FRONTEND_LODO.md)、大型前段の比較は [CNN_FRONTEND_LODO.md](../sim_export/solist_ds/CNN_FRONTEND_LODO.md))。
- 保存するパラメータは CNN 42,528 + β 1,024 = 約 43,600(α 5,344 は seed から再生成するので保存しない)。
  PC の理想モデルの約 1/7、オリジナル IchiPing の 4 層 CNN(約 10.4 万)の半分以下。
- CNN は int8 重み・int8 活性化・int32 積算。層ごとの再量子化 `v = acc·M + B` だけ float32(Cortex-M0+ に FPU は無い)。
- 大型前段 b1(約 174 KB)も動くが、検証精度は small と同等(0.796 vs 0.804)で、seed 間の差の方が大きかった。

### 3.3 データのばらつきと事前学習

同じ窓・扉の状態でも、気温(音速が変わり共鳴周波数が (1+ε) 倍にずれる、約 0.17%/°C)と背景音で反響音が変わり、
収録セッションごとにデータがばらつく。条件を散らして収録し([RUN_SHIFT.md](../sim_export/solist_ds/RUN_SHIFT.md) にセッション別の ε)、
周波数シフトの水増し(±0.5%)と別セッションの baseline で差分を取る水増し(cross-baseline 2)を加えて事前学習した。

- 学習データ:UNO Q 版のデータ(同じマイク・アンプ, 8 セッション)+ このハードの Stamp セッション 13 本(4,290 フレーム)。
  FRDM 世代のデータを足すと悪化した(94.9% vs 97.5%)。
- このハードのデータを足した効果:UNO Q だけの事前学習 80.4% → β だけ再学習 93.1% → 前段も再学習 99.1%
  (セッション単位 leave-one-out, [STAMP_MIX.md](../sim_export/solist_ds/STAMP_MIX.md))。
- 選定:セッションを条件ごとに 4 グループに分け、1 グループを丸ごと除いて学習・採点(時間帯ブロック hold-out)。
  完成版 `unoq_s0.005_x2`:99.2 / 99.9 / 95.8 / 100% → 平均 98.7%、14 クラス換算 100%([BEST_MODEL.md](../sim_export/solist_ds/BEST_MODEL.md))。

### 3.4 現地校正(ODL)

設置場所の変更や家具・部屋の変化など、事前学習で想定しきれない大きな変化は ODL で吸収する。

- EXEC を 2 s 長押し → サーボで扉と窓の開閉の組み合わせ全 32 クラスを巡回 → 各クラス 6 s の PRBS を 2 s 窓 × 5 本(1 s ずらし)に分けて
  OS-ELM で β を 1 サンプルずつ更新(計 160 回, 約 30 分)。EXEC で中断すると事前学習の β に戻る。校正後の β は電源断で消える。
- 初期値:P0 = w (G + λI)⁻¹(λ=10, w=1)。事前学習の Gram 行列 G は `generated/ichi_calib_prior.h`(`sim/emit_calibration_prior.py`)。
- AxlCORE 内蔵の学習(ODL_StartTrain)は β と P を bf16 で持つ。160 回の更新で丸め誤差が積み重なり β が発散した
  (全状態が同じ答えになり 3.1%。PC で bf16 を再現すると |β| が 7 → 10⁹)。λ・重みの調整で発散は止まったが校正の効果が出なかった(90.6%)。
- そこで β・P の更新だけ Cortex-M0+ の float32 で計算し、β を bf16 に変換して AxlCORE へ書き戻す形にした。
  P・β(float32 で 8 KB)は空き RAM に入らないため評価ボードの FRAM に置き 1 行ずつ読み書きする。所要時間の増加は 75 s。
- 効果:事前学習に含めていない条件の PC 検証で 89.4% → 100%(bf16 のままだと 95〜98%)。一部の状態だけ校正すると逆に悪化するので、全 32 クラスを校正する
  ([ODL_CALIBRATION.md](../sim_export/solist_ds/ODL_CALIBRATION.md))。

### 3.5 実機での注意点

- α は seed・scaleAlpha に加えて inputSize に依存する。167 入力は Sim で採取した α と一致したので、ELM 入力は 167 にゼロ埋めしている。
- hidden 64 では 63 番ユニットの出力が常に 0 になる(このため hidden は 32)。
- Stamp-S3A の I²S 出力(BCLK / WS / DOUT)は駆動力を最小(CAP_0)にする。既定の CAP_2 ではリンギングで INMP441 がクロックを誤計数し、無音でも約 −25 dBFS のノイズが乗った([io_allocation.md](io_allocation.md))。
- 画面の 5 桁は左から c・BC・b・AB・a。扉が閉じていて本来聞こえない桁は暗く表示する(判定:Complete / Conditional(14 クラス一致)/ Failure)。

## 4. 評価ルール

1. 評価するセッション(日・時間帯)は、学習・標準化・ハイパーパラメータ選択・early stop のいずれにも使わない。
2. ハイパーパラメータは学習側の内側 hold-out で選ぶ(frame 分割の validation は楽観的になる)。
3. 各セッションの音は、そのセッションの起動時に取った baseline だけで差分を取る(評価側に水増しは掛けない)。
4. baseline に使ったフレームは評価から除く。
5. 校正の評価は、校正に使ったのと別のセッションで行う。
6. グループ分けは日付ではなく、条件(気温)で考える(同日でも 11 °C 違うことがある)。

## 5. 残課題

- 推論に約 14 s かかる。大半は I²C で 2 s 分の PCM を 240 サンプルずつ受け取る時間(FFT と重ねて約 8.4 s)。
- ラベル無しの自動適応(運用中の逐次更新)は未実装。ODL は校正時のラベル付きデータだけを使う。
- FRDM 世代と UNO Q / Stamp 世代のデータ差の原因は未特定。
- 推論は 1 回の録音を 1 回推論する(複数回の多数決はしていない)。

## 6. 用語

- **周波数シフト**:気温による共鳴周波数の (1+ε) 倍の比例シフト(一定 Hz の加算ではない)。`sim/freq_shift.py`, `--shift`。
- **D167**:Sim 段階の特徴(時間波形 baseline 差分 → 1024 点 FFT → 400–3000 Hz の 167 点)。
- **N333**:完成版の特徴(noise_diff_norm の 400–3000 Hz, 334 bin)。
- **事前学習モデル**:出荷前に PC で学習して書き込むモデル。**現地校正 / ODL**:設置先で Solist-AI が β を学び直すこと。
- **LODO**:leave-one-day-out。

## 7. 資料の所在

| 内容 | 場所 |
|---|---|
| 完成版モデルと選定根拠 | `sim_export/solist_ds/board_model_frontend_32cls_best_s0.005.npz`, `BEST_MODEL.md`, `best_model/results.jsonl`, `BOARD_FRONTEND_PC.json` |
| UNO Q だけで学習した初期モデル(比較用) | `sim_export/solist_ds/board_model_frontend_32cls.npz`(`sim/eval_stamp_mix.py` の factory) |
| 探索の結果 | `sim_export/solist_ds/*.md`(IMPROVE_LODO, IDEAL_VS_SOLIST, TINY / CNN_FRONTEND_LODO, FULL_DATA, STAMP_MIX, RUN_SHIFT, ODL_CALIBRATION) |
| Sim 段階の成果物 | `sim_export/model1_BEST_*`, `docs/index.html`, `docs/models/` |
| 実機サーベイ(最終 3 回) | `docs/results/` |
| 端子割り当て・中間基板・発注データ | `docs/io_allocation.md`, `docs/solist_connection.html`, `docs/bom_tht.html`, `hardware/`, `output/` |
| PV・作品ページの素材と制作条件 | `media/README.md` |
