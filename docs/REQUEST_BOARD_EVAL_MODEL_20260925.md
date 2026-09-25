# 実機評価用モデル一式の依頼（学習 PC → 実機 PC, 2026-09-25）

目的: `HANDOFF_SOLIST_CNN_FRONTEND_20260925.md` §2 の確定構成を実機に載せ、
**Stamp-S3A から本番と同じ経路（16 kHz mono PCM16 を I²C で Solist へ）でダミー音を入力**し、
Solist 上の推論精度が PC の予想値どおり出るかを確かめる。学習 PC で下記を作って push してほしい。

実機側の現状（2026-09-25 確認済み）: Solist ⇔ 中間基板 ⇔ Stamp の I²C 配線、Stamp=0x42 への書き込み
（46/46 ACK・欠落なし）、LCD とスイッチ。Solist 側 I²C 読み出しは未実装（実機 PC で実装する）。

## 1. モデルの選定（学習 PC の推奨で可）

- 第一候補: **T8-16-32 FC32 + ELM m32**（int8 41.5 KB, 活性化 2.5 KB）。まず経路全体の正しさを確認するため。
- 余裕があれば B1/B2（≈170 KB）も同じ形式で追加してよい。実機 Flash は 256 KB、現行ファーム本体は約 12 KB。
- 学習データ・評価日（テストに使う日/セット）は学習 PC で決め、`manifest.json` に明記する。
  テストに使う日/セットは学習・標準化・ハイパラ選択・early stop に使わない（HANDOFF §3 の評価ルール）。

## 2. 置き場所

`sim_export/board_eval/<model_id>/`（`model_id` 例: `t8_16_32_fc32_m32_frdm_lodo_v21`）

## 3. 必要なファイル

### 3.1 `manifest.json`

- model_id、学習データ（run 一覧）、テスト日/セット、seed、作成スクリプトとコミット
- 以下すべてのファイル名と形状、dtype、エンディアン（little endian）

### 3.2 特徴抽出の厳密な仕様 `feature_spec.json`（Solist で同じ計算をするため）

- 入力: サンプルレート、1 frame のサンプル数、使う区間（先頭/末尾の除外があれば）
- FFT 長 2048、Welch の hop、窓関数（種類と対称/周期）、平均の取り方、PSD のスケール（片側、係数）
- log の底と ε、dB 変換の有無
- baseline: どの frame から、何 frame を、どう平均するか（起動時 s00000 相当）
- frame 正規化: 平均/分散の対象範囲（333 bin か全 bin か）、ε
- 400–3000 Hz → bin index の開始・終了（両端を含むか）
- 前段入力の量子化: scale, zero_point（int8）

### 3.3 前段 CNN（int8, BN は融合済み）

- `frontend_arch.json`: 層の順序、種類（conv1d / pool / fc）、in/out ch、kernel、stride、padding、活性化、flatten 順
- 層ごとの `w_int8`（レイアウトを明記）、`b_int32`、再量子化パラメータ（per-channel の multiplier と shift、
  または float scale）、出力 zero_point
- PC 上で **int8 整数演算をそのまま再現する参照実装**（numpy、`sim/board_ref_frontend.py` など）。
  実機で出力がビット単位で一致するか確かめるのに使う。
- float 版の重み（参考）

### 3.4 ELM ヘッド

- 入力（埋め込み）の標準化 mu/sd と入力スケール、hidden m、活性化（hard sigmoid）、λ
- **β を再計算するための学習側データ**: 学習 frame の埋め込み（N×32 float32）とラベル（y32, y14）
  - α は実機の生成値を `probe_alpha.py` で取得してから（入力 32 次元・m32）、実機 PC で β を作り直す。
    そのため β の計算スクリプトを α ファイルを入力にとる形で用意してほしい（`emit_board_model.py --alpha-file` と同じ形）。
- 参考として、一様乱数 α で計算した β と、その条件での PC の精度

### 3.5 テスト音声（Stamp の Flash に入れて I²C で流す）

- **16 kHz mono PCM16（Stamp の 48→16 kHz 変換の後段と同じ形）**、`clips/<run>/<state>/frame_xxx.wav` か連結 `.bin` と索引 json
- 内容: テスト日/セットの baseline 用 s00000 frame（baseline 計算に使う分）＋ 32 状態 × 各 2 frame 程度
  （Stamp Flash の空き約 6 MB、1 frame = 64 KB）。baseline に使った frame は評価から除外
- 各クリップの正解ラベル（y32, y14）

### 3.6 正解値（ビット一致・精度の照合用）`golden.npz`

クリップごとに:
- N333 特徴（float32）と、量子化後の前段入力（int8）
- 前段の出力埋め込み（int8 と float）
- ELM の出力 32 点と予測クラス（一様乱数 α 版。実 α 版は実機 PC で再計算）

### 3.7 予想精度 `expected.md`

- このテストクリップ集合での精度: PC float、int8 シミュレーション、ELM（乱数 α）。校正なし・校正ありを分けて
- 32 クラスと 14 クラス換算

## 4. 実機 PC 側で実装するもの（参考）

1. Solist の I²C 読み出し。Stamp から PCM16 を chunk 単位で受け取る（プロトコルは実機 PC で定義し、CONNECTIONS に追記）
2. Stamp: テストクリップを Flash から読み、本番と同じ PCM16 chunk として I²C で供給する
3. Solist: FFT（HW）→ Welch → log-PSD → baseline 差分 → 正規化 → N333 → int8 前段 → ELM
4. UART（COM18）で中間値を PC に返し、`golden.npz` と照合してから、全クリップで精度を測る
