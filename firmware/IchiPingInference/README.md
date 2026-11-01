# IchiPing ファームウェア(DT-EBML63Q2557 / ML63Q2557)

Solist-AI 評価ボードで動く IchiPing 本体。Stamp-S3A([../StampMeasure](../StampMeasure))から I²C で録音を受け取り、
特徴抽出・int8 CNN 前段(CPU)・ELM(AxlCORE)・現地校正(CPU float32 → AxlCORE)・TFT 表示・サーボ制御を行う。
構成と設計判断は [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md)。

## 現在の完成状態（2026-11-01）

- 標準の生成ヘッダと `prebuilt/ichiping_infer.flash.bin` は、実機確認済みのフル21セッション・クロス16モデル。
  Stampのフルセッションのみを使い、窓C限定セッションとUNO Q学習音声は除外。帯域は約100〜2700 Hz、334入力、周波数シフト±1%、追加ノイズなし。
- クロスベースラインの参照元は元の16セッションに固定。学習・検証セッションの一覧は
  [モデル情報](../../sim_export/solist_ds/BOARD_FRONTEND_PC.json)に記録。
- 実機ODLの160更新後、P・β・基準音スペクトルをPCへ取得。評価専用5セッション1600件の32クラス精度は
  ODL前1589/1600（99.3125%）、ODL後1593/1600（99.5625%）。14クラスは前後とも100%。
  各評価セッション自身の基準音を使ったPC評価であり、実機サーベイの精度ではない。
- 通常推論用ファームへの切り替え後、保存ODLの復元と1回の正しい実機推論を確認。
  初回・電源投入時は全閉の基準音を3回取り直す。通常推論はEXEC短押しで1回。
- [ODL状態NPZ](../../docs/results/full21_device_odl_state_20261101.npz)と
  [PC比較結果](../../docs/results/full21_pc_odl_comparison_20261101.json)を保存。
  フラッシュイメージには各個体のFRAM状態を含まないため、新しいデバイスではODLを実施する。
- PCM重複転送を減らすFRAMキャッシュ案は `experimental/ichi_feature_overlap.c` に保管。
  標準ビルドは今回の実機・ODL検証で使用した特徴抽出方式と一致させている。

## main(`tools/build.ps1 -Main`)

| main | 用途 | PC 側ツール |
|---|---|---|
| `ichi_infer_main`(既定) | 製品の動作。トグル → EXEC で 1 回推論、EXEC 2 s 長押しで現地校正 | `tools/infer_monitor.py`(任意。ログ記録、PC から推論を起動) |
| `ichi_survey_main` | サーボで 32 状態を巡回して推論・採点する実機サーベイ(校正あり / なし) | `tools/survey_monitor.py` → `docs/results/board_survey.{md,json}` |
| `ichi_collect_main` | 学習データ採取。PC の指示でサーボを動かす | `tools/collect_session.py`（Solist経由でPCMを保存、Stamp USB不要） |

## ソース

| パス | 内容 |
|---|---|
| `src/ichi_infer_main.c`, `ichi_survey_main.c`, `ichi_collect_main.c` | 各 main(ベンダープロジェクトの `S_System/main.c` を置き換える)。UART のメッセージ種別は各ファイル冒頭 |
| `src/ichi_stamp_link.c` | Stamp-S3A との I²C(0x42):STATUS / MEASURE / READ |
| `src/ichi_feature.c` | N333 特徴(2048 点 FFT int16 ブロック浮動小数点、Welch、dB、全閉差分、正規化、int8) |
| `src/ichi_inference.c` | int8 CNN 前段、ELM(AxlCORE)、現地校正(OS-ELM float32、P・β は FRAM) |
| `src/ichi_tft.c`, `ichi_ui.c` | ILI9341(ソフト SPI)と画面表示 |
| `src/ichi_servo.c` | PCA9685 + SG90 ×5(OE は P42) |
| `src/ichi_protocol.c` | UART フレーム(COBS + CRC-32)。PC 側は `tools/ichi_serial.py` |
| `generated/ichiping_model.h` | 完成版モデル(CNN int8 重み・層定義、ELM β bf16) |
| `generated/ichi_feature_tables.h` | 特徴計算の窓・余弦表と入力標準化(モデルごと) |
| `generated/ichi_calib_prior.h` | 現地校正の初期 P(事前学習の Gram 行列) |
| `prebuilt/ichiping_infer.flash.bin` | 現行クロス16モデル、ODL保存・復元対応の推論用イメージ（2026-11-01） |
| `prebuilt/ichiping_collect.flash.bin` | Solist経由採取・UART 500000 bpsの動作確認済みイメージ |
| `prebuilt/ichiping_survey.flash.bin` | 過去モデルのサーベイ用。現行モデルを測る場合は再ビルドする |

generated/ は `sim/emit_frontend_model.py` と `sim/emit_calibration_prior.py` が作る(手順は sim_export/solist_ds/README.md)。

## 書き込み

ビルド環境が無くても prebuilt/ のイメージを書き込める(LEXIDE 同梱の OpenOCD、MCU-Link ドライバ、ROHM DFP が必要)。

```powershell
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\flash.ps1 -Firmware firmware\IchiPingInference\prebuilt\ichiping_infer.flash.bin -Execute
```

Stamp-S3A には `firmware/StampMeasure` を PlatformIO で書き込む(`pio run -t upload`。PowerShell では UTF-8 の出力設定が必要)。

## ビルド

```powershell
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\build.ps1 -Main ichi_infer_main
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\flash.ps1 -Firmware <build.ps1 が表示する .elf> -Execute
```

- `-SourceProject` は ROHM AIVibrationInference ベースの LEXIDE プロジェクト(Debug の makefile 生成済み)。
  ベンダーのドライバ・ライブラリだけを使い、使い捨てのコピー(`.local/firmware-build/`)に IchiPing のソースを入れてビルドする。元のプロジェクトは変更しない。
- `make clean` は使わない(Eclipse が生成した `.res` が消える)。
- 書き込み用の flash.bin は flash.ps1 が .elf から作る(RAM 領域を含む HEX ではなく Flash 部分だけ)。

## 使い方

- **推論**(`ichi_infer_main`):電源投入でサーボを全閉にし、全閉を 3 回測って baseline にする。トグルで窓・扉を決めて EXEC を押すと、
  Listening(PRBS 2 s)→ Inferring(ゲージ)→ 結果。画面の 5 桁は左から c・BC・b・AB・a(1 = 開)、扉の奥で本来聞こえない桁は暗く表示。
  判定は Complete Success(32 クラス一致)/ Conditional Success(14 クラス一致)/ Failure。
- **現地校正**:EXEC を 2 s 長押し。扉と窓の開閉の組み合わせ全 32 クラスを巡回し(各 6 s の PRBS を 2 s × 5 窓)、β を逐次更新(約 30 分)。
  途中で EXEC を押すと中断して事前学習の β に戻る。成功時はP・βをFRAMへ保存し、モデル識別情報とチェックサムが一致する場合に起動時復元する。
- **サーベイ**:`python firmware/IchiPingInference/tools/survey_monitor.py --port COM3`(PC から 'G' で開始、'X' で中断。`--wait-exec` なら EXEC で開始)。
- **データ採取**:`python firmware/IchiPingInference/tools/collect_session.py --name s1 --solist COM3 [--rotate N] [--reverse]`
  → `captures/stamp_<日付>_<name>_wav/`(全閉 10 + 32 状態 × 10 フレーム、Gray code 順、欠けたら撮り直し)。`ichi_collect_main` はStampへI²Cで測定を指示し、PCMをSolist UARTでPCへ中継するため、StampのUSB接続は不要。

## 実機ODL状態をPCで評価する

`build.ps1 -Main ichi_survey_main -SurveyCalibrate -SurveyExportOdl` で、サーベイをせずODLと状態送信だけを行う。
`-ModelGeneratedDir` で指定したモデルのヘッダ・特徴テーブル・ODL初期行列を揃える。

```powershell
python firmware/IchiPingInference/tools/export_odl_state.py --port COM3 --model <model.npz> --identity-header <build>/S_IchiPing/ichi_model_identity.h --out docs/results/odl_state
python sim/evaluate_exported_odl.py --model <model.npz> --odl docs/results/odl_state.npz --sessions <評価セッション名...> --out docs/results/odl_comparison.json
```

UARTは115200 bps。モデル識別情報、80チャンクの順序、10240バイト全体のチェックサムを検証し、
float32のP・β各4096バイトとint16の基準スペクトル2048バイトを保存する。
主評価 `session_baseline` と、実機ODL時の基準音を共通使用する補足評価 `device_odl_baseline` は分けて報告する。
