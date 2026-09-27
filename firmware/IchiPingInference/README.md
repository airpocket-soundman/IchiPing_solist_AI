# IchiPing ファームウェア(DT-EBML63Q2557 / ML63Q2557)

Solist-AI 評価ボードで動く IchiPing 本体。Stamp-S3A([../StampMeasure](../StampMeasure))から I²C で録音を受け取り、
特徴抽出・int8 CNN 前段(CPU)・ELM(AxlCORE)・現地校正(CPU float32 → AxlCORE)・TFT 表示・サーボ制御を行う。
構成と設計判断は [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md)。

## main(`tools/build.ps1 -Main`)

| main | 用途 | PC 側ツール |
|---|---|---|
| `ichi_infer_main`(既定) | 製品の動作。トグル → EXEC で 1 回推論、EXEC 2 s 長押しで現地校正 | `tools/infer_monitor.py`(任意。ログ記録、PC から推論を起動) |
| `ichi_survey_main` | サーボで 32 状態を巡回して推論・採点する実機サーベイ(校正あり / なし) | `tools/survey_monitor.py` → `docs/results/board_survey.{md,json}` |
| `ichi_collect_main` | 学習データ採取。PC の指示でサーボを動かす | `tools/collect_session.py`(Stamp の USB から録音を保存) |

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
| `prebuilt/ichiping_{infer,survey,collect}.flash.bin` | 書き込み用イメージ(完成版モデル、2026-09-27 ビルド) |

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
  途中で EXEC を押すと中断して事前学習の β に戻る。校正後の β は電源を切るまで有効。
- **サーベイ**:`python firmware/IchiPingInference/tools/survey_monitor.py --port COM3`(PC から 'G' で開始、'X' で中断。`--wait-exec` なら EXEC で開始)。
- **データ採取**:`python firmware/IchiPingInference/tools/collect_session.py --name s1 --solist COM3 --stamp COM13 [--rotate N] [--reverse]`
  → `captures/stamp_<日付>_<name>_wav/`(全閉 10 + 32 状態 × 10 フレーム、Gray code 順、欠けたら撮り直し)。
