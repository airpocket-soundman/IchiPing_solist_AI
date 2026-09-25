# IchiPing 推論ファームウェア (DT-EBML63Q2557 / ML63Q2557)

PC から特徴量を UART で送り、Solist-AI 上で 32 クラス (扉 5 枚の開閉状態) を推論して PC の参照計算と突き合わせる
実機試験用ファーム。構成と根拠は [docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md](../../docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md)。

| 処理 | 実行場所 |
|---|---|
| N333 特徴 (log-PSD 差分 400–3000 Hz, 334 bin) の標準化・int8 量子化 | PC (`AI_INFER` で 334 B 送信。将来は Solist 上で計算) |
| int8 CNN 前段 (Conv1d → FC → 埋め込み) | ML63Q2557 CPU (`ichi_inference.c`) |
| ELM ヘッド 167→32→32 (埋め込みを 167 入力へゼロ埋め) | AxlCORE (α = Sim seed1 の 167 入力 α、実機一致確認済み) |

## 構成

| パス | 内容 |
|---|---|
| `src/ichi_protocol.c`, `include/ichi_protocol.h` | UART フレーム (COBS + CRC-32)。PC 側は `tools/ichi_serial.py` |
| `src/ichi_inference.c`, `include/ichi_inference.h` | int8 CNN 前段 (層定義表を順に実行) と ELM (AxlCORE)。ELM のみのモデルにも対応 |
| `src/ichi_app.c`, `include/ichi_app.h` | 要求処理 (HELLO / STATUS / AI_SELFTEST / AI_INFER) と LCD 表示 |
| `src/ichi_main.c` | ベンダープロジェクトの `S_System/main.c` を置き換える main |
| `generated/` | `ichiping_model.h` (モデル), `golden_outputs.json` (自己テスト期待値), `stream_cases.npz` (試験入力) |
| `prebuilt/` | 書き込み用イメージ (`*.flash.bin`) と hex |
| `tools/build.ps1` | ベンダープロジェクトの使い捨てコピーに `S_IchiPing/` を追加してビルド (元プロジェクトは変更しない) |
| `tools/flash.ps1` | MCU-Link (CMSIS-DAP) + LEXIDE 同梱 OpenOCD で書き込み・検証 |
| `tools/board_test.py` | 自己テスト + 試験入力の送信推論、結果を `docs/board_inference_test_cnn_frontend.{md,json}` へ |
| `tools/probe_alpha.py` | 実機 α の読み出し (ELM のみの probe モデル用) |

UART は 115200 bps のバイナリ要求/応答のみ。

| 要求 | 応答 |
|---|---|
| HELLO | HELLO `IchiPing` |
| STATUS | version u8, frontend u8, 入力バイト数 u16, 出力数 u8, 自己テスト数 u8, 直近の処理時間 ms u16 |
| AI_SELFTEST (case u8) / AI_INFER (入力) | AI_RESULT: case u8 (AI_INFER は 0xFE), class u8, AxlCORE µs u16, 全体 ms u16, 出力 float32 × N |

LCD には推論結果 (状態ビット a b c AB BC とクラス番号) と処理時間を表示する。

## 現在のモデル (2026-09-25)

- 前段 `small`: Conv1d 8/16/32 (k9/7/5, stride 2) → FC 1216→32。int8 重み 42 KB、活性化 2×1.3 KB。
- 学習: UNO Q train session1–8 (session4 を early stop・ハイパラ・seed 選択に使用)、周波数シフト aug ±2%。
- ビルド: text 76 KB / bss 6.6 KB。
- PC 参照精度 (int8 前段 + ELM、実機と同じ計算、校正なし): gray 87.5% / evening 61.9% / survey 89.1% / crowd 82.5%。
- 大型前段 `b1` (Conv 16/32/64/64 → FC64, 約 174 KB) も同じファームで動く (`--arch b1`)。検証精度は小型と同等 (0.796 vs 0.804)。

## 別 PC での試験手順

ビルド環境が無くても `prebuilt/` のイメージをそのまま書き込める (LEXIDE の OpenOCD と MCU-Link ドライバ、ROHM DFP は必要)。

```powershell
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\flash.ps1 -Firmware firmware\IchiPingInference\prebuilt\ichiping_frontend_small_32cls.flash.bin -Execute
```

```powershell
python firmware\IchiPingInference\tools\board_test.py --port COM3
```

- `board_test.py` は pyserial と numpy だけで動く (acrylic_pan リポジトリは不要)。
- 書き込まれたファームと `generated/` のモデルが一致しない場合は STATUS の照合で停止する。
- 確認ポイント: 自己テストのクラス一致 32/32、Board と PC 参照の argmax 一致 ≈100%、評価セット別正解率が上の PC 参照値と一致。
- 全 1056 frame の送信には時間がかかるので、まず `--stream-limit 100` で確認してもよい。

## モデルの作り直しとビルド

```powershell
D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/emit_frontend_model.py --arch small   # GPU 学習 → generated/ を更新
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\build.ps1           # PowerShell から実行
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\flash.ps1 -Firmware <build.ps1 が表示する .elf> -Execute
```

- `build.ps1` の `-SourceProject` は ROHM AIVibrationInference ベースの LEXIDE プロジェクト (Debug makefile 生成済み)。
  ベンダーのドライバ・ライブラリだけを使い、アプリケーションは `S_IchiPing/` と `S_System/main.c` に差し替える。
- `make clean` は使わない (Eclipse 生成の `.res` が消える)。
- ELM のみのモデル (α プローブ等) は `python sim/emit_board_model.py ...` で `generated/` を上書きしてからビルドする。

## α プローブ

実機の α は seed/scaleAlpha に加えて inputSize に依存する (167 入力は Sim 採取値と一致、14 入力は別物)。
`python sim/emit_board_model.py --probe <ni> --hidden <m>` でプローブ用モデルを生成・ビルド・書き込みし、
`tools/probe_alpha.py --ni <ni> --m <m>` で α を読み出す (`sim_export/alpha_probe/`)。
hidden=64 では 63 番ユニットの出力が常に 0 になるため、`emit_board_model.py` は α=0 の列を β から除外する。

## 過去の結果

- ELM のみ 167→32→14 (Sim α): [docs/board_inference_test.md](../../docs/board_inference_test.md) (2026-09-24) evening 64.4%
- 線形前段 + ELM 14→64→14 (プローブ α): [docs/board_inference_test_frontend.md](../../docs/board_inference_test_frontend.md) (2026-09-25) evening 96.2%, PC 一致 100%
  (いずれも旧 acrylic_pan オーバーレイ版ファームでの結果)
