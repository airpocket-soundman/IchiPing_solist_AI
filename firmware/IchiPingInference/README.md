# IchiPing 推論部の実機検証ファームウェア (DT-EBML63Q2557 / ML63Q2557)

`D:/GitHub/acrylic_pan/firmware/AcrylicPanCollector` のオーバーレイ構成 (LEXIDE 私的プロジェクト
`AcrylicPanCollector_lowlatency` + `S_AcrylicPan`) をそのまま流用し、次だけ差し替える。

| ファイル | 変更 |
|---|---|
| `include/apan_ai_selftest.h`, `src/apan_ai_selftest.c` | 167 入力 / 32 hidden / 14 出力の IchiPing モデル。`ApanAiInfer()` (PC 送信ベクトル推論) と推論時間計測を追加 |
| `generated/ichiping_model.h` | `python sim/emit_board_model.py` が生成。β (bf16 32×14)、自己テスト 14 ベクトル、期待クラス |
| `include/apan_protocol.h`, `src/apan_protocol.c` | コマンド payload 上限 16→352 B (`AI_INFER` 0x16 = 167 × bfloat16)、デコーダ作業領域を static 化 |
| `integration/apan_collector_app.c` | `AI_RESULT` を `<BBH14f>` に拡張、`AI_INFER` ハンドラ追加、予約 u16 にアクセラレータ推論時間 (µs) |

α は `ODL_SetWeightAlpha` が stub のため転送せず、`seed=1, scaleAlpha=0x3E52` からアクセラレータが再生成する
(PC 側は `sim_export/_alpha32_sim.npy` = Sim seed1 採取値を使用)。

## 手順

```powershell
python sim/emit_board_model.py --variant unoq_ir2 --eval unoq_evening --scale 0.25
powershell -ExecutionPolicy Bypass -File firmware/IchiPingInference/tools/build.ps1      # PowerShell から実行 (Git Bash 経由は make が echo を見失う)
powershell -ExecutionPolicy Bypass -File D:\GitHub\acrylic_pan\scripts\flash-firmware.ps1 -FirmwareHex <hex> -Execute   # MCU-Link CMSIS-DAP + LEXIDE 同梱 OpenOCD
C:/ProgramData/anaconda3/python.exe firmware/IchiPingInference/tools/board_test.py --port COM3
```

`build.ps1` は私的プロジェクトを `.local/firmware-build/<stamp>/` に複製して make するので元プロジェクトは変更しない。
`make clean` は Eclipse 生成の `.res` を消すので使わない。

## α プローブ (実機の α を直接読む)

実機の α は seed/scaleAlpha に加えて **inputSize に依存**する (167 入力は Sim 採取値と一致、14 入力は別物)。
`python sim/emit_board_model.py --probe <ni> --hidden <m>` で β=単位行列・出力=hidden のファームを作り、
`tools/probe_alpha.py --ni <ni> --m <m>` が ±c·e_i を送って α_ij を読み出す (`sim_export/alpha_probe/`)。
hidden=64 では 63 番ユニットの出力が常に 0 になるため、`emit_board_model.py` は α=0 の列を β から除外する。

## 線形前段 + ELM (現行の最良構成)

`python sim/emit_board_model.py --variant unoq+frdm_ir2 --frontend ridge --hidden 64 --scale 4.0 --alpha-file sim_export/alpha_probe/alpha_ni14_m64.npz`
PC 学習の ridge 線形段 (167→14, z-score) を Stamp-S3A か Solist CPU で計算し、14 値を ELM (14→64→14) に渡す。
`board_model_<variant>_14cls.npz` に W / ms / ss / beta / alpha を保存する。

## CNN 前段 + ELM ヘッド 32 クラス (2026-09-25 時点の推奨構成・実機試験用)

構成の根拠は [docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md](../../docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md)。

| 処理 | 実行場所 |
|---|---|
| N333 特徴 (log-PSD 差分 400–3000 Hz, 334 bin) の標準化と int8 量子化 | PC (`AI_INFER` で 334 B 送信。将来は Solist 上で計算) |
| Conv1d 8/16/32 (k9/7/5, stride 2, BN 融合, ReLU) → FC 1216→32 (ReLU)、int8 重み・int8 活性化 | ML63Q2557 CPU (`apan_ai_selftest.c` の `run_frontend`) |
| 埋め込み 32 を標準化・bf16 化し 167 入力へゼロ埋め → ELM 167→32→32 | AxlCORE (α = Sim seed1 の 167 入力 α、実機一致確認済み) |

- 前段の作業領域 (int8 活性化 2 面 = 2.6 KB) は、コレクタ停止中は未使用の KX134 `capture` バッファを流用し、推論後に `ApanCaptureReset` する。
- ビルド結果 (2026-09-25): text 104 KB / bss 13.2 KB (スタック 1.25 KB との間に約 1.9 KB の空き)。
- 学習 = UNO Q train session1–8 (session4 を early stop・ハイパラ選択に使用)、評価 = UNO Q eval 4 セット (学習に不使用)。
  PC 参照精度 (int8 前段 + ELM, 実機同等計算): gray 86.5% / evening 57.2% / survey 65.9% / crowd 70.3% (校正なし)。
  小型前段 (RAM 制約) のため、全データ評価の大型前段 (約 78–81%) より低い。大型前段は KX134 系バッファを外して RAM を空けてから。

### 別 PC での試験手順

```powershell
# 1) ビルド済み hex をそのまま書き込む (ビルド環境が無い場合)
powershell -ExecutionPolicy Bypass -File D:\GitHub\acrylic_pan\scripts\flash-firmware.ps1 -FirmwareHex firmware\IchiPingInference\prebuilt\IchiPing_cnn_frontend_32cls.hex -Execute
#    (ビルドする場合は tools\build.ps1。generated/ichiping_model.h は生成済みでコミットしてある)
# 2) 自己テスト 32 ケース + UNO Q eval 1056 frame の送信推論
C:/ProgramData/anaconda3/python.exe firmware/IchiPingInference/tools/board_test.py --port COM3
#    結果: docs/board_inference_test_cnn_frontend.{md,json}
```

`board_test.py` は `generated/golden_outputs.json` の metadata (`input_format=int8_frontend`, `output_count=32`) を見て
int8 入力・32 出力に切り替わる。確認ポイント: 自己テストのクラス一致 32/32、Board と PC bf16 参照の argmax 一致率 (≈100% が目標)、
評価セット別正解率が上の PC 参照値と一致すること。

モデルの再生成: `D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/emit_frontend_model.py` (GPU 学習、特徴キャッシュは `sim/eval_full_data.py` と共用)。
以前の ELM のみのモデルに戻す場合は `python sim/emit_board_model.py ...` を実行してから build する。

## 結果

- 167→32→14 (Sim α): [docs/board_inference_test.md](../../docs/board_inference_test.md) (2026-09-24) evening 64.4%
- 線形前段 + 14→64→14 (プローブ α): [docs/board_inference_test_frontend.md](../../docs/board_inference_test_frontend.md) (2026-09-25) evening 96.2%, PC 一致 100%
