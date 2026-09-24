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

## 結果

- 167→32→14 (Sim α): [docs/board_inference_test.md](../../docs/board_inference_test.md) (2026-09-24) evening 64.4%
- 線形前段 + 14→64→14 (プローブ α): [docs/board_inference_test_frontend.md](../../docs/board_inference_test_frontend.md) (2026-09-25) evening 96.2%, PC 一致 100%
