# Solist-AI 実機推論テスト (2026-09-25T00:08:17)

- ボード: DT-EBML63Q2557 (ML63Q2557), UART COM3 115200 bps, MCU-Link CMSIS-DAP で書込み
- モデル: ichiping_unoq+frdm_ir2_ridge_14x64x14 (前段: PC学習 線形ridge 167→14 + z-score ×4, ELM 14→64→14, hard sigmoid, bfloat16), α=実機プローブ値 (sim_export/alpha_probe/alpha_ni14_m64.npz), scaleAlpha=0x3E52, 入力scale=4.0
- STATUS: (3, 15, 2048, 64, 25600, 0)

## 1. 内蔵自己テスト (AI_SELFTEST 0x14, クラス別 14 ベクトル)

- クラス一致 14/14, 許容内 14/14, 最大スコア誤差 0.0317 (許容 abs 0.035 / rel 0.05)

| case | 期待cls | PC cls | Board cls | max abs err | 判定 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0234 | PASS |
| 1 | 1 | 1 | 1 | 0.0137 | PASS |
| 2 | 2 | 2 | 2 | 0.0317 | PASS |
| 3 | 3 | 3 | 3 | 0.0132 | PASS |
| 4 | 4 | 4 | 4 | 0.0178 | PASS |
| 5 | 5 | 5 | 5 | 0.0199 | PASS |
| 6 | 6 | 6 | 6 | 0.0232 | PASS |
| 7 | 7 | 7 | 7 | 0.0171 | PASS |
| 8 | 8 | 8 | 8 | 0.0254 | PASS |
| 9 | 9 | 9 | 9 | 0.0156 | PASS |
| 10 | 10 | 10 | 10 | 0.0254 | PASS |
| 11 | 11 | 11 | 11 | 0.0166 | PASS |
| 12 | 12 | 12 | 12 | 0.0176 | PASS |
| 13 | 13 | 13 | 13 | 0.0181 | PASS |

## 2. PC からの特徴ベクトル送信推論 (AI_INFER 0x16, 評価セット unoq_evening 320 frame)

- Board vs PC(bf16参照) argmax 一致: 100.0%
- Board vs PC(float32) argmax 一致: 99.7%
- 正解率 (frame): Board 96.2% / PC bf16 96.2% / PC float32 95.9%
- 正解率 (状態別投票): Board 96.9%
- スコア誤差: max 0.0391, mean 0.0065, p99 0.0234, 許容内 100.0%
- 往復時間 (UART 334B 送信 + 推論 + 60B 受信): 平均 62.2 ms, 最大 64.8 ms
- アクセラレータ推論時間 (SysTick 実測, 14→64→14): 平均 245 us, 最大 245 us

PC bf16 参照の top-2 マージン別 argmax 一致率 (不一致が僅差ケースに限られるかの確認):

| margin | frames | 一致率 |
|---|---|---|
| 0-0.02 | 4 | 100.0% |
| 0.02-0.05 | 5 | 100.0% |
| 0.05-0.1 | 9 | 100.0% |
| 0.1-inf | 302 | 100.0% |
