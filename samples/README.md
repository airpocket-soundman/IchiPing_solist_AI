# 音声サンプル

## uno_q_eval_evening/

UNO Q (INMP441 マイク / MAX98357A アンプ, PRBS 励振) で 2026-09-12 夕方に収録した評価セット
(`IchiPing-UNO-Q/pc/captures/uno_q_eval_20260912_evening_wav`) の一部。16 kHz mono int16、1 frame = 2 秒。

| フォルダ | 内容 |
|---|---|
| `baseline/` | 起動時 baseline (全閉) 10 frame。特徴量はこの平均 log-PSD との差分で計算する |
| `sXXXXX/` | 各状態の先頭 1 frame (32 状態)。状態名は扉 a b c AB BC の開閉 (1 = 開)、クラス番号は bit k = 2^k |
| `manifest.json` | 元ファイル・クラス番号・収録時刻 |

32 状態の frame は、実機ファーム (`firmware/IchiPingInference`) の自己テスト 32 ケースの元データ。
`python sim/export_audio_samples.py --verify-only` で、この wav だけから計算した int8 入力が
`generated/ichiping_model.h` の自己テスト入力と 32/32 一致することを確認できる
(学習時の特徴キャッシュが float16 のため、検証でも float16 を経由する)。
