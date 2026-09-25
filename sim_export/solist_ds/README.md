# Solist-AI 向け学習データ (UNO Q + IchiPing データセット, 周波数シフト augmentation)

生成: `python sim/make_solist_dataset.py --sources unoq --shift ir --n-shift 2` (2026-09-24)
評価: `python sim/eval_solist_dataset.py` → [EVAL.md](EVAL.md)

## 特徴量 (実機と同一)

`feature_schema_id = tdiff-rfft1024-h512-symhann-magmean-log20-floor80-bin26-192-f32-v1`
16 kHz mono 2 s PCM → **時間波形で baseline (全閉平均) を減算** → 1024 点 FFT (hop 512, Hann) →
|X| を 61 窓平均 → 20log10 → floor −80 dB → DC 除外 → bin 26..192 (406.25..3000 Hz) の **D=167**。
`sim/bench_v612.py` のキャッシュと完全一致 (誤差 0) を `--selftest` で確認済み。

## ソース

| domain | 学習 run | 評価セット (学習に混ぜない) |
|---|---|---|
| `unoq` | `D:/GitHub/IchiPing-UNO-Q/pc/captures/uno_q_train_20260912_session1..8*_wav` (8 セッション, 13,200 frame, 2026-09-12 08:50–19:19) | `uno_q_eval_20260912_{gray,evening,survey,crowd}_wav` (raw から `pc/uno_q_export_dataset.py` で書き出し) |
| `frdm` | `D:/GitHub/IchiPing/pc/captures/full_32_train_v21..v25` (FRDM 世代, 8,000 frame) | `full_32_eval_v1` |

評価セットの baseline は各セットの `group=="baseline"` frame (電源投入時校正の想定)。

## 気温差 (周波数シフト) 対策

UNO Q で有効だった「学習時に周波数を ±3 % シフトする」対策 (`pc/training/dataset.py warp_logmag_psd`,
夕方ドリフト −2.15 %) を Solist パイプライン向けに移植した。Solist は baseline を時間波形で引くため
dB スペクトルのシフトを流用できず、`sim/freq_shift.py` で **IR シフト** を実装した。

1. 既知 PRBS (seed 20260912) で正則化逆畳み込みしてインパルス応答 h を推定 (再構成 SNR 27.6 dB)
2. h(t) → h(t·(1+ε)) と時間伸縮 (= 共鳴周波数 (1+ε) 倍、音速変化そのもの)
3. `a_w = a + prbs ⊛ (h_w − h)` で再合成し、**シフトしていない baseline** で diff → 「古い baseline で
   推論する」失敗モードをそのまま再現

検証 (`--selftest`): +3 % シフト後のスペクトルから推定した ε=+3.1 %。実データの session6→夕方ドリフトは
ε=−1.8 % (UNO Q 報告 −2.15 %)、→survey −0.85 % (−1.05 %)。全閉 frame の diff レベルは同セッション −20.5 dB、
実夕方 −5.9 dB、IR シフト後 −7.7 dB で失敗モードを再現できている。

励振が再現不能な FRDM 世代は diff スペクトルの周波数軸伸縮 (`--shift feat`) を使う (stale-baseline 項は再現されない)。

## ファイル

| ファイル | 内容 |
|---|---|
| `train_<variant>_14cls_5k.csv` / `_32cls_5k.csv` | Solist-AI Sim 用学習 CSV (標準化済 167 特徴 + one-hot, 層化サブサンプル, ≤1,000,000 セル) |
| `test_<evalset>_<variant>_{14,32}cls.csv` | 同形式の評価 CSV (学習統計で標準化) |
| `norm_<variant>.npz` | 標準化統計 mu/sd (167) |
| `MANIFEST_<variant>.json`, `BUILD_<variant>.log` | レシピと生成ログ |
| `sim/_cache/solist_ds_<variant>.npz` (git 管理外) | 全学習行 (float32, 未標準化) + 評価セット + メタ (run/baseline/ε) |
| `board_model_<variant>_14cls.npz` | 実機用 β / mu / sd / scale / α (`sim/emit_board_model.py`) |

variant: `unoq_none` (シフト無し), `unoq_ir2` (IR シフト ×2 コピー, ±3.5 %), `unoq_feat2`, `unoq+frdm_ir2`。

Sim 設定: Input First col=1 / rows=1 / cols=167、Expected First col=168 / cols=14 (32cls は 32)、
Normalize OFF (標準化済)、Hidden=32 (実機 α=seed1)、Hard sigmoid / MSE、scaleAlpha≈0.205。

## 評価結果の要点 (EVAL.md)

- **m=32 (実機 α) の ELM では 14cls 別時間帯評価が 65–88 % に留まり、シフトの有無で差が出ない**
  (unoq_none: gray 87.5 / evening 70.0 / survey 78.1 / crowd 71.9 %)。容量律速。
- m=256 (乱数 α) では evening が none 85.6 % → feat 90.0 % / unoq+frdm+ir 93.8 % と改善し、
  シフト augmentation の効果が現れる。UNO Q の CNN (104k param) は同条件で 100 %。
- 実機ライブラリ (`SolistAi_Library_2_256_64.a`) の上限は hidden 64。m=64 の α を Sim から採取すれば
  中間の検証ができる。
