# 実機用モデル(int8 CNN 前段 + ELM)と評価結果

完成版モデルと、その構成・選定の根拠になった評価結果。経緯と設計判断は [docs/DEVELOPMENT.md](../../docs/DEVELOPMENT.md)。

## モデル

| ファイル | 内容 |
|---|---|
| `board_model_frontend_32cls_best_s0.005.npz` | **完成版**。UNO Q 8 セッション + Stamp 13 セッション、周波数シフト ±0.5%、cross-baseline 2、前段 small。実機 32 クラス 100% |
| `BOARD_FRONTEND_PC.json` | 完成版の学習条件・PC 評価・sha256 |
| `board_model_frontend_32cls.npz` | UNO Q のデータだけで学習した最初のモデル(実機 75.0%)。`sim/eval_stamp_mix.py` の比較用 |

## 再生成

PyTorch が要るので IchiPing 側の GPU 用 venv(`D:/GitHub/IchiPing/pc/.venv`)で実行する。学習データは git 管理外:
UNO Q 版の収録(`D:/GitHub/IchiPing-UNO-Q/pc/captures`)と、このハードの収録(`captures/stamp_*_wav`, `firmware/IchiPingInference/tools/collect_session.py`)。

```powershell
$S = "stamp_20260926_s1_wav stamp_20260926_s2_wav stamp_20260926_s3_wav stamp_20260926_s4_wav stamp_20260926_s5_wav " +
     "stamp_20260926_s6_wav stamp_20260926_s7_wav stamp_20260926_s8_wav stamp_20260926_s9_wav " +
     "stamp_20260926_s10_wav stamp_20260926_s11_wav stamp_20260927_s1_wav stamp_20260927_s2_wav"
# 1) 前段 + ELM を学習し、generated/ichiping_model.h と ichi_feature_tables.h を書く
python sim/emit_frontend_model.py --arch small --stamp $S.Split(" ") --val stamp_20260927_s2_wav --shift 0.005 --xbase 2 --tag best_s0.005
# 2) 現地校正の初期 P を書く(generated/ichi_calib_prior.h)
python sim/emit_calibration_prior.py --model sim_export/solist_ds/board_model_frontend_32cls_best_s0.005.npz --stamp $S.Split(" ")
# 3) 固定小数点の特徴計算が PC 参照と合うか確認(--emit で ichi_feature_tables.h を書き直す)
python sim/board_fixed_feature.py
```

- `--tag` を省くと `board_model_frontend_32cls.npz`(上の比較用モデル)を上書きするので必ず付ける。
- その後 `firmware/IchiPingInference/tools/build.ps1` でビルドし、`survey_monitor.py` で実機サーベイする。

## 評価結果(根拠)

| ファイル | スクリプト | 内容 |
|---|---|---|
| [BEST_MODEL.md](BEST_MODEL.md), `best_model/results.jsonl` | `sim/eval_best_model.py` | 事前学習モデルの選定(条件グループ hold-out:データ・シフト幅・cross-baseline・前段) |
| [STAMP_MIX.md](STAMP_MIX.md) | `sim/eval_stamp_mix.py` | このハードの収録を学習に加えた効果(セッション単位 leave-one-out) |
| [ODL_CALIBRATION.md](ODL_CALIBRATION.md) | `sim/eval_odl_calibration.py` | 現地校正の効果と、float32 / bf16、校正する状態数の比較 |
| [RUN_SHIFT.md](RUN_SHIFT.md), `RUN_SHIFT_drift.png` | `sim/estimate_run_shift.py` | 全閉の収録から推定したセッション・時刻ごとの周波数シフト(気温) |
| [FULL_DATA.md](FULL_DATA.md) | `sim/eval_full_data.py` | FRDM + UNO Q 全データでの前段 + ELM(大型前段、校正 5 窓) |
| [TINY_FRONTEND_LODO.md](TINY_FRONTEND_LODO.md) | (探索用、削除済み) | ML63Q2557 向け小型前段の比較(Flash・活性化・MAC 併記)。small を採用 |
| [CNN_FRONTEND_LODO.md](CNN_FRONTEND_LODO.md) | (探索用、削除済み) | PC の CNN XL 前段(約 30 万パラメータ)+ ELM |
| [IMPROVE_LODO.md](IMPROVE_LODO.md) | `sim/eval_improve_lodo.py` | 特徴(D167 / N333 / N1024)× 線形 / ELM、日単位 leave-one-out |
| [IDEAL_VS_SOLIST.md](IDEAL_VS_SOLIST.md) | `sim/eval_ideal_vs_solist.py` | PC 理想モデルと Solist ELM の比較(周波数シフト水増し込み) |

`sim/make_solist_dataset.py` は Sim 段階の D167 データセット(Solist-AI Sim 用 CSV、`sim/_cache/`)を作る。CSV は再生成できるので git 管理外。
