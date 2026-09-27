# sim_export

| パス | 内容 |
|---|---|
| [solist_ds/](solist_ds/README.md) | **完成版**の実機用モデル(int8 CNN 前段 + ELM)、再生成手順、評価結果 |
| `model1_BEST_14cls/`, `model1_BEST_32cls_factory/`, `model1_BEST_32cls_ondevice/` | Sim 段階(2026-06)に公式 Solist-AI Sim(SLV1.00.04)へ読み込ませた ELM モデル(D167, m=32)。フォルダ名の factory は事前学習のみ、ondevice は現地データ追加 |
| `test_best_14cls.csv`, `test_BEST_32cls_*.csv` | 上の Sim モデルの評価用 CSV |
| `_alpha32_sim.npy` | Sim(seed 1)から採取した α(167×32)。実機も 167 入力なら同じ α を再生成する(完成版も使用) |
| `_best_models.pkl`, `_best.txt` | Sim 段階の最良 β の記録 |

Sim 段階の結果(14 クラス 99.4%、32 クラス 99.1%(現地データ追加後))は同じ時期に収録した別セッションでの値で、気温や背景音の条件が異なるセッションへの汎化は含まない。
報告は [docs/index.html](../docs/index.html)。Sim 段階のスクリプトは `sim/bench_v612.py`, `build_best_mcu.py`, `emit_best.py`
(外部リポジトリ `D:/GitHub/IchiPing` の `pc/training/dataset.py` と、削除済みの Sim テンプレートが必要なため、xlsx は完全には再現できない)。
