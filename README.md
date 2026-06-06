# IchiPing → Solist-AI 移植

能動音響（PRBS 励振音）による扉/窓の開閉状態センシング **IchiPing** を、ROHM のオンデバイス学習IC **Solist-AI（ML63Q2557）** へ移植するプロジェクト。移植元の深い 1D-CNN を、Solist-AI の浅い **ELM（α 乱数固定・β のみ学習）** に置き換える。

## 📊 検証レポート（GitHub Pages）

**https://airpocket-soundman.github.io/IchiPing_solist_AI/**

> 公開設定：Settings → Pages → Source = `main` ブランチ `/docs` フォルダ

モデル構造・特徴抽出/学習パイプライン・シミュレーション結果・実機ハードウェア構成を SVG 図つきで解説（`docs/index.html`）。

## 結果サマリ（公式 Solist-AI Sim SLV1.00.04 で実証）

| モデル | 運用 | フレーム精度 | 投票精度 |
|---|---|---|---|
| **14cls** | 校正不要（工場 β のみ） | **99.4%** | **100%** |
| **32cls** | 現地で各状態を約10フレーム校正 | **99.1%** | **100%** |
| 32cls（参考） | 校正前（出荷直後） | 77.8% | 81.2% |

いずれも **m=32 / D=167 / AI RAM ~11KB**＝実機（SRAM 16KB / Flash 256KB）に収まる。IchiPing CNN（実機 LIVE 59%）を両粒度で大きく上回る。

## 構成

```
docs/        GitHub Pages 公開レポート（自己完結 HTML + SVG）
sim/         検証パイプライン（純 numpy の Solist-AI 互換 ELM、特徴抽出、学習・評価スクリプト）
  build_best_mcu.py / emit_best.py   最良モデルの構築・ロード用モデル生成
  bench_v612.py / common.py / solist_elm.py / feats_diff.py   コア
sim_export/  成果物（ベスト版のみ）
  model1_BEST_14cls/            14cls 校正不要モデル（β=32×14）
  model1_BEST_32cls_ondevice/   32cls 現地校正モデル（β=32×32）
  model1_BEST_32cls_factory/    32cls 校正前モデル
  test_best_14cls.csv / test_BEST_32cls_*.csv   各検証用テスト
  _alpha32_sim.npy / _best_models.pkl           再現用（Sim α / best β 記録）
doc/         ROHM 仕様・アプリノート（著作権のため .gitignore＝ローカルのみ）
```

## モデル概要

- 入力：`FFT(audio − baseline)` を 400–3000Hz にクロップ（D=167）、z-score 標準化。
- α：(167×32) 一様乱数 U[−0.205,+0.205]（seed=1）。**学習せず**チップ上で `ODL_GenerateRandomNumber` により再生成（保存 0）。
- β：(32×C)。PC で最小二乗（batch-LS）学習し焼き込み。学習パラメータは IchiPing CNN 比 約 1/84。
- 演算 bfloat16 / 活性化 hard sigmoid / 入力スケール ×0.5（実効 scaleAlpha 0.1）。

## 実機での進め方

実機ではスピーカ・マイクが変更されるため **データは再取得**。本リポジトリのパイプライン（クロスベースライン×3・オーグメンテーション×3）をそのまま流用して β を再学習し、14cls は焼くだけ／32cls は現地校正で運用する。詳細はレポート §8 参照。

---
*検証は IchiPing 既収集データ（v6–v12）に基づく。学習は公式 Sim のオンデバイス学習が安定しなかったため PC 自作パイプライン（batch-LS）で実施。*
