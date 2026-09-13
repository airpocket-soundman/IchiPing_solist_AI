# IchiPing → Solist-AI 移植

能動音響（PRBS 励振音）による扉/窓の開閉状態センシング **IchiPing** を、ROHM **Solist-AI（ML63Q2557）** へ移植するプロジェクト。移植元の深い1D-CNNを、PCで学習してSolist-AIで推論する浅い **ELM（α乱数固定・βのみ学習）** に置き換える。

## 📊 検証レポート（GitHub Pages）

**https://airpocket-soundman.github.io/IchiPing_solist_AI/**

> 公開設定：Settings → Pages → Source = `main` ブランチ `/docs` フォルダ

モデル構造・特徴抽出/学習パイプライン・シミュレーション結果・実機ハードウェア構成を SVG 図つきで解説（`docs/index.html`）。

## 結果サマリ（公式 Solist-AI Sim SLV1.00.04 で実証）

| モデル | 運用 | フレーム精度 | 多数決精度 |
|---|---|---|---|
| **14cls** | 追加学習不要（工場 β のみ） | **99.4%** | **100%** |
| **32cls** | 各状態を約10フレーム追加しPC再学習 | **99.1%** | **100%** |
| 32cls（参考） | 現地データ追加前 | 77.8% | 81.2% |

いずれも **m=32 / D=167 / AI RAM ~11KB**＝実機（SRAM 16KB / Flash 256KB）に収まる。移植元IchiPing CNNも現地校正で32cls最大100%を達成しており、本移植版は**学習をPCへ集約し、学習パラメータ約1/51の小型モデルを16KB MCUで推論**する構成とする。

## 構成

```
docs/        GitHub Pages 公開レポート（自己完結 HTML + SVG）
  io_allocation.md             全公開端子監査と確定ピン割当
IMPLEMENTATION_PLAN.md  実機実装計画（Solist-AI 主制御 + Stamp-S3A I²S音響／GPIO）
hardware/stamp_s3a_interposer/  KiCad 10中間基板（回路接続表、配線済みPCB、BOM、検証スクリプト）
sim/         検証パイプライン（純 numpy の Solist-AI 互換 ELM、特徴抽出、学習・評価スクリプト）
  build_best_mcu.py / emit_best.py   最良モデルの構築・ロード用モデル生成
  bench_v612.py / common.py / solist_elm.py / feats_diff.py   コア
sim_export/  成果物（ベスト版のみ）
  model1_BEST_14cls/            14cls 追加学習不要モデル（β=32×14）
  model1_BEST_32cls_ondevice/   32cls 現地データ追加モデル（旧検証名を保持、β=32×32）
  model1_BEST_32cls_factory/    32cls 校正前モデル
  test_best_14cls.csv / test_BEST_32cls_*.csv   各検証用テスト
  _alpha32_sim.npy / _best_models.pkl           再現用（Sim α / best β 記録）
doc/         ROHM 仕様・アプリノート（著作権のため .gitignore＝ローカルのみ）
```

## モデル概要

- 入力：`FFT(audio − baseline)` を 400–3000Hz にクロップ（D=167）、z-score 標準化。
- α：(167×32) 一様乱数 U[−0.205,+0.205]（seed=1）。**学習せず**チップ上で `ODL_GenerateRandomNumber` により再生成（保存 0）。
- β：(32×C)。PC で最小二乗（batch-LS）学習し焼き込み。学習パラメータは IchiPing CNN 比 約 1/51。
- 演算 bfloat16 / 活性化 hard sigmoid / 入力スケール ×0.5（実効 scaleAlpha 0.1）。

## 実機での進め方

実機は **Solist-AI を主制御、M5Stack Stamp-S3A を I²S音響・GPIO補完サブコントローラ**とする。DT-EBML63Q2557 は I²S を持たないため、現行INMP441マイクの収録、現行MAX98357AアンプへのPRBS再生、再生・収録同期は Stamp-S3A が担当する。5状態入力とEXECもStamp-S3Aへ直結し、G44=窓a、G2=窓b、G4=窓c、G6=扉AB、G8=扉BC、G10=EXEC_N（active Low）とする。Solist-AI は ILI9341 TFT（SPI）、PCA9685／5サーボ（I²C）、固定済みELMによる推論、システム状態を管理する。Stampは状態／EXECをSolistへ通知する。**学習モードではStamp-S3AからPCへUSB CDCでPCM原本とラベルを直接送り、PCでβを再学習してSolist-AIへ書き込む。** 推論モードは整列済みPCMをI²Cで渡してSolistでFFTする経路を基準とし、転送時間／SRAMの実測結果によっては、Stamp-S3Aで同一仕様のD=167 raw dB特徴まで計算してI²C送信する経路へ切り替える。標準化とELMはどちらもSolistで行う。

Stamp-S3Aは中間基板へはんだ付けせず、左列1x17・1.27 mm雌ソケットと右列1x6・2.54 mm雌ソケットへ部品面を上にして搭載する。2.54 mm 1x9列と左列の偶数接点用部品を併設すると樹脂が干渉するため、左列は1.27 mmへ全面置換する。中間基板を上面から見たソケット列は、公式のStamp-S3A部品面PinMapに対して**左右鏡像**になる。USB／アンテナ方向、M1 pad番号、pin 1を基準に照合する。JLCPCB PCBA実装版を優先し、対象ソケットを実装できない場合は未実装基板へTHTソケットを手はんだする版も用意する。端子ごとの接続は [デバイス別接続対応表](docs/io_allocation.md#2-使用デバイス別の接続対応表) を参照。

マイクとアンプは現行品を維持するが、制御基板、I²S実装、配線、設置条件が変わるため **データは再取得**。`IchiPing-UNO-Q` で得た I²S format確認、安全な音量校正、pre-roll、相互相関によるPRBS開始位置合わせ、3本のbaseline、古い推論結果の拒否、段階的bring-upの知見をStamp-S3A実装へ流用する。UNO Q固有のDevice Tree／ALSA実装は移植しない。詳細は [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)、[docs/io_allocation.md](docs/io_allocation.md)、[KiCad中間基板](hardware/stamp_s3a_interposer/README.md)、レポート §7–8 を参照。

---
*検証は IchiPing 既収集データ（v6–v12）に基づく。学習は公式 Sim のオンデバイス学習が安定しなかったため PC 自作パイプライン（batch-LS）で実施。*
