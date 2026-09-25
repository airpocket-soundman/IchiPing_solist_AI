# 旧cross-baseline探索結果（運用汎化の採用判定には使用不可）

> **重要:** 学習特徴に別日／別セッションのbaseline差分が含まれる。実運用では起動時に当日のbaselineを取得するため、この条件は不一致であり、以下の数値を汎化性能として採用しない。`audio_day - baseline_same_session`で特徴を作り直し、audioとbaselineを同じ率でワープする再評価が必要。

評価日／評価セットはハイパーパラメータ選択にも使用していない。Original IchiPing は日単位の
leave-one-day-out、UNO Q は8学習セッションと4評価セットを分離した。値は frame accuracy /
macro F1 / 32状態別スコア合算精度。モデル選択も内側holdoutのmacro F1で行った。

## Original IchiPing: leave-one-day-out

| 学習条件 | holdout日 | 選択モデル | frame | macro F1 | state vote |
|---|---|---|---:|---:|---:|
| cross_baseline/linear | 2026-05-30 | linear-ridge-lam100 | 67.7% | 65.0% | 66.7% |
| cross_baseline/linear | 2026-05-31 | linear-ridge-lam100 | 70.0% | 66.6% | 71.9% |
| cross_baseline/linear | 2026-06-01 | linear-ridge-lam100 | 51.4% | 48.5% | 51.6% |
| **cross_baseline/linear 平均** | — | — | **63.0%** | **60.0%** | **63.4%** |
| cross_baseline/elm_m32 | 2026-05-30 | elm-m32-sim-s0.15 | 61.0% | 55.0% | 62.5% |
| cross_baseline/elm_m32 | 2026-05-31 | elm-m32-sim-s0.15 | 64.6% | 60.1% | 67.2% |
| cross_baseline/elm_m32 | 2026-06-01 | elm-m32-sim-s0.15 | 54.7% | 48.2% | 53.1% |
| **cross_baseline/elm_m32 平均** | — | — | **60.1%** | **54.4%** | **60.9%** |
| cross_baseline/elm_m64 | 2026-05-30 | elm-m64-random-s0.15 | 67.8% | 66.0% | 68.8% |
| cross_baseline/elm_m64 | 2026-05-31 | elm-m64-random-s0.15 | 63.2% | 57.7% | 62.5% |
| cross_baseline/elm_m64 | 2026-06-01 | elm-m64-random-s0.15 | 51.1% | 43.7% | 53.1% |
| **cross_baseline/elm_m64 平均** | — | — | **60.7%** | **55.8%** | **61.5%** |
| cross_baseline/elm_m128 | 2026-05-30 | elm-m128-random-s0.15 | 63.6% | 59.9% | 64.6% |
| cross_baseline/elm_m128 | 2026-05-31 | elm-m128-random-s0.15 | 70.1% | 68.0% | 73.4% |
| cross_baseline/elm_m128 | 2026-06-01 | elm-m128-random-s0.15 | 58.4% | 53.7% | 57.8% |
| **cross_baseline/elm_m128 平均** | — | — | **64.0%** | **60.5%** | **65.3%** |
| cross_baseline+freq_warp/linear | 2026-05-30 | linear-ridge-lam100 | 53.4% | 47.4% | 53.1% |
| cross_baseline+freq_warp/linear | 2026-05-31 | linear-ridge-lam100 | 62.7% | 58.0% | 64.1% |
| cross_baseline+freq_warp/linear | 2026-06-01 | linear-ridge-lam100 | 63.7% | 60.0% | 67.2% |
| **cross_baseline+freq_warp/linear 平均** | — | — | **60.0%** | **55.1%** | **61.5%** |
| cross_baseline+freq_warp/elm_m32 | 2026-05-30 | elm-m32-sim-s0.15 | 39.4% | 31.7% | 39.6% |
| cross_baseline+freq_warp/elm_m32 | 2026-05-31 | elm-m32-sim-s0.15 | 48.9% | 39.4% | 48.4% |
| cross_baseline+freq_warp/elm_m32 | 2026-06-01 | elm-m32-sim-s0.25 | 49.9% | 44.1% | 50.0% |
| **cross_baseline+freq_warp/elm_m32 平均** | — | — | **46.1%** | **38.4%** | **46.0%** |
| cross_baseline+freq_warp/elm_m64 | 2026-05-30 | elm-m64-random-s0.15 | 49.4% | 43.4% | 50.0% |
| cross_baseline+freq_warp/elm_m64 | 2026-05-31 | elm-m64-random-s0.15 | 53.6% | 46.5% | 56.2% |
| cross_baseline+freq_warp/elm_m64 | 2026-06-01 | elm-m64-random-s1 | 33.3% | 26.8% | 32.8% |
| **cross_baseline+freq_warp/elm_m64 平均** | — | — | **45.4%** | **38.9%** | **46.4%** |
| cross_baseline+freq_warp/elm_m128 | 2026-05-30 | elm-m128-random-s0.15 | 57.8% | 52.2% | 57.3% |
| cross_baseline+freq_warp/elm_m128 | 2026-05-31 | elm-m128-random-s0.15 | 58.0% | 52.7% | 59.4% |
| cross_baseline+freq_warp/elm_m128 | 2026-06-01 | elm-m128-random-s0.15 | 66.8% | 62.6% | 70.3% |
| **cross_baseline+freq_warp/elm_m128 平均** | — | — | **60.9%** | **55.8%** | **62.3%** |

## Arduino IchiPing / UNO Q: 未使用時刻・雑音セット

| 学習条件 | 評価セット | 選択モデル | frame | macro F1 | state vote |
|---|---|---|---:|---:|---:|
| unoq_none/linear | unoq_gray | linear-ridge-lam1 | 74.0% | 72.1% | 81.2% |
| unoq_none/linear | unoq_evening | linear-ridge-lam1 | 43.8% | 34.0% | 43.8% |
| unoq_none/linear | unoq_survey | linear-ridge-lam1 | 46.9% | 36.5% | 43.8% |
| unoq_none/linear | unoq_crowd | linear-ridge-lam1 | 58.4% | 52.5% | 65.6% |
| **unoq_none/linear 平均** | — | linear-ridge-lam1 | **55.8%** | **48.7%** | **58.6%** |
| unoq_none/elm_m32 | unoq_gray | elm-m32-sim-s0.5 | 47.9% | 41.0% | 50.0% |
| unoq_none/elm_m32 | unoq_evening | elm-m32-sim-s0.5 | 21.9% | 13.8% | 21.9% |
| unoq_none/elm_m32 | unoq_survey | elm-m32-sim-s0.5 | 30.3% | 21.1% | 31.2% |
| unoq_none/elm_m32 | unoq_crowd | elm-m32-sim-s0.5 | 39.4% | 31.2% | 43.8% |
| **unoq_none/elm_m32 平均** | — | elm-m32-sim-s0.5 | **34.9%** | **26.8%** | **36.7%** |
| unoq_none/elm_m64 | unoq_gray | elm-m64-random-s0.5 | 64.6% | 61.0% | 75.0% |
| unoq_none/elm_m64 | unoq_evening | elm-m64-random-s0.5 | 37.5% | 25.5% | 37.5% |
| unoq_none/elm_m64 | unoq_survey | elm-m64-random-s0.5 | 40.6% | 28.9% | 40.6% |
| unoq_none/elm_m64 | unoq_crowd | elm-m64-random-s0.5 | 37.2% | 34.5% | 40.6% |
| **unoq_none/elm_m64 平均** | — | elm-m64-random-s0.5 | **45.0%** | **37.5%** | **48.4%** |
| unoq_none/elm_m128 | unoq_gray | elm-m128-random-s0.25 | 67.7% | 64.7% | 81.2% |
| unoq_none/elm_m128 | unoq_evening | elm-m128-random-s0.25 | 31.2% | 20.3% | 31.2% |
| unoq_none/elm_m128 | unoq_survey | elm-m128-random-s0.25 | 44.7% | 35.0% | 43.8% |
| unoq_none/elm_m128 | unoq_crowd | elm-m128-random-s0.25 | 61.3% | 53.5% | 62.5% |
| **unoq_none/elm_m128 平均** | — | elm-m128-random-s0.25 | **51.2%** | **43.4%** | **54.7%** |
| unoq_ir2/linear | unoq_gray | linear-ridge-lam10 | 68.8% | 66.2% | 78.1% |
| unoq_ir2/linear | unoq_evening | linear-ridge-lam10 | 41.9% | 33.1% | 40.6% |
| unoq_ir2/linear | unoq_survey | linear-ridge-lam10 | 50.0% | 38.8% | 50.0% |
| unoq_ir2/linear | unoq_crowd | linear-ridge-lam10 | 65.9% | 59.4% | 71.9% |
| **unoq_ir2/linear 平均** | — | linear-ridge-lam10 | **56.6%** | **49.4%** | **60.2%** |
| unoq_ir2/elm_m32 | unoq_gray | elm-m32-sim-s0.15 | 39.6% | 34.2% | 46.9% |
| unoq_ir2/elm_m32 | unoq_evening | elm-m32-sim-s0.15 | 21.9% | 12.1% | 21.9% |
| unoq_ir2/elm_m32 | unoq_survey | elm-m32-sim-s0.15 | 23.8% | 13.7% | 25.0% |
| unoq_ir2/elm_m32 | unoq_crowd | elm-m32-sim-s0.15 | 29.7% | 20.4% | 31.2% |
| **unoq_ir2/elm_m32 平均** | — | elm-m32-sim-s0.15 | **28.7%** | **20.1%** | **31.2%** |
| unoq_ir2/elm_m64 | unoq_gray | elm-m64-random-s1 | 53.1% | 46.1% | 59.4% |
| unoq_ir2/elm_m64 | unoq_evening | elm-m64-random-s1 | 31.2% | 21.0% | 31.2% |
| unoq_ir2/elm_m64 | unoq_survey | elm-m64-random-s1 | 40.6% | 28.7% | 40.6% |
| unoq_ir2/elm_m64 | unoq_crowd | elm-m64-random-s1 | 43.4% | 35.3% | 43.8% |
| **unoq_ir2/elm_m64 平均** | — | elm-m64-random-s1 | **42.1%** | **32.8%** | **43.8%** |
| unoq_ir2/elm_m128 | unoq_gray | elm-m128-random-s0.25 | 64.6% | 60.8% | 78.1% |
| unoq_ir2/elm_m128 | unoq_evening | elm-m128-random-s0.25 | 37.5% | 27.3% | 37.5% |
| unoq_ir2/elm_m128 | unoq_survey | elm-m128-random-s0.25 | 35.6% | 24.1% | 37.5% |
| unoq_ir2/elm_m128 | unoq_crowd | elm-m128-random-s0.25 | 60.9% | 56.1% | 65.6% |
| **unoq_ir2/elm_m128 平均** | — | elm-m128-random-s0.25 | **49.7%** | **42.1%** | **54.7%** |
| unoq+frdm_ir2/linear | unoq_gray | linear-ridge-lam10 | 68.8% | 66.3% | 81.2% |
| unoq+frdm_ir2/linear | unoq_evening | linear-ridge-lam10 | 51.6% | 43.4% | 50.0% |
| unoq+frdm_ir2/linear | unoq_survey | linear-ridge-lam10 | 54.4% | 44.6% | 56.2% |
| unoq+frdm_ir2/linear | unoq_crowd | linear-ridge-lam10 | 63.1% | 56.7% | 62.5% |
| **unoq+frdm_ir2/linear 平均** | — | linear-ridge-lam10 | **59.5%** | **52.8%** | **62.5%** |
| unoq+frdm_ir2/elm_m32 | unoq_gray | elm-m32-sim-s0.5 | 44.8% | 41.3% | 53.1% |
| unoq+frdm_ir2/elm_m32 | unoq_evening | elm-m32-sim-s0.5 | 28.1% | 18.7% | 28.1% |
| unoq+frdm_ir2/elm_m32 | unoq_survey | elm-m32-sim-s0.5 | 43.4% | 31.2% | 43.8% |
| unoq+frdm_ir2/elm_m32 | unoq_crowd | elm-m32-sim-s0.5 | 34.1% | 25.6% | 31.2% |
| **unoq+frdm_ir2/elm_m32 平均** | — | elm-m32-sim-s0.5 | **37.6%** | **29.2%** | **39.1%** |
| unoq+frdm_ir2/elm_m64 | unoq_gray | elm-m64-random-s1 | 58.3% | 54.1% | 62.5% |
| unoq+frdm_ir2/elm_m64 | unoq_evening | elm-m64-random-s1 | 41.9% | 32.4% | 43.8% |
| unoq+frdm_ir2/elm_m64 | unoq_survey | elm-m64-random-s1 | 46.9% | 34.3% | 46.9% |
| unoq+frdm_ir2/elm_m64 | unoq_crowd | elm-m64-random-s1 | 40.0% | 33.4% | 40.6% |
| **unoq+frdm_ir2/elm_m64 平均** | — | elm-m64-random-s1 | **46.8%** | **38.6%** | **48.4%** |
| unoq+frdm_ir2/elm_m128 | unoq_gray | elm-m128-random-s0.25 | 68.8% | 66.0% | 78.1% |
| unoq+frdm_ir2/elm_m128 | unoq_evening | elm-m128-random-s0.25 | 50.0% | 42.0% | 50.0% |
| unoq+frdm_ir2/elm_m128 | unoq_survey | elm-m128-random-s0.25 | 56.6% | 47.7% | 56.2% |
| unoq+frdm_ir2/elm_m128 | unoq_crowd | elm-m128-random-s0.25 | 59.4% | 51.9% | 59.4% |
| **unoq+frdm_ir2/elm_m128 平均** | — | elm-m128-random-s0.25 | **58.7%** | **51.9%** | **60.9%** |

## 結論と当初想定との差

- 当初の99.1%はv12から各状態10フレームを校正用に取り、同じv12の残りを評価した現地校正値であり、日跨ぎ汎化値ではない。
- 当初のfactory 32cls約92%もv6-v11学習→v12評価で、v11とv12は同じ2026-06-01収録である。今回の日単位holdoutとは難易度が異なる。
- 厳密なOriginal日単位holdoutでは、実αのm=32 ELMは平均frame 60.1% / macro F1 54.4%。m=128でも64.0% / 60.5%で、95%級の汎化は確認できない。
- OriginalをUNO Qへ加えると、未使用4条件のframe / macro F1平均はm=32で34.9% / 26.8%→37.6% / 29.2%、m=128で51.2% / 43.4%→58.7% / 51.9%。複数日データは有効だが、単純混合だけでは不十分。
- 一律±3% warpはOriginal m=32のframe / macro F1を60.1% / 54.4%→46.1% / 38.4%へ悪化させ、UNO QのIR warpも34.9% / 26.8%→28.7% / 20.1%。温度シフト対策自体ではなく、現行D=167特徴への適用方法と分布設定が合っていない。
- βだけを学習するELMでは固定ランダムαが捨てた識別情報を復元できない。パラメータが小さいことは実装上の長所だが、十分な汎化性能の根拠にはならない。

## 次の改善実験（優先順）

1. UNO Qで効果が確認済みの1024-bin `noise_diff_norm` と、実測シフト分布に基づく周波数伸縮を移植し、最後に167次元へ圧縮する。
2. ランダムαを固定したままβだけを学ぶ条件と、教師ありで学習した167→32/64射影（またはteacherからの蒸留）を同じ日holdoutで比較する。
3. 日・時刻・baseline・機器をgroupとして分離し、augmentation強度とELM scale/ridgeをnested CVで選ぶ。評価日は最後まで触らない。
4. 最終Stamp-S3A収録で最低3日、各日複数時刻・baseline×3を取得し、未知日macro F1を採用判定にする。少量の現地校正あり／なしを別指標で管理する。

## 解釈上の注意

- Originalのfrequency warpはPRBS seedを再現できない世代のため、時間波形ではなく512-bin差分スペクトルを±3%ワープした。
- UNO Qの`unoq_ir2`は既知PRBSからIRを推定して時間伸縮後に再合成する、より物理的なaugmentationである。
- `unoq+frdm_ir2`はUNO Q学習データへOriginal由来FRDMデータを加え、同じUNO Q未使用4セットで評価した条件である。
- m=64/128のαは実機未プローブのため一様乱数による可能性評価。m=32だけが公式Simから採取した実αである。
- UNO Qは複数セッション・複数時刻だが同一日。厳密な日跨ぎ評価はOriginal側だけである。
