# Solist-AI 単体構成 (CNN 前段 + ELM ヘッド) 引継ぎと今後の計画（2026-09-25）

前段の引継ぎ `HANDOFF_ELM_GENERALIZATION_20260925.md` の後に行った検証の結論と、今後の計画をまとめる。
数値はすべて 32 クラス frame 精度（明記がない限り）。

## 1. 結論

- **精度低下の主因は Solist の推論器ではなく特徴抽出だった。** 学習済みの小型 CNN 前段（凍結）を
  ML63Q2557 の CPU で動かし、その埋め込みを Solist ELM ヘッドに入れる構成にすると、PC CNN とほぼ同等になる。
- 評価の厳密化（日単位 LODO、内側 LODO でのハイパラ選択、run 単位 early stop、baseline frame の評価除外）の下で:

| データ / 評価 | 現行 ELM m32 / D167 | 新構成 校正なし | 新構成 校正あり (6 秒・5 窓) |
|---|---:|---:|---:|
| Original 3 日 LODO（小型前段 T8-16-32 FC32 + ELM m32） | 66.3% | 89.7% | 93.2% |
| FRDM 7 日 LODO（B1 Conv16-32-64-64 FC64 + ELM m64, FRDM のみ学習） | — | 93.2% | **95.9%** |
| UNO Q eval 4 セット（同上, UNO Q のみ学習） | — | 80.8% | 81.2% |

- 14 クラスはほぼ解決（FRDM 14cls 換算 100%、Original LODO でも 99.7–100%）。32 クラスの誤りの 90–96% は 14cls 等価内。
- **未解決の最大課題は温度変化への強さ。** UNO Q は同日内で約 11°C（周波数シフト 1.8%）の差があり、
  校正時と評価時で約 1% ずれると校正が逆効果（−5〜7 pt）になる。
- FRDM と UNO Q（どちらも INMP441）を混ぜて学習しても改善しない。差の原因（収録系・励振・設置・季節）は未特定。

## 2. 確定した構成

| 処理 | 担当 |
|---|---|
| INMP441 収録・MAX98357A PRBS 再生（同一 BCLK/WS）、48→16 kHz mono、PCM16 を I2C 転送 | Stamp-S3A（音響 I/O のみ） |
| FFT（2048 点, HW）→ Welch パワー積算 → log-PSD − 起動時 baseline log-PSD → frame 正規化 → 400–3000 Hz 333 bin | ML63Q2557 |
| 畳み込み前段（int8 想定, CPU）→ 埋め込み 32/64 次元 | ML63Q2557 CPU (Cortex-M0+) |
| ELM ヘッド推論、現地校正（β 再計算） | ML63Q2557 AxlCORE |

- 特徴は D167（時間波形 baseline 差分, baseline 64KB, サンプル同期必須）から
  **N333 noise_diff_norm**（log-PSD 差分, baseline 2KB, サンプル同期不要）へ切り替える。
- 推論速度は考慮しない（ユーザー方針: 精度のみ追求）。制約は Flash 256KB / SRAM 16KB。
  候補前段: B1/B2 ≈170KB（int8）、活性化ピーク ≈5KB。GAP（全体平均プーリング）は周波数位置を失うため不可。
- **現地校正は運用で実施する（ユーザー確定）。** 手順: 各状態で 6 秒再生し、2 秒窓・1 秒ずらしで 5 サンプル
  （独立 10 frame と同等の効果を確認済み）。校正は ELM β のみ再計算し、前段・正規化は工場値のまま。
- `IMPLEMENTATION_PLAN.md` の `PCM16` / `FEATURE167_F32` 経路の記述は D167 前提のため、N333 + 前段 CNN に合わせて更新が必要
  （代替経路は Stamp-S3A で log-PSD 333 点を計算して送る形）。

## 3. 評価ルール（今後も維持）

1. 評価日（UNO Q は評価セット）は学習・標準化・ハイパラ選択・early stop に使わない。
2. ハイパラは学習側の日単位（内側）LODO で選ぶ。frame 分割の validation は正則化を弱く選び楽観的になる
   （現行 ELM が 57.7%→66% に変わった主因）。
3. early stopping は学習側 run の hold-out で行う。
4. baseline に使った frame は評価サンプルから除外する。
5. 校正は「評価日の別 run（UNO Q は別セッション）」で行い、校正に使った run では評価しない。
6. グループ分けは日付ではなく run / 温度（ε）単位で考える（同日でも 11°C 違う）。

## 4. 今後の計画

### Phase 1: 温度耐性と校正の改善（既存データ・PC のみ）

| # | 内容 | 判定方法 |
|---|---|---|
| 1-A | 校正の混合重みを下げる。重みは UNO Q 学習セッション間の校正ペアだけで選ぶ | UNO Q eval の校正あり精度が校正なしを下回らないこと |
| 1-B | 推論時の温度補正: 起動時 baseline（s00000）から工場基準との周波数シフト ε を推定し、特徴を逆シフトしてから判定 | UNO Q eval（evening/survey）で改善するか |
| 1-C | 学習時シフト aug の幅（±2% → 実測範囲×2 など）を比較 | 同上 |
| 1-D | seed を増やし、fold 間のばらつきと差の有意性を確認 | — |

### Phase 2: 実機成立性の確認（PC + 公式 Sim）

| # | 内容 |
|---|---|
| 2-A | 前段の int8 化（PTQ、必要なら QAT）で精度低下を測る。M0+ は FPU 無しのため必須 |
| 2-B | SRAM 16KB の全体予算: FFT バッファ、PSD 積算（1024 点）、baseline（2KB）、前段活性化（≈5KB）、AI RAM。AxlCORE の FFT が AI RAM を使うかを確認 |
| 2-C | 公式 Sim（SLV1.00.04, GUI）で ELM ヘッドを検証: Sim の α（入力 = 埋め込み次元）を取得し、PC で β を計算 → model1.xlsx にロード → Test only。m64 の AI RAM 使用量も確認。Sim のオンデバイス学習はバグのため使わない |
| 2-D | 実機で `probe_alpha.py` により入力 32/64 次元の実 α を取得し、β を作り直す |

### Phase 3: 最終機データの収集（予定: 2 日 × 各 3 セッション程度）

- 温度条件をばらす（朝・夕、エアコン ON/OFF）。日数より温度幅が効く。
- 各セッションの**最初と最後に s00000（全閉）**を録る（run 内の温度ドリフト推定に必須）。
- 温度計の値をセッション毎（できれば時刻付き）に記録する。
- 校正手順（各状態 6 秒）も収録しておき、校正評価に使う。
- 学習は最終機データを主軸にし、既存データ（FRDM / UNO Q）は事前学習や比較に使う。

### Phase 4: 実機実装

- Stamp-S3A → Solist の PCM16 ストリーミング受信と Welch 積算（64KB の PCM を保持しない）。
- 前段 CNN（int8）と ELM ヘッドのファーム実装、校正モード（6 秒 × 32 状態）の実装。
- `board_test.py` 系で PC 計算とのビット一致・精度確認。

## 5. 未解決の論点

- FRDM と UNO Q のドメイン差の原因（模型・スピーカ・設置場所・部屋が同じだったか要確認）。
- ELM ヘッド m32 と m64 の選択（m64 は精度 +1〜3 pt だが AI RAM を Sim で要確認）。
- 運用中の自動適応（ラベル無しの逐次 β 更新）は未検証。ODL は教師ありのため、現状は校正時のラベル付きデータのみを使う。
- run 内の温度ドリフトは、全 run が同じ状態順で収録されているため基準 run の自己ドリフトと分離できていない（Phase 3 の s00000 再測定で解消）。

## 6. 用語

- **周波数シフト**: 温度変化による共鳴周波数の (1+ε) 倍の比例シフト（シフト率 ε を % で表記、約 0.17%/°C）。
  一定 Hz の加算シフトではない。旧称「ワープ」から全面改名（`sim/freq_shift.py`, `--shift` 等）。
- **D167**: 現行特徴（時間波形 baseline 差分 → 1024 点 FFT → 400–3000 Hz の 167 点）。
- **N333 / N1024**: noise_diff_norm（log-PSD 差分・frame 正規化）の 400–3000 Hz 333 点 / 全 1024 点。
- **LODO**: leave-one-day-out（1 日ずつ評価日にする）。

## 7. ファイル

| スクリプト | 結果 | 内容 |
|---|---|---|
| `sim/estimate_run_shift.py` | `RUN_SHIFT.{md,json}`, `RUN_SHIFT_drift.png` | s00000 から run 別周波数シフト（温度）推定 |
| `sim/eval_improve_lodo.py` | `IMPROVE_LODO.{md,json}` | 特徴（D167/N333/N1024）× 線形/ELM、校正あり/なし（Original 3 日） |
| `sim/eval_cnn_frontend_lodo.py` | `CNN_FRONTEND_LODO.{md,json}` | CNN XL 前段 + ELM ヘッド |
| `sim/eval_tiny_frontend_lodo.py` | `TINY_FRONTEND_LODO.{md,json}` | ML63Q2557 向け小型前段（Flash / 活性化 / MAC 併記） |
| `sim/eval_32cls_detail.py` | `DETAIL_32CLS.{md,json}` | 32 クラス詳細（fold 別・状態別・混同） |
| `sim/eval_full_data.py` | `FULL_DATA.{md,json}` | 全データ（FRDM 7 日 + UNO Q）、大型前段、win5 校正 |

結果は `sim_export/solist_ds/`。GPU 実行は `D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe`、
特徴キャッシュは `sim/_cache/`（gitignore）。ELM ヘッドの α はすべて一様乱数（実機 / Sim の α は未取得）。
