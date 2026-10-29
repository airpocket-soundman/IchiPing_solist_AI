# 学習環境(デバッグ PC)と学習データ

展示会(Edge Tech+ 2026)の現地再学習に備えて、デバッグ PC に整えた学習環境のメモ。

## Python 環境

リポジトリ直下の `.venv`(git 管理外)。Python 3.13、numpy、matplotlib、PyTorch(CPU 版)。この PC に GPU は無い。

```powershell
.\.venv\Scripts\python.exe sim/emit_frontend_model.py --help
```

## 学習データ `stamp_dataset_20260927.zip`

Stamp-S3A で収録した 13 セッション。[sim_export/solist_ds/README.md](../sim_export/solist_ds/README.md) の再生成手順が指定するセッションと一致する。

| 項目 | 内容 |
|---|---|
| セッション | `stamp_20260926_s1`〜`s11`、`stamp_20260927_s1`、`s2`(各 `_wav`、計 13) |
| 規模 | 各セッション 330 wav(32 状態 × 10 フレーム + 全閉の基準 10 フレーム)、約 21 MB。展開後 約 276 MB |
| 状態ディレクトリ | `s00000`〜`s11111`(窓 3 + 扉 2 の 5 bit、32 通り) |
| 音声 | 16 kHz / 16 bit / モノラル / 32000 サンプル(2 秒) |
| `session.json` | 開始時刻、`frames: 10`、`baseline: 10`、状態の収録順 `order` |

展開先は `captures/`(`captures/stamp_*_wav`)。容量が大きいため git 管理外(`.gitignore`)。

## 旧 IchiPing リポジトリへの依存

`sim/` は特徴量・ラベル定義(`dataset.py`)と収録データを、このリポジトリの隣のフォルダから読む(`../IchiPing`、`../IchiPing-UNO-Q`)。
どちらも `Documents/GitHub` 配下にクローン済み(`airpocket-soundman/IchiPing`、`airpocket-soundman/IchiPing-UNO-Q`)。パスは固定の `D:/` をやめ、スクリプト位置からの相対にした。

UNO Q 版の学習セッション(`uno_q_train_20260912_session1〜6`、`session7_loud`、`session8_loud`)は `unoq_dataset_20260912.zip` から
`IchiPing-UNO-Q/pc/captures/` に展開済み(各 1,650 wav、計 8 セッション、git 管理外)。

### 完成版モデルの再学習に必要なUNO Qデータ

`unoq_dataset_20260912.zip` には、上記の**学習用8セッション**と `full_32_eval_v1` が含まれる。学習用8セッションは
すでに `IchiPing-UNO-Q/pc/captures/` に展開済みだが、現行の `sim/emit_frontend_model.py` を完成版と同じ条件で実行するには、
次の評価用4セッションが別途必要である。

| 必須ディレクトリ | 用途 |
|---|---|
| `uno_q_eval_20260912_gray_wav` | 学習に使わない評価（gray） |
| `uno_q_eval_20260912_evening_wav` | 学習に使わない評価（evening）。生成ヘッダに埋め込む32クラスの参照入力にも使用 |
| `uno_q_eval_20260912_survey_wav` | 学習に使わない評価（survey） |
| `uno_q_eval_20260912_crowd_wav` | 学習に使わない評価（crowd） |

展開先はいずれも `IchiPing-UNO-Q/pc/captures/` 直下とする。これらがない場合、
`emit_frontend_model.py` は `FileNotFoundError` で停止する。`full_32_eval_v1` は現行スクリプトの4評価セットの代替としては
未対応であり、使うには評価処理と参照入力の選択を明示的に変更する必要がある。

## CPU 学習のテスト(2026-10-28)

実データでの学習は旧リポジトリと UNO Q 版の収録が無いため実行できなかった(`emit_frontend_model.py` は既定で `D:\GitHub\IchiPing-UNO-Q\...` を読み、`FileNotFoundError` になる)。
代わりに、前段 `small`(Conv 8/16/32 → FC 32)の学習ループ(`eval_full_data.train_net`)だけを、ダミーデータ(7,000 サンプル × 1,024 bin、32 クラス)で CPU(4 スレッド)で動かした。

| 項目 | 結果 |
|---|---|
| デバイス | CPU(`torch.cuda` なしで自動的に CPU に落ちる) |
| 1 エポック | 約 1.0 秒 |
| 上限(200 エポック × 3 seed) | 約 10 分(早期終了があるので、通常はこれより短い) |

実データでの所要時間は未計測。ダミーデータはサンプル数の目安で、ELM フィットや量子化の時間は含まない。

## 実データでの CPU 学習テスト(2026-10-28)

README の再生成手順 1(`--arch small`、Stamp 13 セッション、`--shift 0.005 --xbase 2`、3 seed)を、このPC の CPU(4 スレッド)で実行した。

| 項目 | 結果 |
|---|---|
| 所要時間 | 約 21 分(特徴量のキャッシュ作成を除く。初回は別に約 5 分) |
| 検証精度(`stamp_20260927_s2_wav`) | 3 seed とも 1.000 |
| 入力スケール `s_in` | 0.0326019(完成版と一致) |
| 重み | 完成版の npz とは一致しない(GPU と CPU、乱数の違い) |

**評価値は無効**: 評価 4 セットが無いため、学習に使った session4 で代用して流しただけで、評価の数字(全て 1.000)に意味は無い。
パイプラインが CPU で最後まで動くことと所要時間の確認だけが目的。生成物(`generated/*.h`、`BOARD_FRONTEND_PC.json`、npz)は git で元に戻した。

## ビルド環境の確認(2026-10-29)

| 項目 | 結果 |
|---|---|
| Solist-AI ファーム(`ichi_infer_main`、`ichi_survey_main`、`ichi_collect_main`) | ビルド成功(下記の修正が前提) |
| 書き込み(`flash.ps1`) | 入力の検査まで確認(`-Execute` なしのドライラン。ボードには書いていない) |
| PC 側ツール(`ichi_serial`、`infer_monitor`、`survey_monitor`、`collect_session`) | `.venv` に pyserial を入れて import 確認 |
| Stamp-S3A ファーム(`firmware/StampMeasure`、PlatformIO) | **ビルド失敗**: `driver/i2s_std.h` が無い |

ビルド手順:

```powershell
powershell -ExecutionPolicy Bypass -File firmware\IchiPingInference\tools\build.ps1 -Main ichi_infer_main -SourceProject "$env:USERPROFILE\lexide\workspace_omega_v2\AIVibrationInference"
```

- このPC のベンダープロジェクトは `AIVibrationInference`(`build.ps1` 既定の `AcrylicPanCollector_lowlatency` ではない)。
- LEXIDE のビルドツールは `Ver.20260904`。`build.ps1` と `flash.ps1` が `Ver.20260317` 固定だったので、インストール済みの最新版を使うよう直した。
  ベンダープロジェクトの makefile と `.res` 内の版番号も、同じ版に置き換える。
- 出力サイズ: 推論 text 91,920 B(prebuilt の flash.bin は 91,968 B)。ベンダープロジェクトとツール版が違うので完全一致はしない。
- Stamp-S3A: `platformio.ini` の `platform = espressif32` が版固定されておらず、Arduino-ESP32 2.x(IDF 4.4)になって `i2s_std.h`(IDF 5 の新 I²S API)が見つからない。Arduino-ESP32 3.x を使う platform(pioarduino など)が要る。
