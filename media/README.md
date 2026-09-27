# PV・作品ページの素材と制作条件

ROHM EDGE HACK CHALLENGE 2026 の応募に使った PV と ProtoPedia 作品ページの素材一式。

| 成果物 | 場所 |
|---|---|
| PV(完成版 v9, 2:58) | [pv/IchiPing_on_Solist-AI_PV_v9.mp4](pv/IchiPing_on_Solist-AI_PV_v9.mp4) / YouTube https://youtu.be/eih0WZhSuuw |
| YouTube サムネイル | [pv/thumbnail.jpg](pv/thumbnail.jpg)(元: [figures/fig_thumb.html](figures/fig_thumb.html)) |
| ProtoPedia 作品ページ | https://protopedia.net/prototype/8556(掲載画像は [../docs/protopedia/](../docs/protopedia/)) |

動画(`*.mp4`)と写真原本(`photos/*.jpg`)は Git LFS で管理している(`git lfs pull` で取得)。

## フォルダ構成

| パス | 内容 |
|---|---|
| `footage/` | 実機デモの撮影素材(スマートフォン, 1920×1080, 2026-09-27 撮影) |
| `photos/` | 実機の写真(4080×2296) |
| `figures/*.html` | 作品ページ・PV の図の原本(HTML + SVG)。`render_figures.py` で `docs/protopedia/` に PNG/JPEG 出力 |
| `figures/frames/` | 操作の流れの図とエンディングに使ったデモ動画のコマ(`VID_20260927_131829.mp4` の 3.5 / 10 / 37 / 42.8 秒) |
| `figures/external/` | 既存作品から流用した図(下表) |
| `pv/narration.json` | ナレーション台本(場面・話速つき) |
| `pv/narration/*.mp3` | 合成済みナレーション(PV v9 で使ったもの) |
| `pv/tts_narration.py` | ナレーション合成 |
| `pv/make_pv.py` | PV の組み立て(場面・字幕・音声の定義を含む) |

### 撮影素材

| ファイル | 長さ | 内容 | PV での使用 |
|---|---:|---|---|
| `VID_20260927_131743.mp4` | 14.7 s | サーボで窓・扉が開く様子(正面・近接) | コンセプト 0–7 s |
| `VID_20260927_131620.mp4` | 56.5 s | トグル操作→サーボ、11111 と 01010 の推論、評価ボード液晶 | しくみ 0–6 s、11111 の推論 19–30 s(2 倍速) |
| `VID_20260927_131829.mp4` | 43.4 s | 10100 / 00100 の操作〜推論〜Complete Success、TFT 接写 | 操作デモ 0.8–21.1 s |
| `VID_20260927_131508.mp4` | 62.9 s | 00000 / 01010 などの推論を複数回 | 未使用(初期版のみ) |
| `VID_20260927_131730.mp4` | 9.7 s | トグル操作の手元 | 未使用 |

### 流用した図(`figures/external/`)

| ファイル | 出典 |
|---|---|
| `manga.png`, `house.png` | DigiKey Make ONE Challenge 2026 応募時(オリジナル IchiPing, https://protopedia.net/prototype/8470)の素材 |
| `fftdiff_s00000_vs_s10000.png`, `fftdiff_all32_heatmap.jpg` | 同上の作品ページ掲載図(IchiPing リポジトリの FFT 差分解析) |

## PV の制作条件

- 映像:1920×1080, 30 fps, H.264(libx264, CRF 19, yuv420p, TV レンジ)。音声:AAC 192 kbps, 48 kHz, ステレオ。ffmpeg は `imageio-ffmpeg` 同梱の 7.1。
- 尺:応募条件の 3 分以内に収めるため 177.97 s。各場面の長さは「ナレーション + 前 0.15 s + 後 0.3 s」と最低表示時間の長い方。ズーム等の演出はしない(静止表示)。
- ナレーション:edge-tts 7.2.8、`ja-JP-NanamiNeural`、話速 +27%(冒頭のみ +34%)、音量 ×1.3。英字は読み間違いを防ぐため台本では仮名書き(例:32通り→さんじゅうにとおり、EXEC→エグゼック)。台本テキストは Microsoft のオンライン音声合成に送られる。
- 実機の音:トグルのクリック、サーボ、PRBS のチャープ音は作品の一部なので、ナレーション中も下げない。撮影素材の音は全体を ×5.6(+15 dB)し、操作デモのチャープ区間(7.4–10.0 s)だけさらに ×8(+18 dB)。操作デモはチャープの前後でナレーションを分け(n07a / n07b)、チャープ中は無言。
- 最後に `alimiter=limit=0.75` をかけてクリップを防止(平均 −18.5 dB, 最大 −1.5 dB)。
- 字幕:Noto Sans JP Bold 46 px(補足 30 px)。撮影素材の上は画面上部の角丸帯、図の上は画面下の帯(図は帯より上に収める)。
- 図:Chrome ヘッドレスで HTML を等倍キャプチャ(`render_figures.py`)。ProtoPedia は 1024 px 幅に縮小して掲載される。

### 場面構成(v9)

| 開始 | 長さ | 場面 | 素材 | ナレーション |
|---:|---:|---|---|---|
| 0:00.0 | 8.9 s | タイトル | `fig_hero` | n01 |
| 0:08.9 | 4.0 s | 課題 | `external/manga.png` | n02 |
| 0:12.9 | 7.3 s | コンセプト | `footage/…131743` | n03 |
| 0:20.2 | 11.6 s | House 模型・観測できない部屋 | `external/house.png` | n04 |
| 0:31.8 | 6.1 s | しくみ(トグル=正解ラベル) | `footage/…131620` | n05 |
| 0:38.0 | 23.0 s | 操作デモ ①〜④(④で静止) | `footage/…131829` | n07a, n07b |
| 1:01.0 | 5.6 s | 11111 の推論(2 倍速) | `footage/…131620` | n09 |
| 1:06.6 | 8.5 s | FFT 差分 | `external/fftdiff_s00000_vs_s10000.png` | n11 |
| 1:15.1 | 12.0 s | 役割分担 | `fig_system` | n13 |
| 1:27.1 | 12.5 s | AI パイプライン | `fig_pipeline`(上半分) | n15 |
| 1:39.6 | 15.7 s | パラメータ数の比較 | `fig_params` | n16 |
| 1:55.3 | 11.4 s | データのばらつきと 2 段構え | `fig_variation` | n18a |
| 2:06.7 | 10.2 s | データ採取の自動化 | `fig_autocollect` | n18 |
| 2:16.9 | 11.0 s | 検出精度 | `fig_accuracy` | n19 |
| 2:27.9 | 23.6 s | オンデバイス学習(ODL) | `fig_odl` | n17 |
| 2:51.5 | 6.5 s | エンディング | `frames/f829_42.8.jpg` | n20 |

### 表現上の決めごと

- データのばらつきは「日をまたぐ」「夜」ではなく「気温・背景音による収録セッションごとのばらつき」と説明する。
- モデル規模は「重み」ではなく「パラメータ数」で比較する(PC 上の理想的な CNN 約 30 万、オリジナル約 10.4 万、本作約 4.36 万)。
- 精度は「32 クラス分類の正解率」と書く。実機の結果は「閉じた扉の向こうも含む 32 クラスの推論に 100% 成功」。
- 校正時の動作は「扉と窓の開閉の組み合わせ全 32 クラスを巡回」。ODL は Solist-AI の目玉機能として扱う。
- 推論は 1 回の録音を 1 回推論(多数決はしていない)。CNN 前段は int8(層ごとの再量子化のみ float32)。

## 再生成

```powershell
python media/figures/render_figures.py     # 図 → docs/protopedia/
pip install edge-tts
python media/pv/tts_narration.py           # 台本を変えたときだけ
python media/pv/make_pv.py                 # → media/pv/build/IchiPing_on_Solist-AI_PV.mp4
```

`make_pv.py` はリポジトリ内の素材だけで v9 と同じ長さ・音量の動画を再現する(Windows のフォント `NotoSansJP-Bold.ttf` を使用)。
