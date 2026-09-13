# IchiPing Solist-AI 実機実装計画

## 1. 採用方針

実機の主制御とAI処理はDT-EBML63Q2557（Solist-AI）とする。**現行INMP441 I²Sマイクと現行MAX98357A I²Sアンプを維持することを必須条件**とし、I²Sを持たないDT-EBML63Q2557の代わりにM5Stack Stamp-S3Aが音響入出力を担当する。

1. INMP441マイクとMAX98357AアンプはStamp-S3Aへ直結し、同一BCLK/WSでPRBS再生と収録を同期する。
2. Stamp-S3Aは48 kHz I²Sを16 kHz monoへ変換し、PRBS onsetを整列する。
3. 推論モードはSolist-AIでbaseline差分・FFTを行う`PCM16`経路を基準とし、実測で転送時間／SRAM余裕が不足する場合だけStamp-S3Aで同一の特徴抽出を行う`FEATURE167_F32`経路へ切り替える。どちらも固定済みELMの推論はSolist-AIで行う。
4. TFTはSolist-AIのSPI、PCA9685サーボはSolist-AIのI²Cへ直結する。
5. 5状態入力とEXEC_NはStamp-S3Aへ直結する。左列は全17極を1.27 mm着脱ソケット、右列は6極を2.54 mmソケットとする。
6. 音声I²C転送、LCD、PCA9685はphaseで時分割し、PCA OEだけは独立GPIOで常時安全停止できるようにする。
7. 学習モードではStamp-S3Aが整列済みPCMとラベルをnative USB CDCでPCへ直接送り、βの学習・評価・生成はPCだけで行う。

Solist-AIのCN6アナログ入力＋SPI DAC案は技術的には可能だが、マイクとアンプの交換を伴うため採用しない。

## 2. 役割分担

| 機能 | 担当 | 理由 |
|---|---|---|
| INMP441 I²Sマイク収録 | Stamp-S3A | 現行デバイス維持、Solist-AIにI²Sなし |
| PRBS生成、MAX98357A出力、収録同期 | Stamp-S3A | TX/RXでBCLK/WSを共有 |
| AMP shutdown | Stamp-S3A G1 + 10 kΩ pull-down | reset中も既定OFF、左1.27 mm列 |
| 48→16 kHz、mono、onset整列 | Stamp-S3A | I²C転送を64 KB/測定へ削減 |
| 5状態入力 | Stamp-S3A G44/G2/G4/G6/G8 | input pull-up、Low=CLOSE。G2/G4/G6/G8のみ1.27 mm着脱ソケット |
| EXEC入力 | Stamp-S3A G10 | `EXEC_N`、input pull-up、active Low。StampからSolistへ通知 |
| FFT、baseline差分 | Solist-AI（基準）／Stamp-S3A（推論時の条件付き代替） | 同じversion付きD=167特徴仕様を使用 |
| 標準化、入力scale、ELM推論 | Solist-AI | モデル固有のμ／σ／scaleとPC学習済みβを主MCUに集約 |
| 学習データ収集 | Stamp-S3A → PC USB CDC | Solist／400 kHz I²Cを迂回して直接保存 |
| β学習・評価・生成 | PC | batch-LS、holdout評価後にSolistへ書込み |
| ILI9341 TFT | Solist SPI0 | 表示を主MCUで管理 |
| PCA9685 + SG90×5 | Solist I²C | サーボ安全を主MCUで管理 |
| PCA9685 OE | Solist P42 + 10 kΩ pull-up to 3.3 V | I²C固着中も全PWM停止 |
| SolistオンボードSW2/P50 | 補助／保守入力 | 通常EXECはStamp G10へ集約 |
| MCU間通信 | 共用I²C | 同時大容量通信がなく、追加信号を最小化 |

詳細な全ピン監査と配線は [docs/io_allocation.md](docs/io_allocation.md) を正とする。

## 3. ピン割当

### Solist-AI

- I²C SCL/SDA：CN3-1 P73 / CN3-3 P74
- TFT：CN3-12 P40=SCK、CN3-9 P41=MOSI、CN3-14 P43=CS、CN3-7 P22=D/C、CN3-6 P23=RST
- PCA9685 OE：CN3-10 P42、外付け10 kΩ pull-up
- SW2/P50：補助／保守入力（通常EXECには使用しない）
- TFT BL：固定点灯または外付け電源回路

### Stamp-S3A

- I²S：右2.54 mm列G43=BCLK、左1.27 mm列G5=WS／G7=DOUT→amp／G9=DIN←mic
- I²C target：左1.27 mm列G13=SDA、G15=SCL、7-bit `0x42`
- AMP SD：左1.27 mm列G1、外付け10 kΩ pull-down
- 5状態：G44=窓a（2.54 mm）、G2=窓b、G4=窓c、G6=扉AB、G8=扉BC（後4本は1.27 mm）、input pull-up、Low=CLOSE
- EXEC：G10=`EXEC_N`（1.27 mm）、input pull-up、active Low。デバウンス後にI²C status/eventでSolistへ通知
- PC学習データ：内蔵USB D−/D+（ESP32-S3 G19/G20、Stamp基板内配線）、外部GPIO追加なし
- 使用禁止：G0/G3/G46（boot strap）
- 温存：G39–G42（1.27 mm、pad-JTAG候補）。G43/G44のUART0はI²S BCLK／窓aへ転用し、書込み／保守logはnative USBを使う

通常版Stamp-S3Aは、左列M1-1..17へ1x17・1.27 mmピンヘッダ、右列M1-18/20/22/24/26/28へ1x6・2.54 mmピンヘッダを取り付け、中間基板の対応する雌ソケットへ着脱式で搭載する。左側の2.54 mm 1x9と偶数接点用1.27 mm部品は樹脂が干渉するため併設しない。Stampは部品面を上向きにするため、中間基板上面から見た左右のソケット列は公式部品面PinMapの**左右鏡像**になる。USB／アンテナ方向、M1 pad番号、pin 1をシルクと導通検査で照合する。

## 4. Stamp-S3A音響シーケンス

1. I²S TX/RXを48 kHz、stereo、32-bit slot、64 BCLK/frameで同時確保する。
2. BCLK=G43、WS=G5をmic/ampへ共通配線し、Stamp-S3Aだけをmasterにする。G43の起動時UART TX pulseはAMP SD=G1のpull-downで無音化する。
3. AMP SD=Low、zero PCMでDMAとclockを安定させる。
4. pre-roll後にAMP SD=High、決定論的PRBSを再生しながら同時収録する。
5. 48 kHz rawは保持せず、DMA ringから16 kHz mono PCM16へ逐次変換する。
6. 相互相関でPRBS onsetを検出し、整列済み2秒=64,000 byteを固定する。
7. 正常／異常終了とも、最初にAMP SD=Low、その後I²S/DMAを停止する。

INMP441はL/R=GNDでleft slot、MAX98357AはSD_MODEを直接Highにしてleft wordを使う。有効24 bitの位置、符号、slot、実効gainはUNO-Q知見を初期値にしつつ、Stamp実機のlogic analyzerと既知toneで再確認する。端子別配線と安全条件は [docs/io_allocation.md](docs/io_allocation.md#2-使用デバイス別の接続対応表) を正とする。

## 5. I²Cプロトコル

Solist-AIを唯一のcontrollerとし、LCD=`0x3E`、PCA9685=`0x40`、Stamp=`0x42`とする。PCA9685 ALLCALL `0x70`は初期化後に無効化する。

フレームは `version | type | length | sequence | offset | payload | CRC` とする。

- Solist→Stamp：`START_CAPTURE`（出力形式、schema、baseline IDを含む）、`GET_STATUS`、`READ_PAYLOAD`、`ACK`、`ABORT`
- Stamp→Solist：`BUSY/PAYLOAD_READY`、5状態＋`EXEC_N`の`INPUT_EVENT/STATUS`、`PCM_CHUNK`または`FEATURE_CHUNK`、`HEALTH`、`FAULT`

Stampは収録中に送信せず、収録完了後に選択されたpayloadをoffset付きreadへ返す。基準の`PCM16_LE`は32,000 sample／64,000 byte、条件付きの`FEATURE167_F32_LE`は167値／668 byteとする。後者も推論時専用で、学習時には生成せずPCM原本をUSBでPCへ送る。24–28 byte payloadからbring-upし、双方のbuffer上限確認後128–256 byteを目標にする。400 kHzで64 KB転送はACK込み理論約1.44秒以上だが、668 byteなら同じ概算で約15 ms以上まで短縮できる。chunk境界でバスを解放する。

PCA9685 PWMは最後の設定で自走する。通常更新は音声転送後とするが、緊急停止はI²Cを待たずP42をHighにしてOEを無効化する。これらのI²C bulk readは推論モードだけで使用する。

## 6. PC学習用USBプロトコル

Stampは `INFERENCE` と `TRAINING` を排他モードとし、boot時は必ず `INFERENCE` とする。modeとサーボの所有者はSolistだけとし、PCまたはUIからの収集開始要求を受けたSolistがI²Cで `SET_MODE(TRAINING)` と `START_CAPTURE(capture_id, committed_state)` をStampへ送る。学習モードでもI²Cのhealth／状態応答は維持するが、同じcaptureをSolistへPCM転送しない。PCからStampへサーボ／mode変更commandは設けない。

PC→Stampは `HELLO`、`CONFIG`、`ACK`、`NACK/RETRY`、`ABORT_TRANSFER`、Stamp→PCは `META`、複数の`DATA`、`END` frameとする。capture開始とmode変更はSolist→Stamp I²Cだけに限定する。1 captureの主成果物は整列済み2秒・16 kHz・mono・PCM16 little-endian 32,000 sample（64,000 byte）。`DATA`は4 KiBを初期値とし、sequence、offset、payload CRC32を付け、`END`に総byte数と全体CRC32を付ける。

metadataにはprotocol／schema version、session／capture ID、5-bit state maskと `sABCDE` label（bit0..4=窓a/b/c・扉AB/BC、1=OPEN）、収録直前／直後state、sample条件、PRBS seed／version／振幅、onset、scale／decimator版、RMS、peak、clip数、DMA error、firmware build IDを含める。PCは検証済みPCMを既存学習器互換の `captures/<session>/sABCDE/frame_NNNNNN.wav` とappend-only `records.jsonl`へ保存し、`.part`からatomic renameする。ACK受信まではStamp側bufferを再利用しない。

USBはESP32-S3のnative USBを用い、D−/D+に固定されたGPIO19/20を外部用途へ割り当てない。USB Serial/JTAGとUSB-OTGは同じ内蔵PHYを共有するため、初期実装は書込み／ログにも使えるUSB Serial/JTAG CDCを優先し、独自USB classは必要になった場合だけ検討する。学習収録時はUSB給電を基本とし、Stamp外部5 Vとの同時給電は逆流防止を実測できるまで禁止する。

## 7. Solist-AIのstreaming推論

- 64 KB PCMを16 KB SRAMへ一括保持しない。
- 小容量I²C ping-pong bufferへ受信し、対応するbaseline chunkをFlash/FeRAMから読む。
- time-domainで`current - baseline`を作り、1024 sample窓、hop 512でFFTする。
- 400–3000 Hz、D=167特徴を累積し、ELMへ渡す。
- baseline原本3本と全学習captureはPCへ保存し、Solistには実運用代表baselineとPC学習済みβだけを置く。
- linker mapでstack、I²C buffer、FFT ring/scratch、AI領域を合算し、余裕を実測する。

### 条件付きStamp FFT経路

`FEATURE167_F32`はI²C転送時間またはSolistのFFT用SRAMが実測基準を満たさない場合にだけ採用する。Stampはモデル非依存のraw dB特徴までを計算し、モデル固有処理はSolistに残す。

1. onset整列済み16 kHz mono PCM16 32,000 sampleを`sample / 32768.0`へ変換し、同じ`baseline_id`の時間波形をsample単位で減算する。
2. `NFFT=1024`、`hop=512`、対称Hann `0.5 - 0.5 cos(2πn/1023)`で61窓を処理する。
3. 各窓の`rFFT`複素絶対値を線形領域で平均し、DCを除外後、`max(20 log10(mag + 1e-9), -80 dB)`とする。
4. FFT bin 26..192（406.25..3000 Hz）を取り出し、float32 little-endian 167値を返す。per-frame正規化、power化、窓gain補正は行わない。
5. Solistがモデル同梱の各次元`mu`／`sd`（population std + 1e-6）と`input_scale`を適用し、bfloat16化してELMへ渡す。

`feature_schema_id = tdiff-rfft1024-h512-symhann-magmean-log20-floor80-bin26-192-f32-v1`、`baseline_id`／SHA-256、`normalization_id`、`model_id`、window数、PCM scale、FFT実装versionをpayload metadataへ持たせる。不一致は推論入力として使わず、明示的に失敗させる。既存のUNO-Q CNN用`noise_diff_norm`（2048点、power平均、スペクトルbaseline差分、per-frame正規化）とは互換でないため流用しない。形式はSolistがcapture開始前に明示指定し、Stampが自動切替しない。

採用ゲートは、同一PCM／baselineのPC golden vectorに対する167要素の数値比較、bfloat16化後の一致、回帰データでraw PCM経路とのtop-1 100%一致、holdout macro-F1の許容外劣化なし、Stamp heap high-water mark、I²C実効時間を測って決める。数値許容差とF1許容差は実機FFT誤差分布を見て固定する。合格するまでは`PCM16`を既定とし、capture途中で形式を混在させない。

## 8. 安全と状態整合性

各EXECにrequest IDと、その時点の5-bit requested stateを持たせる。収録中の状態変更、古いrequest、CRC error、timeoutの結果は表示しない。

- reset/未arm：Stamp AMP SD=Low、Solist PCA OE=High、TFTにunknown
- Stamp異常：ABORT、推論開始禁止、音声buffer破棄
- I²C固着：bus clear、失敗時PCA OE=High、状態unknown
- PCA異常：OE=High、fault latch、明示操作まで再arm禁止
- TRAINING中：Solistの推論要求をbusyで拒否し、1 captureの送信先はUSBだけに固定
- USB切断／CRC不一致：capture未完了としてPC側で破棄し、amp shutdown後に再試行
- 正常完了：同じrequest IDとstateが維持された場合だけ結果更新

## 9. IchiPing-UNO-Qから流用する知見

### 流用するもの

- 決定論的PRBS、pre-roll/post-roll、再生・収録同期
- I²S slot幅、有効bit位置、符号、L/Rの確認手順
- 相互相関によるonset整列
- 無音、clip、RMS、DMA欠落のhealth gate
- 低音量から最大2倍刻みで上げる校正
- 独立baseline×3、model/PRBS/音量/配線版のメタデータ
- request ID、busy、requested stateによる古い結果の拒否
- `bit0..4 = 窓a, 窓b, 窓c, 扉AB, 扉BC`、Low=CLOSE
- Gray code順の32状態収集、一度に動かすサーボは1台
- cross-baseline、実雑音、frequency-warp、日付分離評価

### 流用しないもの

- UNO Q固有の1.8 V、Device Tree、ALSA、Linux worker
- MPU/MCU mailboxをそのままI²Cへ置き換えること（I²C bulkは推論PCM／条件付き特徴だけ、学習データはUSB）

## 10. 実装フェーズ

### Phase 0：配線・安全

全ピン、電圧、pull、基板Revを照合し、AMP SDとPCA OEの既定停止を実測する。

### Phase 1：Stamp I²S音響

zero PCM、既知tone、PRBS、同時収録、48→16 kHz、onset整列を段階bring-upする。完了条件は64 KB整列frame生成、欠落なし、AMP確実停止。

### Phase 2：TFT・PCA・状態入力

Solist SPI TFT、I²C PCA9685、Stamp 5入力を単体確認する。全32状態とreset中の安全を検証する。

### Phase 3：I²C推論音声統合

chunk、CRC、sequence、offset、timeout、retry、bus clearを実装する。64 KB実効転送時間とPCA緊急OE停止を測る。

続いて推論時専用`FEATURE167_F32`を実装し、schema／baseline ID検証、668 byte転送、PC golden vector、`PCM16`経路との判定一致を確認する。採用判定まではbuild-timeまたはboot設定で`PCM16`を既定とする。

### Phase 4：USB学習データ収集

Stamp→PC USB CDCのbinary frame、CRC、capture ID、state label、atomic保存を実装する。32状態の収録を途中再開でき、重複／欠落captureを自動検出できることを完了条件とする。

### Phase 5：特徴処理・AI統合

streaming baseline差分、FFT、ELM、同一request ID表示まで接続し、PC golden vectorと照合する。

### Phase 6：実機データ再取得・PC再学習

全閉baseline×3と32状態を複数日・雑音条件でStampからPCへ直接収録する。PCでbatch-LS学習とholdout評価を行い、合格したβだけをSolistへ書き込む。14clsを先に焼込み検証し、その後32clsの現地データ追加再学習を評価する。

## 11. 未確定・実測事項

- 現物基板RevとCN3/CN1シルク
- mic/amp breakoutとILI9341 TFT現物のメーカー、SKU、revision、端子順（ICはINMP441/MAX98357A/ILI9341で確定）
- Stamp I²S full-duplexのDMA、onset jitter、heap high-water mark
- 400 kHz I²Cのchunk上限、実効時間、再送率、clock stretching
- 推論時`PCM16`と`FEATURE167_F32`のend-to-end遅延、消費電力、判定一致率
- USB CDCの実効帯域、長時間連続収録時の欠落率、Windows再接続時のport再識別
- LCD/PCA/Stamp pull-up合成値とbus容量
- SolistのFFT/baseline/AIを含むlinker map

## 12. JLCPCB PCBA部品方針

Rev.BはJLCPCB PCBAで調達可能な部品を第一候補とする。現時点の候補LCSC番号は、33 Ω=`C23140`、10 kΩ=`C25804`、100 kΩ=`C25803`、100 nF=`C14663`、JST XH 2P=`C265283`、3P=`C144394`、4P=`C144395`、5P=`C157991`、6P=`C144397`、左列1.27 mm雌ソケット=`C41360907`、Stamp側雄ヘッダ=`C41360852`、右列1x6・2.54 mm雌ソケット=`C42431861`である。**LCSC番号、実装区分、在庫、最小発注数、フットプリント寸法、実装面／向きは発注直前にJLCPCB Partsで再確認する。** 番号が一致しても代替部品を無確認で使わない。

JLCPCBでソケットを実装できない場合は、ソケットを未実装で発注し、左列には秋月電子1.27 mm雌ソケット`103866`、Stamp側雄ヘッダ`103865`を1x17へ切り分けて手はんだする。右列1x6・2.54 mmにも適合ソケットを手はんだするTHT版を用意する。外部JST XHもPCBA調達困難時には未実装とし、純正品を手はんだできるよう同じ穴・シルク・pin 1、ネットを維持する。
