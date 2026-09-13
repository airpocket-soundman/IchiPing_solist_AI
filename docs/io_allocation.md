# I/O・ピン割当台帳（現行I²Sデバイス維持）

対象は DT-EBML63Q2557 Rev.B（回路図 Rev.20260109）、通常版 M5Stack Stamp-S3A（S007-V033）、現行の **INMP441搭載I²Sマイク基板**、**MAX98357A搭載I²Sアンプ基板**、**ILI9341 2.4インチ240×320 SPI TFT** である。**マイクとアンプの交換は行わない。DT-EBML63Q2557はI²Sを持たないため、Stamp-S3AがI²S音響入出力と5状態入力を担当する。** Solist-AIはTFT、PCA9685、固定βでの推論、システム安全を担当し、学習はPCだけで行う。

元IchiPing/UNO Qの記録から音響IC型番は確定できたが、マイクとアンプの実装基板メーカー／SKUは記録されていない。アンプはAdafruit Product 3006相当として扱う。TFTのBOMはAdafruit Product 2478またはgeneric品を許容しており、現物SKUは未確定である。実装前に各現物の表裏写真、シルク、端子順、基板revisionを保存し、下表の「機能名」で照合する。generic基板は同じICでも端子順が異なるため、表の並び順をコネクタの物理順として使用しない。

公式資料とRev.B回路図ではメイン基板の14ピンデジタルI/Fは `CN3`、旧資料の一部では `CN1` と表記される。配線前に現物基板Revとシルクを照合する。

## 1. 採用する割当

### Solist-AI（DT-EBML63Q2557）

| 機能 | 物理端子 | MCU端子 | MCU pin | 接続／設定 | 起動・障害時 |
|---|---|---|---:|---|---|
| 共用I²C SCL | CN3-1 | P73/SCLF0-2/TMOUT0 | 62 | LCD、PCA9685、Stamp-S3A | open-drain、基板上pull-up |
| 共用I²C SDA | CN3-3 | P74/SDAF0-2/TMOUT1 | 61 | 同上 | SDA固着時bus clear |
| TFT RESET | CN3-6 | P23/TXDF2 | 18 | GPIO、ILI9341 RST | 基板上pull-upを利用 |
| TFT D/C | CN3-7 | P22/RXDF2 | 17 | GPIO、ILI9341 D/C | TFT CS=Highで無害化 |
| TFT MOSI | CN3-9 | P41/SOUTF0-2/TXDF3/PWM01 | 20 | SPI0 MOSI | CS=High |
| PCA9685 OE | CN3-10 | P42/SINF0-2/PWM10 | 21 | GPIO、active-high disable | 外付け10 kΩ pull-upで全PWM停止 |
| TFT SCK | CN3-12 | P40/SCKF0-2/RXDF3/PWM00 | 19 | SPI0 SCK | CS=High |
| TFT CS | CN3-14 | P43/SSNF0-2/PWM11 | 22 | SPI0 CS、active-low | 外付け10 kΩ pull-up推奨 |
| 補助SW | SW2/TP37 | P50/AIN6 | 42 | 保守用。通常EXECには使用しない | pull-up input |
| 状態LED | LED1..3/TP41..43 | P54/P55/P56 | 46/47/48 | BUSY/RESULT/FAULT | 消灯 |

TFTはwrite-onlyとし、MISO用だったP42をPCA9685 OEへ転用する。TFT BLはGPIOを使わず、使用モジュールの定格に合わせて固定点灯または外付け電源回路とする。PCA9685 OEをI²Cから独立させることで、バス固着中もサーボPWMを停止できる。

### Stamp-S3A

| 機能 | Stamp GPIO | 方向／設定 | 接続先 | 起動・障害時 |
|---|---:|---|---|---|
| I²S BCLK | G43 | output、3.072 MHz | mic SCK + amp BCLK | **2.54 mm**、起動時UART TX pulse中もampはG1で停止 |
| I²S WS/LRCLK | G5 | output、48 kHz | mic WS + amp LRC | **左1.27 mm列**、I²S停止 |
| I²S DOUT | G7 | output | amp DIN | **左1.27 mm列**、zero PCM |
| I²S DIN | G9 | input | mic SD | **左1.27 mm列**、入力のみ |
| I²C SDA/SCL | G13/G15 | target、open-drain | Solist CN3-3/1 | **左1.27 mm列**、address `0x42` |
| AMP SD/EN | G1 | output | I²S amp SD | **左1.27 mm列**、外付け10 kΩ pull-down、既定OFF |
| 窓a | G44 | input pull-up | switch→GND | **2.54 mm**、Low=CLOSE |
| 窓b/c | G2/G4 | input pull-up | switch→GND | **左1.27 mm列**、Low=CLOSE |
| 扉AB/BC | G6/G8 | input pull-up | switch→GND | **左1.27 mm列**、Low=CLOSE |
| EXEC | G10 | input pull-up | EXEC_N switch→GND | **1.27 mm**左列、active Low、30 ms debounce |
| PC学習data | internal G19/G20 | USB D−/D+ | PC native USB CDC | 外部GPIO追加なし、再割当禁止 |
| 条件付き予備 | G39–G42 | GPIO | pad-JTAG非使用時のみ | 今回はソケットなし |
| UART0 | G43/G44 | U0TXD/U0RXD | I²S BCLK／窓aへ転用 | 書込み・logはnative USBを使用 |
| 使用禁止 | G0/G3/G46 | boot strapping | 接続しない | G0はBOOTボタン、G46は既定pull-down |

I²SはStamp-S3Aをmasterとし、TX/RXでBCLKとWSを物理共有する。初期値は48 kHz、stereo、32-bit slot、64 BCLK/frame、BCLK=3.072 MHzとする。INMP441のL/RはGNDでleft slot、MAX98357AはSD_MODEをG1から直接Highにしてleft wordを選ぶ。PRBSはleft slotへ出力し、right slotも同値またはzeroとする。AMP SDはDMA、clock、zero PCMが安定した後だけ有効化する。G43はreset直後にUART0 TXとしてpulseを出し得るが、その間はG1の外付けpull-downでampをshutdownする。

通常版Stamp-S3Aは左列M1-1..17を1x17・1.27 mm、右列M1-18/20/22/24/26/28を1x6・2.54 mmの着脱コネクタで中間基板へ載せる。左側の2.54 mm 1x9と偶数接点用部品は樹脂が干渉するため併設せず、左列全体を1.27 mmへ置換する。中間基板上面から見た左右列は公式の部品面PinMapに対して**左右鏡像**であり、USB／アンテナ方向、M1 contact番号、pin 1を外観と導通の両方で照合する。

| 側面位置 | M1 contact | GPIO | 今回の用途／判断 |
|---|---|---|---|
| 右2.54 mm列 | 20 | G0 | BOOT button／boot strapのため不使用 |
| 左1.27 mm列 | 1 | G1 | AMP SD、10 kΩ pull-down |
| 左1.27 mm列 | 3 | G3 | JTAG source strapのため不使用 |
| 左1.27 mm列 | 5/7/9 | G5/G7/G9 | I²S WS／DOUT／DIN |
| 左1.27 mm列 | 15/17 | G13/G15 | I²C SDA／SCL |
| 右2.54 mm列 | 26/24 | G43/G44 | I²S BCLK／窓a。UART0は不使用 |
| 左1.27 mm列 | 2/4/6/8 | G2/G4/G6/G8 | 窓b／窓c／扉AB／扉BC。1x17ソケット内 |
| 左1.27 mm列 | 10/12/14/16 | G10/G11/G12/G14 | G10=EXEC_N、他は未使用予備 |
| 右側1.27 mm padのみ | 19/21/23/25 | G39/G40/G41/G42 | 今回はソケットなし。pad-JTAG候補として温存 |
| 右側1.27 mm padのみ | 27 | G46 | 今回はソケットなし。boot strapのため不使用 |

公式上2.54 mm位置でもあるM1-1/3/5/7/9/11/13/15/17は、本キャリアでは左1x17・1.27 mmコネクタの奇数接点として接続する。右列M1-18/20/22/24/26/28だけ2.54 mm 1x6を使用する。ここで対象にするのは通常版`Stamp-S3A`（S007-V033）であり、別製品`Stamp-S3A PIN2.54`ではない。

## 2. 使用デバイス別の接続対応表

### INMP441搭載I²Sマイク基板

| 機能 | 基板でよくある印字 | 接続先 | 電圧／方向 | 固定条件・注意 |
|---|---|---|---|---|
| 電源 | `VDD` / `VCC` / `3V3` | Stamp-S3A `3V3` | 3.3 V | **5 V禁止**。マイク直近に0.1 µF。相手無給電時はI²S clockを出さない |
| GND | `GND` | Stamp GND | ground | Solist、Stamp、外部5 VのGNDと共通。ただしservo/amp大電流returnと分けてstar接続 |
| bit clock | `SCK` / `BCLK` / `BCK` | Stamp-S3A `G43` | Stamp→mic、3.072 MHz | 2.54 mm位置、amp BCLKと共有。配線を短くし、必要なら源端に22–33 Ω |
| word select | `WS` / `LRCLK` / `LRC` | Stamp-S3A `G5` | Stamp→mic、48 kHz | 左1.27 mm列、amp LRCと共有、WS Low=left |
| 音声data | `SD` / `DOUT` / `DATA` | Stamp-S3A `G9` I²S DIN | mic→Stamp、3.3 V | 左1.27 mm列。非選択slotでHi-ZになるためG9側へ100 kΩ pull-down推奨 |
| slot選択 | `L/R` / `LR` / `SEL` | GND | static Low | left slot固定。浮かせない |
| chip enable | `CHIPEN`（露出時のみ） | Stamp `3V3` | static High | 一般的6-pin基板では基板内でVDD固定。現物回路を確認 |

INMP441は24-bit、two's complement、MSB-first、標準I²Sの1 BCLK遅延で受信する。48 kHz×64 BCLK/frame = 3.072 MHzは仕様上限3.2 MHzに近いため、logic analyzerで3.2 MHzを超えないことを確認する。電源投入後少なくとも218 SCK cycleの無効期間を捨てる。

### MAX98357A搭載mono I²Sアンプ基板

| 機能 | 基板でよくある印字 | 接続先 | 電圧／方向 | 固定条件・注意 |
|---|---|---|---|---|
| アンプ電源 | `VIN` / `VDD` / `5V` | ノイズ対策した外部5 V | 5 V | アンプ近傍に10 µF＋0.1 µF。サーボ5 Vとreturn／bulkを分ける |
| GND | `GND` | 共通GND | ground | Stamp I²Sの基準。大電流speaker returnをmic配線へ流さない |
| bit clock | `BCLK` / `BCK` | Stamp-S3A `G43` | Stamp→amp、3.3 V | 2.54 mm位置、mic SCKと共有。起動時pulse中はSD=Low。MCLKは不要 |
| word select | `LRC` / `LRCLK` / `WS` | Stamp-S3A `G5` | Stamp→amp、3.3 V | 左1.27 mm列、mic WSと共有 |
| 音声data入力 | `DIN` / `SDIN` / `DATA` | Stamp-S3A `G7` I²S DOUT | Stamp→amp、3.3 V | 左1.27 mm列、left slotへPRBSを出力 |
| shutdown／channel | `SD` / `SD_MODE` / `EN` | Stamp-S3A `G1` | Low=shutdown、High=left | 左1.27 mm列。GNDへ外付け10 kΩ。reset中はLow、clock＋zero PCM安定後だけHigh |
| gain | `GAIN` / `GAIN_SLOT` | GND | static Low | 現行UNO Qと同じ**12 dB**。旧資料の「3 dB」は誤り |
| speaker出力 | `SPK+` / `OUT+` | speaker + | BTL output | speakerの片側だけを接続しない |
| speaker出力 | `SPK-` / `OUT-` | speaker − | BTL output | **GND接続禁止**。接地オシロのGND clipも接続禁止 |

停止は `G1=Low` を先に行い、その後I²S clockを止める。開始はBCLK、LRCLK、zero PCMを先に安定させてからG1をHighにする。BCLKを残したままLRCLKだけ止めない。現行の8 Ω 0.25 W speakerを使うため、PRBS振幅はUNO Qで確認済みの低レベルから校正する。

### ILI9341 2.4インチ240×320 SPI TFT

第一候補はAdafruit Product 2478（touch／microSD付きbreakout）で、この基板はVinとlogic level変換を持つ。現物が別のILI9341 generic moduleなら、Vin、logic、LED/BLの定格が異なるため本表をそのまま適用しない。

| 機能 | Product 2478印字／別名 | Solist-AI接続 | 電圧／方向 | 固定条件・注意 |
|---|---|---|---|---|
| 電源 | `3-5V Vin` | 外部5 V（または確認済み3.3 V） | power | CN3-5/8から給電しない。`3.3Vout`は出力なので給電入力にしない |
| GND | `GND` | CN3-11または共通GND | ground | Solist logic基準 |
| SPI clock | `CLK` / `SCK` / `SCL` | CN3-12 `P40/SCKF0-2` | Solist→TFT、3.3 V | `SCL`印字でもI²Cではない |
| SPI data | `MOSI` / `SDI` / `SDA` | CN3-9 `P41/SOUTF0-2` | Solist→TFT、3.3 V | `SDA`印字でもI²Cではない |
| chip select | `CS` / `TFT_CS` | CN3-14 `P43/SSNF0-2` | active-low | 3.3 Vへ外付け10 kΩ pull-up推奨 |
| data／command | `D/C` / `DC` / `RS` / `A0` | CN3-7 `P22` | Solist→TFT | TFT CS=Highの間に初期化 |
| reset | `RST` / `RESET` / `RES` | CN3-6 `P23` | active-low | 評価基板pull-up値を実測。電源投入後Low pulse |
| controller data out | `MISO` / `SDO` | 未接続 | TFT→Solist | write-only運用。P42はPCA9685 OEに使用 |
| backlight | `Lite` / `LED` / `BL` | 未接続（基板既定Highで固定点灯） | module依存 | Product 2478では既定ON。generic基板は抵抗／FET有無を確認 |
| microSD | `CCS` / `Card CS` | 未接続 | active-low | 3.3 VへHigh固定しbus競合を防ぐ |
| touch | `X+` / `X-` / `Y+` / `Y-` | 未接続 | analog | 今回は使用しない |

Product 2478はSPI modeになるよう基板jumpers（IM1/IM2/IM3 closed、IM0 open）を現物確認する。240×320をlandscape表示し、初期SPI clockは低速から開始して描画確認後に上げる。

### 電源とboot時の必須状態

| 状態 | 必須結果 |
|---|---|
| Stamp-S3A reset／無給電 | G1の10 kΩ pull-downでamp shutdown。SCK/WS/DINを外部から印加しない |
| Stamp起動直後 | amp shutdownのままI²S clockとzero PCMを開始し、安定後にenable |
| Solist reset中 | TFT CS=High、PCA9685 OE=High。表示unknown、servo PWM禁止 |
| Solistだけ給電 | I²C pull-up経由でStampへback-powerしないことを実測 |
| Stamp USB＋外部5 V | 使用するStampキャリアの逆流防止回路を確認。未確認なら同時給電しない |
| servo／amp動作中 | 3.3 V rail、mic noise floor、ground bounceを測り、音響収録中はservoを停止 |

## 3. Solist-AI CN3全14ピン監査

| CN3 | 信号 | 可能なMCU機能 | 今回 | 注意 |
|---:|---|---|---|---|
| 1 | P73/SCL | GPIO、EXI/TMCKI、SCLF0-2、TMOUT0 | I²C SCL | LCD共有、pull-up実装済み |
| 2 | GND | ground | 共通GND |  |
| 3 | P74/SDA | GPIO、EXI/TMCKI、SDAF0-2、TMOUT1 | I²C SDA | LCD共有、pull-up実装済み |
| 4 | GND | ground | 共通GND |  |
| 5 | Power Out | JP1で3.3/5 V選択 | 未使用 | JP1出荷時は供給停止 |
| 6 | P23/INT2 | GPIO、EXI/TMCKI、TXDF2 | TFT RST | 基板pull-up値を実測 |
| 7 | P22/INT1 | GPIO、EXI/TMCKI、RXDF2 | TFT D/C | 基板pull-up値を実測 |
| 8 | Power Out | JP1で3.3/5 V選択 | 未使用 | pin 5と同じ電源系 |
| 9 | P41/MOSI | GPIO、SOUTF0-2、TXDF3、PWM01 | TFT MOSI | SPI0 |
| 10 | P42/MISO | GPIO、SINF0-2、PWM10 | PCA9685 OE | TFT readback禁止 |
| 11 | GND | ground | TFT/PCA GND |  |
| 12 | P40/SCK | GPIO、SCKF0-2、RXDF3、PWM00 | TFT SCK | SPI0 |
| 13 | GND | ground | Stamp GND |  |
| 14 | P43/CS | GPIO、SSNF0-2、PWM11 | TFT CS | active-low |

CN3信号入力は3.6 V絶対最大で、5 V logicを直結しない。CN3-5/8からStamp-S3A、SG90、I²Sアンプの電力は取らない。Stamp、サーボ、アンプは十分な容量の別電源を使い、GNDだけ共通化する。

## 4. Stamp-S3A公開23 GPIO・端子ピッチ監査

ESP32-S3のGPIO matrixにより公開digital GPIOはI²C/SPI/I²S/UARTへrouteできる。基板競合とboot/debug制約を優先した。

| GPIO | 代表的な固定／アナログ機能 | 制約 | 今回 |
|---:|---|---|---|
| G0 | RTC GPIO0 | strapping、BOOTボタン | 使用禁止 |
| G1/G2 | ADC1_CH0/1、RTC | キャリア左1.27 mm列 | AMP SD G1、窓b G2 |
| G3 | ADC1_CH2、RTC | strapping | 使用禁止 |
| G4–G10 | ADC1_CH3–9、RTC | 左1x17・1.27 mm列 | 状態G4/G6/G8、I²S G5/G7/G9、EXEC G10 |
| G11–G15 | ADC2_CH0–4、RTC | キャリア左1.27 mm列 | I²C G13/G15、他はNC |
| G39/G40/G41/G42 | MTCK/MTDO/MTDI/MTMS | pad-JTAG候補、1.27 mm | ソケットなし、条件付き予備 |
| G43/G44 | U0TXD/U0RXD | 2.54 mm、UART0競合 | I²S BCLK／窓a。保守はUSB |
| G46 | RTC GPIO46 | strapping、既定pull-down | 使用禁止 |

非公開のGPIO19/20はUSB D−/D+としてPC学習データ直送に使用する。GPIO26–32はFlash、GPIO33–38は基板裏LCD FPC等のため外部候補にしない。Stamp-S3AではRGB LED電源とLCD backlight電源がG38で共用され、RGB信号G21も公開23 GPIOには含まれない。

## 5. 推論I²CとPC学習USBのデータ経路

| target | 7-bit address | 備考 |
|---|---:|---|
| オンボードLCD | `0x3E` | 既存driverの8-bit `0x7C`から換算 |
| PCA9685 | `0x40` | A0–A5=Low。起動後ALLCALLを無効化 |
| PCA9685 ALLCALL | `0x70` | power-on既定で有効 |
| Stamp-S3A | `0x42` | 全modeの制御／状態／health、推論modeだけPCMまたはD=167特徴chunk |

Stampは収録中に送信せず、48 kHz I²S DMAから逐次16 kHz monoへ変換し、onset整列済み2秒PCM16（64 KB）を保持する。`INFERENCE` modeは次の2形式を排他的に選ぶ。`PCM16_LE`を基準とし、`FEATURE167_F32_LE`はPC golden vectorとの同値性を実証でき、かつ転送時間／Solist SRAMに実益がある場合だけ採用する。

| 推論payload | Stamp処理 | I²C payload | Solist処理 | 位置づけ |
|---|---|---:|---|---|
| `PCM16_LE` | 16 kHz mono、onset整列 | 64,000 byte | 時間baseline差分、FFT、crop、標準化、ELM | 基準／既定 |
| `FEATURE167_F32_LE` | 上記に加え時間baseline差分、FFT、raw dB crop | 668 byte | μ／σ、input scale、bfloat16化、ELM | 推論時だけの条件付き代替 |

特徴仕様は`NFFT=1024`、hop 512、対称Hann、61窓の複素絶対値を線形平均、DC除外、`max(20log10(mag+1e-9), -80 dB)`、FFT bin 26..192（406.25..3000 Hz）の167値である。Stampではper-frame正規化を行わず、モデル固有のμ／σ／input scaleはSolistへ残す。400 kHzで64 KBは理論約1.44秒以上、668 byteは同じ概算で約15 ms以上。24–28 byteからbring-upし、実装上限確認後128–256 byteを目標にし、chunk境界でバスを解放する。

`START_CAPTURE`で出力形式、`feature_schema_id`、`baseline_id`／SHA-256、`normalization_id`、`model_id`を指定し、応答にはStampがGPIOから確定した5-bit `committed_state`、同じID群、capture ID、window数、payload長、health flags、CRCを含める。不一致やcapture途中の形式変更は拒否する。出力形式はSolistだけがcapture開始前に選び、Stampは自動切替しない。Stamp特徴経路にはSolistと同じ時間波形baselineを配備する必要があり、モデル／PRBS／音量／配線versionと一体で管理する。

`TRAINING` modeでは同じ64 KB PCMをSolistへ迂回させず、Stampのnative USB CDCからPCへ直接送る。USB転送は `META`、4 KiB単位の`DATA`、`END`で構成し、capture ID、5-bit state maskと`sABCDE` label、sample条件、PRBS識別子／振幅、onset、scale、RMS／peak／clip、DMA health、firmware版、offset／sequence、chunk CRC32、全体CRC32を保持する。PCはCRCと32,000 sampleを確認してから `.part` を既存学習器互換の `captures/<session>/sABCDE/frame_NNNNNN.wav` へatomic renameし、詳細を`records.jsonl`へ追記する。

2 modeは排他でboot時は`INFERENCE`とする。Solistをmode／capture ID／サーボの唯一の所有者とし、PCまたはUIの収集要求を受けたSolistがI²Cで`SET_MODE(TRAINING)`と`START_CAPTURE(capture_id, committed_state)`を送る。StampのUSB commandにはサーボ操作やmode変更を持たせない。学習用bulk PCMのI²C readは拒否し、USBのPC ACKを受けるまでStampはbufferを再利用しない。

## 6. メモリと安全

- Stampは3秒の48 kHz stereo rawを保持せず、DMA ringから16 kHz monoへ逐次変換する。
- 中間基板はF.Cu/B.Cuの両面をGNDベタとし、GNDパッドは低インピーダンス優先で直結する。Stamp-S3Aアンテナ直下は両面とも銅箔を除外する。
- PCM推論時、Solistは64 KBを一括保持せず、I²C ping-pong buffer、baseline chunk、FFT窓だけを使う。Stamp特徴推論時は668 byteもchunk受信する。
- PC学習時はUSB CDCへ直接送信し、Solistは学習もPCM中継もしない。
- reset/未arm時はStamp AMP SD=Low、Solist PCA OE=High、TFT CS=High、表示unknown。
- I²C timeout/CRC error/状態変更では推論を破棄し、PCA OEをHigh、StampへABORTを送る。
- AMP SDは外付けpull-down、PCA OEは**PCA9685の3.3 V VCCへ**外付けpull-upとし、firmwareやI²Cに依存しない既定停止を作る。PCA9685のlogic `VCC=3.3 V` とservo `V+=外部5 V`を混同しない。

## 7. 不採用としたSolist-AI直接音響案

Solist-AI単独でも、`analog mic preamp→CN6/SA-ADC` と `SPI DAC MCP4821→analog amp` を追加すれば音響制御は技術的に可能である。しかしこれは現行I²SマイクとI²Sアンプを別デバイスへ交換する案であり、今回の前提に反するため採用しない。PWM+LPF案も同じ理由に加え、分解能、carrier漏れ、EMIのリスクがあるため採用しない。

## 8. 現物確認ゲート

1. 基板Rev、14ピン端子のCN3/CN1シルク、LED数を記録する。
2. Stamp-S3Aを部品面上向きに挿した状態で、キャリア側の左1x17・1.27 mm／右1x6・2.54 mmが公式部品面PinMapの左右鏡像であること、USB／アンテナ方向、M1 pad 1、G10 EXEC_Nを含む全使用接点の導通を確認し、G1 AMP SDのreset波形を測る。
3. I²C scanでLCD=`0x3E`、PCA=`0x40/0x70`、Stamp=`0x42`を確認する。
4. 48 kHz I²S TX/RXのBCLK/WS共有、slot、有効bit位置、欠落sampleを測る。
5. 100回のPRBSでonset jitter、RMS、clip、DMA over/underrunを測る。
6. 片側reset、I²C固着、電源投入順逆転でAMP SD=Low、PCA OE=Highを確認する。
7. Solistのlinker mapとStampのheap high-water markを記録する。
8. マイク、アンプ、TFT各基板のメーカー、SKU、revision、端子シルクを写真で記録する。
9. Stamp USB＋外部5 V、およびStamp無給電＋Solist給電で逆流／back-power電流を測る。
10. speaker未接続でamp SDのreset波形を確認してから、低振幅・短時間の発音試験へ進む。
11. USB CDCで64 KB×1,650 captureを連続取得し、capture ID、CRC、sample数、state labelの欠落／重複がないことを確認する。
12. 推論時だけStampでD=167特徴を生成し、固定PCM／baselineのPC golden vector、Solist PCM経路、最終bfloat16入力、判定結果を比較する。

## 9. PCBA／THT実装variant

JLCPCB PCBA版の候補LCSC番号は、33 Ω=`C23140`、10 kΩ=`C25804`、100 kΩ=`C25803`、100 nF=`C14663`、JST XH 2P=`C265283`、3P=`C144394`、4P=`C144395`、5P=`C157991`、6P=`C144397`、左列1.27 mm雌ソケット=`C41360907`、Stamp側雄ヘッダ=`C41360852`、右列1x6・2.54 mm雌ソケット=`C42431861`である。**LCSC番号、在庫、実装区分、最小発注数、実装可否、フットプリント、pin 1、CPL回転は発注直前にJLCPCBで再確認する。**

THT手はんだ版は、左列に秋月の1.27 mm雌ソケット`103866`／Stamp側雄ヘッダ`103865`を1x17へ切り分けて使用し、右列には適合する1x6・2.54 mmソケットを実装する。外部XHの信号順、Stampソケット穴位置、pin 1、firmwareのGPIO割当はPCBA版と共通とする。

## 10. 根拠資料

- [DT-EBML63Q2557公式ハードウェアマニュアル](https://www.datatecno.co.jp/datatecno_core/content/uploads/2025/06/DT-EBML63Q2557_hardware_users_manual_Rev.20250527.pdf)
- ローカル回路図 `D:/GitHub/acrylic_pan/solist-ai/DT-EBML63Q2557_circuit_diagram_Rev.20260109.pdf`
- [ROHM ML63Q2500 Group User's Manual](https://fscdn.rohm.com/lapis/en/products/databook/applinote/ic/micon/FEUL63Q2500.pdf)
- [M5Stack Stamp-S3A公式仕様](https://docs.m5stack.com/en/core/Stamp-S3A)
- [Stamp-S3A公式PinMap](https://m5stack-doc.oss-cn-shenzhen.aliyuncs.com/1150/S007-V033_PinMap_01.jpg)
- [Stamp-S3A公式回路図](https://m5stack-doc.oss-cn-shenzhen.aliyuncs.com/1150/Sch_StampS3_v0.3.3.pdf)
- [Espressif ESP32-S3 GPIO](https://docs.espressif.com/projects/esp-idf/en/v5.0.4/esp32s3/api-reference/peripherals/gpio.html)
- [Espressif ESP32-S3 Hardware Design Guidelines（strapping）](https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/schematic-checklist.html)
- [Espressif ESP32-S3 USB Device Stack](https://docs.espressif.com/projects/esp-usb/en/latest/esp32s3/usb_device.html)
- [NXP PCA9685 data sheet](https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf)
- [TDK/InvenSense INMP441 data sheet](https://invensense.tdk.com/wp-content/uploads/2015/02/INMP441.pdf)
- [Analog Devices MAX98357A/MAX98357B data sheet](https://www.analog.com/media/en/technical-documentation/data-sheets/MAX98357A-MAX98357B.pdf)
- [Adafruit MAX98357A mono breakout pinout](https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp/pinouts)
- [Adafruit Product 2478 ILI9341 TFT pinout](https://learn.adafruit.com/adafruit-2-4-color-tft-touchscreen-breakout/pinouts)
- `D:/GitHub/IchiPing-UNO-Q/docs/HANDOFF_2026-09-12.md`（実機通過済みI²S形式／現行接続）
- `D:/GitHub/IchiPing-UNO-Q/uno_q/audio/reports/2026-09-11-microphone-pass.md`（INMP441実測）
