# 接続対応表（ハーネス正本）

すべて基板上面から見たJSTヘッダのpin 1からの順です。純正JST XHは2.50 mmピッチです。基板シルクのpin 1表示、ハウジングのロック向き、圧着端子側の挿入向きを必ず導通計で確認します。

## Solist-AI 2x7 MILコネクタ

| Pin | Solist機能 | 中間基板net | 接続先 |
|---:|---|---|---|
| 1 | P73 / I2C SCL | I2C_SCL | Stamp G15、PCA SCL |
| 2 | GND | GND | 共通GND |
| 3 | P74 / I2C SDA | I2C_SDA | Stamp G13、PCA SDA |
| 4 | GND | GND | 共通GND |
| 5 | Power out | NC | 使用しない |
| 6 | P23 | TFT_RST_N | J_TFT_PWR-1 |
| 7 | P22 | TFT_DC | J_TFT_SIG-5 |
| 8 | Power out | NC | 使用しない |
| 9 | P41 / SPI MOSI | TFT_MOSI | J_TFT_SIG-4 |
| 10 | P42 | PCA_OE | J_PCA_OE-1、10k pull-up |
| 11 | GND | GND | 共通GND |
| 12 | P40 / SPI SCK | TFT_SCK | J_TFT_SIG-3 |
| 13 | GND | GND | 共通GND |
| 14 | P43 / SPI CS | TFT_CS_N | J_TFT_PWR-2、10k pull-up |

MILコネクタは奇数列と偶数列で並び、ケーブル側から見ると鏡像になります。番号ではなく外観の左右だけで配線しません。

## Stamp-S3A S007-V033着脱ソケット

Stamp側には左列1x17・1.27 mmと右列1x6・2.54 mmのピンヘッダを立て、部品面を上にして中間基板の雌ソケットへ載せます。左列は2.54 mm 1x9と偶数接点用部品の樹脂干渉を避けるため、全17極を1.27 mmへ置換します。**中間基板を上面から見たソケット列は、公式のStamp部品面PinMapに対して左右鏡像です。** 下表は見た目の左右位置ではなくM1 pad番号を正とします。USB／アンテナ方向、角形pad 1、シルクの外形を確認してから挿入します。

| M1 pad | GPIO | 配置ピッチ | 中間基板net | 接続先 |
|---:|---|---|---|---|
| 1 | G1 | 左1.27 mm列 | AMP_SD | J_AMP_PWR-1、10k GND pull-down |
| 2 | G2 | 1.27 mm位置 | SW_WIN_B | J_WIN_B-1 |
| 3 | G3 | 左1.27 mm列 | NC | strappingのため未接続 |
| 4 | G4 | 1.27 mm位置 | SW_WIN_C | J_WIN_C-1 |
| 5 | G5 | 左1.27 mm列 | I2S_WS_SRC | 33Ω後mic/ampへ |
| 6 | G6 | 1.27 mm位置 | SW_DOOR_AB | J_DOOR_AB-1 |
| 7 | G7 | 左1.27 mm列 | I2S_DOUT_SRC | 33Ω後amp DINへ |
| 8 | G8 | 1.27 mm位置 | SW_DOOR_BC | J_DOOR_BC-1 |
| 9 | G9 | 左1.27 mm列 | I2S_DIN_MIC | mic SD、100k GND pull-down |
| 10 | G10 | 左1.27 mm列 | EXEC_N | J_EXEC-1、active Low |
| 11 | GND | 左1.27 mm列 | GND | 共通GND |
| 12 | G11 | 左1.27 mm列 | NC | 未接続 |
| 13 | 5V | 左1.27 mm列 | +5V_STAMP | JP_STAMP_5V経由 |
| 14 | G12 | 左1.27 mm列 | NC | 未接続 |
| 15 | G13 | 左1.27 mm列 | I2C_SDA | Solist/PCA共有 |
| 16 | G14 | 左1.27 mm列 | NC | 未接続 |
| 17 | G15 | 左1.27 mm列 | I2C_SCL | Solist/PCA共有 |
| 18 | GND | 右2.54 mm列 | GND | 共通GND |
| 20 | G0 | 右2.54 mm列 | NC | BOOT strappingのため未接続 |
| 22 | EN | 右2.54 mm列 | NC | 未接続 |
| 24 | G44 | 右2.54 mm列 | SW_WIN_A | J_WIN_A-1 |
| 26 | G43 | 右2.54 mm列 | I2S_BCLK_SRC | 33Ω後mic/ampへ |
| 28 | 3V3 | 右2.54 mm列 | +3V3_STAMP | mic/PCA logic/pull-up |

左列M1-1..17は全17極を1.27 mm着脱接点、右列M1-18/20/22/24/26/28は6極を2.54 mm着脱接点とします。Stampが5状態とEXEC_Nを直接読み、I²C status/eventでSolistへ通知します。

## 純正JST XH外部コネクタ

| Ref | Header / housing | Pin 1からの順 | 接続対象／注意 |
|---|---|---|---|
| J_WIN_A | B2B-XH-A / XHP-2 | WIN_A, GND | Low=CLOSE |
| J_WIN_B | B2B-XH-A / XHP-2 | WIN_B, GND | Low=CLOSE |
| J_WIN_C | B2B-XH-A / XHP-2 | WIN_C, GND | Low=CLOSE |
| J_DOOR_AB | B2B-XH-A / XHP-2 | DOOR_AB, GND | Low=CLOSE |
| J_DOOR_BC | B2B-XH-A / XHP-2 | DOOR_BC, GND | Low=CLOSE |
| J_EXEC | B2B-XH-A / XHP-2 | EXEC_N, GND | active Low |
| J_MIC | B6B-XH-A / XHP-6 | GND, 3V3, SD, SCK/BCLK, WS, L/R=GND | INMP441。pin 2へ5 V禁止。UNO Q版1.8 Vとの差に注意 |
| J_AMP_SIG | B4B-XH-A / XHP-4 | LRC/WS, BCLK, DIN, GAIN=GND | MAX98357A、GAIN=12 dB想定 |
| J_AMP_PWR | B3B-XH-A / XHP-3 | SD_MODE, GND, +5V_LOGIC | pin 1はStamp G1。UNO Qと端子機能順は同じ |
| J_TFT_SIG | B5B-XH-A / XHP-5 | MISO=NC, BL=+5V, SCK, MOSI, D/C | ILI9341 write-only。現物BL定格を確認 |
| J_TFT_PWR | B4B-XH-A / XHP-4 | RST, CS, GND, VCC=+5V | 現物モジュールの5 V入力対応を確認 |
| J_SERVO_CTRL | B4B-XH-A / XHP-4 | GND, SCL, SDA, VCC=3.3V | UNO Qと同順。VCCはPCA logic、servo V+ではない |
| J_PCA_OE | B2B-XH-A / XHP-2 | OE, GND | 新設。PCA9685 OEへ。10kで3.3 V pull-up |
| J_SERVO_5V_OUT | B2B-XH-A / XHP-2 | +5V_LOGIC, GND | 共通5 VからPCA9685 V+へ出力 |
| J_PWR_LOGIC | B2B-XH-A / XHP-2 | +5V_LOGIC, GND | 共通5 V入力（J_PWR_SERVOと並列） |
| J_PWR_SERVO | B2B-XH-A / XHP-2 | +5V_LOGIC, GND | 共通5 V入力（J_PWR_LOGICと並列） |

PCA9685の出力割当は`ch0=WIN_A, ch1=WIN_B, ch2=WIN_C, ch3=DOOR_AB, ch4=DOOR_BC`です。SG90はPCA9685モジュールの3ピン端子へ接続します。

`J_PWR_LOGIC`と`J_PWR_SERVO`は基板上で同一レールです。両方を使う場合も必ず同じ5 V電源から配線し、異なる電源やUSB由来5 Vを並列接続しません。5本のサーボ電流を単一XH接点へ集中させないよう、電源容量・配線・コネクタ定格を確認します。

## PCBA部品候補

JLCPCB PCBAの候補LCSC番号は、33 Ω=`C23140`、10 kΩ=`C25804`、100 kΩ=`C25803`、100 nF=`C14663`、JST XH 2P=`C265283`、3P=`C144394`、4P=`C144395`、5P=`C157991`、6P=`C144397`、左列1.27 mm雌ソケット=`C41360907`、Stamp側雄ヘッダ=`C41360852`、右列1x6・2.54 mmソケット=`C42431861`です。**在庫、部品仕様、フットプリント、実装可否、pin 1と回転は発注直前にJLCPCBで再確認します。** ソケットがPCBA困難なら未実装にし、左列は秋月電子の1.27 mm雌ソケット`103866`／雄ヘッダ`103865`を1x17へ切って手はんだします。右列1x6・2.54 mmにも適合ソケットを手はんだします。

## ケーブル製作検査

1. 旧2.54 mmハウジングを流用せず、純正XHPハウジングと適合SXH圧着端子を使う。
2. 各ケーブルを両端のpin番号で記録し、色だけを根拠にしない。
3. 無通電で全極の導通、隣接極短絡、GND、電源極性を検査する。
4. 電流制限付きの単一5 V電源で共通レールを確認し、2個の入力XH間が同極性で導通することを確認する。
5. マイクpin 2=3.3 V、PCA pin 4=3.3 V、amp pin 3=5 V、servo pin 1=5 Vを実測してからモジュールを挿す。
