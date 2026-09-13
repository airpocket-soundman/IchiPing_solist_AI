# IchiPing Solist-AI / Stamp-S3A 中間基板

KiCad 10用のRev.B試作設計です。Solist-AIを主制御、Stamp-S3AをI2S音響・GPIO補完に使い、外部配線を純正JST XH（公称2.50 mm）へ集約します。

> **起動時の注意:** このPCにはKiCad 8と10が共存していますが、本設計のPCBはKiCad 10形式です。`.kicad_pro`の関連付けからKiCad 8が起動する場合があるため、`C:\Program Files\KiCad\10.0\bin\kicad.exe`からプロジェクトを開いてください。

## 設計ファイル

- `stamp_s3a_interposer_pcba.kicad_pro` / `.kicad_sch` / `.kicad_pcb`: JLCPCB PCBA版
- `stamp_s3a_interposer_tht.kicad_pro` / `.kicad_sch` / `.kicad_pcb`: 手はんだTHT版
- `stamp_s3a_interposer.kicad_sch`: 旧コネクタ接続表（参照用。設計正本ではない）
- `CONNECTIONS.md`: ハーネス製作時の正本
- `BOM.csv`: 版別BOMの索引
- `BOM_PCBA.csv` / `BOM_THT.csv`: JLCPCB実装版／手はんだ版の部品表
- `tools/generate_board.py`: 配置・ネット・初期配線の再生成
- `tools/generate_schematic.py`: parity確認済みPCBA回路図からTHT回路図を再生成し、ERC／PCB parityを検証
- `tools/import_route.py`: Specctra SESの取込みと線幅正規化
- `validate.ps1`: ERC、DRC、Gerber、ドリル、レンダリング

## 互換性方針

UNO Q基板の**ピン番号と信号順**を維持します。旧基板の外部コネクタは2.54 mmのXH互換品でしたが、本基板はユーザー指定どおり純正JST XH 2.50 mmです。このため旧ハウジングをそのまま挿す機械互換性はありません。`XHP-n`ハウジングでケーブルを作り直し、導線順を同じにします。

例外は次の2点です。

- `J_MIC` pin 2はStamp-S3Aに合わせて3.3 V。UNO Q音響基板の1.8 Vケーブルを無確認で流用しない。
- PCA9685の安全停止用`J_PCA_OE`を追加。従来の`J_SERVO_CTRL` 4極はそのまま維持する。

## 基板構成

- 外形: 110 x 72 mm、2層、1.6 mm FR-4、1 oz想定
- 外部コネクタ: 純正JST `B?B-XH-A` 垂直THT、2.50 mm
- Solist: 2x7、2.54 mm MILコネクタ。奇数／偶数列の向きとpin 1を現物で照合する
- Stamp-S3A: 通常版S007-V033を左列1x17・1.27 mm雌ソケットと右列1x6・2.54 mm雌ソケットへ搭載。Stamp側に対応ピンを立て、部品面を上向きにする
- Stampソケット: 中間基板上面から見た左右列は公式の部品面PinMapに対して左右鏡像。USB／アンテナ方向、M1 pad番号、pin 1をシルクで明示する
- 左列は2.54 mm 1x9と偶数接点用1.27 mm部品の樹脂干渉を避けるため、M1-1..17全体を1.27 mmへ置換する
- 入力: Stamp G44=窓a、G2=窓b、G4=窓c、G6=扉AB、G8=扉BC、G10=EXEC_N。Stampが直接読みSolistへ通知する
- 信号線0.20 mm、3.3 V 0.30 mm、logic 5 V 0.40 mm、servo 5 V 1.00 mm、clearance 0.20 mm
- F.Cu/B.CuともGNDベタを設け、GNDパッドは低インピーダンス優先で直結する。Stamp antenna直下だけは両面とも銅箔・配線禁止とする
- Servo電源とlogic/audio電源は別入力、GNDのみ共通

## 電源・安全条件

- `J_PWR_LOGIC`: Stamp、MAX98357A、TFT用の安定化5 V
- `J_PWR_SERVO`: PCA9685 V+／SG90用の別5 V
- `JP_STAMP_5V`: 外部5 VでStampを給電するときだけ短絡。StampをPC USBから給電するときは必ず開放
- `AMP_SD`: Stamp G1制御、10 kΩ GND pull-downでreset中shutdown
- `PCA_OE`: Solist P42制御、10 kΩ Stamp 3.3 V pull-upでreset中PWM停止
- `TFT_CS_N`: 10 kΩ Stamp 3.3 V pull-up
- PCA9685 logic VCCとservo V+を混同しない

## JLCPCB PCBAとTHTフォールバック

JLCPCB PCBA向けの第一候補は次のとおりです。

| 機能 | LCSC候補 |
|---|---|
| 10 kΩ | `C25804` |
| 100 nF | `C14663` |
| 33 Ω / 10 kΩ / 100 kΩ | `C23140` / `C25804` / `C25803` |
| JST XH 2P / 3P / 4P / 5P / 6P | `C265283` / `C144394` / `C144395` / `C157991` / `C144397` |
| Stamp左列1.27 mm雌ソケット / module側雄ヘッダ | `C41360907` / `C41360852` |
| Stamp右列1x6・2.54 mm雌ソケット | `C42431861` |

**LCSC番号、在庫、基本／拡張部品区分、最小発注数、実装可否、実部品寸法、pin 1と実装向きは発注直前にJLCPCB Partsで再確認します。** 自動代替は許可せず、CPLの回転と基板シルクを照合します。

ソケットのPCBA調達が難しい場合は、その部品を未実装（DNP）にし、左列には秋月電子の1.27 mm雌ソケット（通販コード`103866`）とStamp側雄ヘッダ（`103865`）を1x17へ切り分けて手はんだします。右列1x6・2.54 mmにも適合ソケットを手はんだするTHT版を用意します。外部JST XHも必要に応じて未実装とし、同じTHT穴へ純正品を手はんだします。PCBA版とTHT版でネット、穴位置、pin 1、ハーネス順は変えません。

`BOM_THT.csv`の秋月調達品には、通販コードに加えて公式商品ページの直接URLを記載しています。価格と在庫は発注時に再確認します。

## 検証状態

2026-09-13にKiCad 10.0.6で確認済みです。

- 回路図ERC: 0 error / 0 warning
- PCB connectivity: 未配線0
- PCB electrical DRC: short、clearance、track/via、courtyard違反0
- 残件: KiCad標準フットプリント内のシルク同士／シルクとパッドのwarningはPCBA版147件、THT版153件。電気違反ではないが、量産前に参照文字と注意書きを整理する

`powershell -ExecutionPolicy Bypass -File .\validate.ps1`でレポートと製造出力を`build/`へ再生成できます。

## 製造前の停止条件

このRev.Bは回路・ピン割当の試作版です。次を現物で確認するまで発注しません。

1. Solist-AI現物のCN3/CN1名称、2x7コネクタのgender、pin 1位置、嵌合方向
2. Stamp-S3A S007-V033を部品面上向きで載せたときの左右鏡像、左1x17・1.27 mm／右1x6・2.54 mmソケット中心、USB挿抜空間、アンテナ方向、M1 pad 1位置
3. INMP441、MAX98357A、ILI9341、PCA9685各モジュールのメーカー、SKU、revision、端子シルク
4. TFTのVCC/BL電圧、PCA9685のOE端子と既設pull-up、I2C pull-up合成値
5. Stamp USB給電と外部5 Vの同時給電禁止が`JP_STAMP_5V`運用で守れること
6. XHハウジングを作り直した後の全極導通検査とpin 1照合
7. JLCPCB部品在庫、LCSC番号、CPL回転、またはTHT手はんだ版のDNP指定

サーボはUNO Qと同じくPCA9685モジュールch0..4へ接続します。中間基板には個別SG90用XH3を追加していません。
