# JLCPCB発注データ

`stamp_s3a_interposer_pcba_rev_b/`がJLCPCB PCBA用の生成物です。

## アップロードする3ファイル

1. `GERBER_JLCPCB.zip` — PCB製造画面へアップロード
2. `BOM_JLCPCB.csv` — PCB AssemblyのBOM欄へアップロード
3. `CPL_JLCPCB.csv` — PCB AssemblyのCPL欄へアップロード

推奨する初回試作条件は、2層、FR-4、外形110 x 72 mm、板厚1.6 mm、銅厚1 oz、HASL lead-free、緑レジスト、白シルクです。部品はすべて表面実装指定ですが、THT品を含むためJLCPCB画面ではwave soldering／hand soldering対象と実装追加費用を確認してください。

## 発注画面で必ず止めて確認する項目

- Gerberプレビューで外形が110 x 72 mm、取付穴4個、両面GNDベタ、Stampアンテナ直下の銅箔禁止領域が正しいこと
- BOMが34 designatorに展開され、CPLも34行で一致すること
- `R1..R9`、`C2`、`C4`は0603 SMT、その他は表面側THTとして認識されること
- 電解コンデンサ`C1`、`C3`、`C5`の極性をシルクと部品プレビューで照合すること
- `J_STAMP_17`、`J_STAMP_6`、`J_SOLIST`のpin 1、開口面、嵌合高さを現物と照合すること
- JST XH各コネクタのpin 1とハーネス信号順を`hardware/stamp_s3a_interposer/CONNECTIONS.md`と照合すること
- 自動代替を無効にし、LCSC番号・在庫・実装方式・部品向きを確認すること
- Stamp-S3A本体、Stamp側1x17雄ヘッダ、2P shuntはJLCPCB実装対象外であること

上記の機械部品・コネクタ方向は、現物照合が終わるまで注文確定しないでください。特にStamp-S3Aは中間基板上面から見て公式部品面PinMapの左右鏡像です。

## 再生成

KiCad 10をインストールしたWindows PCで、次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\hardware\stamp_s3a_interposer\package_jlcpcb.ps1
```

スクリプトはERCとPCB／回路図parityを検査してから、Gerber、Excellon drill、BOM、CPL、SHA-256 manifestを再生成します。DRCレポートには既知のシルク警告が残るため、`reports/drc_pcba.rpt`も発注前に確認してください。
