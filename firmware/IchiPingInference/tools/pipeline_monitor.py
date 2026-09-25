"""Stamp → I2C → Solist 推論パイプライン試験 (build.ps1 -Main ichi_pipeline_main) の UART 結果を受信・照合する。

Solist は起動直後 (と SW2 押下時) に Stamp から baseline → 32 状態のクリップを I2C で読み、N333 特徴を実機で計算して
推論し、クリップ毎に PIPE_CLIP を送る。本スクリプトは受信して generated/pipeline_expected.json
(sim/board_fixed_feature.py の固定小数点参照) と比べ、結果を docs/board_pipeline_test.{md,json} に書く。

usage: python firmware/IchiPingInference/tools/pipeline_monitor.py --port COM18 [--timeout 900]
       (起動後に Solist をリセットするか SW2 を押す)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
import sys
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from ichi_serial import decode_frame  # noqa: E402

EXPECTED = HERE.parent / "generated" / "pipeline_expected.json"
PIPE_INFO, PIPE_CLIP, PIPE_DONE, PIPE_ERROR = 0x30, 0x31, 0x32, 0x33
STAGES = {1: "INFO (Stamp 応答なし)", 2: "CLIP 情報", 3: "PCM 読み出し", 4: "推論"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM18")
    ap.add_argument("--timeout", type=float, default=900.0, help="秒 (全クリップ完了まで)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "board_pipeline_test"))
    args = ap.parse_args()
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))
    exp_by_class = {r["class_id"]: r for r in exp["states"]}

    ser = serial.Serial(args.port, 115200, timeout=0.1)
    buf = b""
    info, clips, done, error = None, [], None, None
    t0 = time.time()
    print(f"waiting for the Solist on {args.port} (reset it or press SW2) ...", flush=True)
    while time.time() - t0 < args.timeout and done is None and error is None:
        buf += ser.read(4096)
        while b"\x00" in buf:
            enc, buf = buf.split(b"\x00", 1)
            fr = decode_frame(enc) if enc else None
            if fr is None:
                continue
            p = fr.payload
            if fr.type == PIPE_INFO:
                info = dict(clips=p[0], baseline=p[1], samples=struct.unpack_from("<H", p, 2)[0])
                clips.clear()
                print(f"INFO: {info}", flush=True)
            elif fr.type == PIPE_CLIP:
                clip, cls, pred, pred_h = p[0], p[1], p[2], p[3]
                mism, maxd, expn = struct.unpack_from("<H", p, 4)[0], p[6], p[7]
                xfer_ms, feat_ms, retries = struct.unpack_from("<IHH", p, 8)
                row = dict(clip=clip, class_id=None if cls == 0xFF else cls, xfer_ms=xfer_ms, feat_ms=feat_ms,
                           retries=retries, exponent=expn)
                if cls != 0xFF:
                    out = list(struct.unpack_from(f"<{(len(p) - 16) // 4}f", p, 16))
                    e = exp_by_class.get(cls, {})
                    row.update(pred=pred, pred_header=pred_h, mismatch=mism, max_diff=maxd, outputs=out,
                               exp_pred=e.get("pred_fixed"), exp_mismatch=e.get("mismatch"),
                               exp_outputs=e.get("outputs_fixed"))
                    if e.get("outputs_fixed"):
                        row["max_out_diff"] = max(abs(a - b) for a, b in zip(out, e["outputs_fixed"]))
                    print(f"  clip {clip:2d} class {cls:2d}: pred {pred:2d} (PC fixed {e.get('pred_fixed')}, header {pred_h})"
                          f"  int8 diff {mism:3d} (PC {e.get('mismatch')}) max {maxd}  "
                          f"|Δout| {row.get('max_out_diff', float('nan')):.4f}  xfer {xfer_ms} ms  retries {retries}",
                          flush=True)
                else:
                    print(f"  clip {clip:2d} baseline  xfer {xfer_ms} ms  exp {expn}  retries {retries}", flush=True)
                clips.append(row)
            elif fr.type == PIPE_DONE:
                done = dict(correct=p[0], agree=p[1], states=p[2], ok=bool(p[3]),
                            total_s=struct.unpack_from("<H", p, 4)[0],
                            retries=struct.unpack_from("<I", p, 6)[0], bytes=struct.unpack_from("<I", p, 10)[0])
                print(f"DONE: {done}", flush=True)
            elif fr.type == PIPE_ERROR:
                error = dict(stage=p[0], stage_name=STAGES.get(p[0], "?"), clip=p[1])
                print(f"ERROR: {error}", flush=True)
    ser.close()
    if done is None and error is None:
        print("timeout", flush=True)

    states = [c for c in clips if c.get("class_id") is not None]
    n = len(states)
    same_pred = sum(c["pred"] == c["exp_pred"] for c in states)
    same_mism = sum(c["mismatch"] == c["exp_mismatch"] for c in states)
    correct = sum(c["pred"] == c["class_id"] for c in states)
    exp_correct = sum(c["exp_pred"] == c["class_id"] for c in states)
    xfer = [c["xfer_ms"] for c in clips]
    summary = dict(date=dt.datetime.now().isoformat(timespec="seconds"), port=args.port, info=info, done=done,
                   error=error, states=n, board_correct=correct, pc_fixed_correct=exp_correct,
                   header_correct=exp.get("header_correct"), pred_equal_pc_fixed=same_pred,
                   int8_mismatch_equal_pc_fixed=same_mism,
                   max_output_diff=max((c.get("max_out_diff", 0.0) for c in states), default=None),
                   xfer_ms_mean=(sum(xfer) / len(xfer)) if xfer else None)
    Path(args.out + ".json").write_text(json.dumps(dict(summary=summary, clips=clips), indent=1), encoding="utf-8")
    lines = ["# Stamp → I2C → Solist 推論パイプライン試験", "",
             f"- 日時: {summary['date']}  ポート: {args.port}",
             f"- 入力: samples/uno_q_eval_evening (baseline {info['baseline'] if info else '?'} + 状態 {n} クリップ, 16 kHz PCM16)",
             "- 経路: Stamp-S3A Flash → I2C 400 kHz (0x42) → Solist: 固定小数点 FFT/N333 → int8 前段 → ELM (AxlCORE)", "",
             "| 項目 | 実機 | PC 参照 |", "|---|---:|---:|",
             f"| 正解 (32 状態) | {correct}/{n} | 固定小数点 {exp_correct}/{n}, 浮動小数点入力 {exp.get('header_correct')}/32 |",
             f"| 予測クラスが PC 固定小数点参照と一致 | {same_pred}/{n} | — |",
             f"| int8 入力の不一致数が PC 固定小数点参照と一致 | {same_mism}/{n} | — |",
             f"| 出力の最大差 (PC 固定小数点参照比) | {summary['max_output_diff']} | — |",
             f"| 1 クリップの転送+FFT 時間 (平均) | {summary['xfer_ms_mean']} ms | — |",
             f"| I2C 再送 | {done['retries'] if done else '?'} | — |", ""]
    if error:
        lines += [f"**エラー**: stage {error['stage']} ({error['stage_name']}), clip {error['clip']}", ""]
    lines += ["| clip | class | 実機 pred | PC 固定 pred | header pred | int8 差 (実機/PC) | 転送 ms |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in states:
        lines.append(f"| {c['clip']} | {c['class_id']} | {c['pred']} | {c['exp_pred']} | {c['pred_header']} | "
                     f"{c['mismatch']}/{c['exp_mismatch']} | {c['xfer_ms']} |")
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}.md / .json")


if __name__ == "__main__":
    main()
