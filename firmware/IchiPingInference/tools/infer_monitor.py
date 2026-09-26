"""推論ファーム (build.ps1 -Main ichi_infer_main) の UART 出力を表示し、JSON Lines で記録する。

  0x50 BASE     index u8 | exponent u8 | transfer ms u32
  0x51 RESULT   actual u8 | pred u8 | calibrated u8 | exponent u8 | transfer ms u32 | outputs float32 x 32
  0x52 CAL_STEP state u8 | window u8 | pred before update u8 | 0 | transfer ms u32
  0x53 CAL_DONE correct-before u16 | samples u16 | aborted u8 | seconds u16
  0x54 ERROR    stage u8 | state u8

usage: python firmware/IchiPingInference/tools/infer_monitor.py --port COM3 [--log docs/board_infer_log.jsonl] [--seconds 0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ichi_serial import decode_frame  # noqa: E402

STAGES = {1: "Stamp 応答なし", 3: "測定", 4: "PCM 読み出し", 5: "推論", 6: "サーボ", 7: "校正の学習"}


def label(c: int) -> str:
    """"h" + 0/1 in the TFT order C, BC, B, AB, A (1 = OPEN), e.g. h01101."""
    return "h" + "".join(str((c >> k) & 1) for k in (2, 4, 1, 3, 0))


def class14(c: int) -> int:
    a, b, cc, ab, bc = [(c >> k) & 1 for k in range(5)]
    return a if ab == 0 else (2 + a + 2 * b if bc == 0 else 6 + a + 2 * b + 4 * cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--log", default="")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = Ctrl+C まで")
    a = ap.parse_args()
    s = serial.Serial(a.port, 115200, timeout=0.2)
    log = open(a.log, "a", encoding="utf-8") if a.log else None
    buf, t0 = bytearray(), time.time()
    try:
        while a.seconds <= 0 or time.time() - t0 < a.seconds:
            buf += s.read(4096)
            while b"\x00" in buf:
                chunk, _, rest = bytes(buf).partition(b"\x00")
                buf = bytearray(rest)
                fr = decode_frame(chunk) if chunk else None
                if fr is None:
                    continue
                p, rec = fr.payload, dict(time=dt.datetime.now().isoformat(timespec="seconds"), type=fr.type)
                if fr.type == 0x50:
                    i, e, ms = struct.unpack_from("<BBI", p)
                    rec.update(kind="baseline", index=i, exponent=e, transfer_ms=ms)
                    print(f"baseline {i + 1}: 転送 {ms} ms")
                elif fr.type == 0x51:
                    act, pred, cal, e, ms = struct.unpack_from("<BBBBI", p)
                    out = np.frombuffer(p, "<f4", count=32, offset=8).tolist()
                    rec.update(kind="result", actual=act, pred=pred, calibrated=bool(cal), transfer_ms=ms, outputs=out)
                    verdict = "完全一致" if pred == act else ("14クラス一致" if class14(pred) == class14(act) else "不一致")
                    print(f"推論 実際 {label(act)} -> 推論 {label(pred)}  {verdict}  ({'校正済み' if cal else '工場モデル'}, 転送 {ms} ms)")
                elif fr.type == 0x52:
                    st, w, pred, _, ms = struct.unpack_from("<BBBBI", p)
                    rec.update(kind="cal_step", state=st, window=w, pred_before=pred, transfer_ms=ms)
                    print(f"  校正 {label(st)} 窓{w}: 更新前の推論 {label(pred)} {'○' if pred == st else '×'}  (転送 {ms} ms)")
                elif fr.type == 0x53:
                    c, n, ab, sec = struct.unpack_from("<HHBH", p)
                    rec.update(kind="cal_done", correct_before=c, samples=n, aborted=bool(ab), seconds=sec)
                    print(f"校正{'中断' if ab else '完了'}: 更新前の正解 {c}/{n}, {sec} 秒")
                elif fr.type == 0x54:
                    rec.update(kind="error", stage=p[0], state=p[1])
                    print(f"エラー: {STAGES.get(p[0], p[0])} ({label(p[1])})")
                if log:
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
