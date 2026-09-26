"""このハード (Solist サーボ + Stamp-S3A の PRBS 再生・INMP441 録音) で 32 状態のセッションを採取する。

Solist に ichi_collect_main を書き込んでおく (build.ps1 -Main ichi_collect_main)。
PC が Solist (UART, 'S' + 状態) でサーボを動かし、Stamp (USB CDC, firmware/StampMeasure) に
'M' で測定させ 'D' で揃えた 2 秒 16 kHz フレームを受け取る。保存形式は UNO Q の captures と同じ:
  <out>/sXXXXX/frame_NNNNNN.wav (16 kHz mono int16) + meta.json (全閉の先頭 baseline 枚は group=baseline)
  (sXXXXX の k 文字目 = bit k-1: a b c AB BC, 1 = OPEN)
順序は UNO Q collector と同じく全閉 baseline → Gray code で 32 状態 (各状態 --frames 枚連続)。

usage: python firmware/IchiPingInference/tools/collect_session.py --name s1 [--frames 10] [--baseline 10]
       [--solist COM3] [--stamp COM13] [--rotate 0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from ichi_serial import decode_frame  # noqa: E402

MOVED = 0x48


def dirname(c: int) -> str:
    return "s" + "".join(str((c >> k) & 1) for k in range(5))


class Solist:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.2, write_timeout=2.0)
        self.buf = bytearray()

    def move(self, state: int, timeout=30.0) -> bool:
        self.s.reset_input_buffer(); self.buf.clear()
        self.s.write(bytes([ord("S"), state]))
        t = time.time()
        while time.time() - t < timeout:
            self.buf += self.s.read(256)
            while 0 in self.buf:
                i = self.buf.index(0)
                raw, self.buf[:] = bytes(self.buf[:i]), self.buf[i + 1:]
                try:
                    f = decode_frame(raw)
                except Exception:
                    continue
                if f is not None and f.type == MOVED and f.payload[0] == state:
                    return f.payload[1] == 1
        raise TimeoutError(f"Solist: no MOVED for state {state}")

    def release(self):
        self.s.write(b"R")


class Stamp:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=1.0)
        time.sleep(0.3); self.s.reset_input_buffer()

    def line(self, timeout):
        t = time.time()
        while time.time() - t < timeout:
            l = self.s.readline()
            if l:
                return l.decode(errors="replace").strip()
        raise TimeoutError("Stamp: no reply")

    def measure(self, retries=3):
        for _ in range(retries):
            self.s.reset_input_buffer()
            self.s.write(b"M\n")
            l = ""
            t = time.time()
            while not l.startswith("MEAS") and time.time() - t < 10:
                l = self.line(10)
            if " ok " not in l:
                print("   ", l, "-> retry", flush=True); continue
            info = dict(kv.split("=") for kv in l.split() if "=" in kv)
            self.s.write(b"D\n")
            h = self.line(5)
            if not h.startswith("PCM16"):
                continue
            n = int(h.split()[1])
            data = self.s.read(2 * n)
            end = self.line(5)
            if len(data) != 2 * n or end != "END":
                print("    short dump -> retry", flush=True); continue
            return np.frombuffer(data, "<i2").copy(), int(info["onset"]), float(info["rms"])
        raise RuntimeError("Stamp: measurement failed")


def write_wav(p: Path, x: np.ndarray):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="セッション名 (出力 captures/stamp_<日付>_<name>_wav)")
    ap.add_argument("--frames", type=int, default=10, help="状態毎の frame 数")
    ap.add_argument("--baseline", type=int, default=10, help="開始時の全閉 baseline frame 数")
    ap.add_argument("--rotate", type=int, default=0, help="Gray code 列の開始位置 (セッション毎に変えると順序効果が散る)")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--solist", default="COM3")
    ap.add_argument("--stamp", default="COM13")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "captures" / f"stamp_{dt.date.today():%Y%m%d}_{args.name}_wav"
    if out.exists():
        raise SystemExit(f"already exists: {out}")
    out.mkdir(parents=True)
    solist, stamp = Solist(args.solist), Stamp(args.stamp)
    order = [(p ^ (p >> 1)) for p in range(32)]
    order = order[args.rotate:] + order[:args.rotate]
    if args.reverse:
        order = order[::-1]
    metas: dict[int, dict] = {}
    t0 = time.time()

    def grab(state, group):
        d = out / dirname(state)
        d.mkdir(exist_ok=True)
        m = metas.setdefault(state, dict(label=dirname(state), source="Stamp-S3A StampMeasure 'M'+'D'",
                                         quantize="int16", capture="Stamp-S3A I2S 48k -> FIR 16k, x16 scale",
                                         frames=[]))
        x, onset, rms = stamp.measure()
        name = f"frame_{len(m['frames']):06d}.wav"
        write_wav(d / name, x)
        m["frames"].append(dict(wav=name, group=group, time=dt.datetime.now().isoformat(timespec="seconds"),
                                onset=onset, rms=rms / 32768.0, clipped_samples=int(np.sum(np.abs(x) >= 32767))))
        (d / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        return rms

    try:
        if not solist.move(0):
            raise RuntimeError("servo error")
        for i in range(args.baseline):
            r = grab(0, "baseline")
            print(f"baseline {i + 1}/{args.baseline} rms={r:.0f}", flush=True)
        for n, c in enumerate(order):
            if not solist.move(c):
                raise RuntimeError(f"servo error at {dirname(c)}")
            rs = [grab(c, "state") for _ in range(args.frames)]
            el = time.time() - t0
            print(f"{n + 1:2d}/32 {dirname(c)} rms={np.mean(rs):.0f}  {el / 60:.1f} min "
                  f"(残り {el / (n + 1) * (31 - n) / 60:.1f} min)", flush=True)
    finally:
        solist.release()
    (out / "session.json").write_text(json.dumps(dict(
        started=dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"), seconds=round(time.time() - t0),
        frames=args.frames, baseline=args.baseline, rotate=args.rotate, reverse=args.reverse, order=order), indent=1),
        encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
