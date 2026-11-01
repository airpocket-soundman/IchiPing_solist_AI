"""このハード (Solist サーボ + Stamp-S3A の PRBS 再生・INMP441 録音) で 32 状態のセッションを採取する。

Solist に ichi_collect_main を書き込んでおく (build.ps1 -Main ichi_collect_main)。
PC は Solist (UART) だけへ接続する。'S' + 状態でサーボを動かし、'M' でSolistにStampの
I2C測定とPCM中継をさせる。StampのUSB CDC接続は不要。保存形式は UNO Q の captures と同じ:
  <out>/sXXXXX/frame_NNNNNN.wav (16 kHz mono int16) + meta.json (全閉の先頭 baseline 枚は group=baseline)
  (sXXXXX の k 文字目 = bit k-1: a b c AB BC, 1 = OPEN)
順序は UNO Q collector と同じく全閉 baseline → Gray code で 32 状態 (各状態 --frames 枚連続)。

usage: python firmware/IchiPingInference/tools/collect_session.py --name s1 [--frames 10] [--baseline 10]
       [--solist COM3] [--rotate 0]
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
    def __init__(self, port, baud=500000):
        self.s = serial.Serial(port, baud, timeout=0.2, write_timeout=2.0)
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

    def _frame(self, timeout: float):
        """Read one valid COBS frame emitted by the collection firmware."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Drain available bytes immediately; read(512) waits for its timeout
            # at the end of every short UART chunk.
            if 0 not in self.buf:
                self.buf += self.s.read(max(1, self.s.in_waiting))
            while 0 in self.buf:
                i = self.buf.index(0)
                raw, self.buf[:] = bytes(self.buf[:i]), self.buf[i + 1:]
                try:
                    f = decode_frame(raw)
                except Exception:
                    continue
                if f is not None:
                    return f
        raise TimeoutError("Solist: no collection frame")

    def measure(self, retries=2) -> np.ndarray:
        """Ask Solist to measure through Stamp I2C and relay the complete PCM frame."""
        for attempt in range(retries):
            try:
                self.s.reset_input_buffer(); self.buf.clear()
                self.s.write(b"M")
                header = self._frame(20.0)
                if header.type != 0x49 or len(header.payload) != 3:
                    raise ValueError("unexpected capture header")
                ok, samples = header.payload[0], int.from_bytes(header.payload[1:3], "little")
                if not ok or samples != 32000:
                    raise ValueError(f"Stamp capture failed (ok={ok}, samples={samples})")
                pcm = bytearray(2 * samples)
                offset = 0
                while offset < samples:
                    f = self._frame(20.0)
                    if f.type != 0x4A or len(f.payload) < 4:
                        raise ValueError("unexpected PCM frame")
                    got_offset = int.from_bytes(f.payload[0:2], "little")
                    count = int.from_bytes(f.payload[2:4], "little")
                    raw = f.payload[4:]
                    if got_offset != offset or count == 0 or len(raw) != 2 * count or offset + count > samples:
                        raise ValueError("invalid PCM chunk")
                    pcm[2 * offset:2 * (offset + count)] = raw
                    offset += count
                return np.frombuffer(pcm, "<i2").copy()
            except (TimeoutError, ValueError) as e:
                if attempt + 1 == retries:
                    raise RuntimeError(f"Solist/Stamp measurement failed: {e}") from e
                print(f"    {e} -> retry", flush=True)
                time.sleep(1.0)


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--states", nargs="+", type=int, choices=range(32), help="Capture selected numeric state masks only")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "captures" / f"stamp_{dt.date.today():%Y%m%d}_{args.name}_wav"
    if out.exists():
        raise SystemExit(f"already exists: {out}")
    out.mkdir(parents=True)
    solist = Solist(args.solist)
    order = [(p ^ (p >> 1)) for p in range(32)]
    order = order[args.rotate:] + order[:args.rotate]
    if args.reverse:
        order = order[::-1]
    if args.states:
        order = [c for c in order if c in args.states]
    metas: dict[int, dict] = {}
    t0 = time.time()
    completed = 0
    total = args.baseline + len(order) * args.frames

    def grab(state, group):
        nonlocal completed
        print(f"[{args.name}] {completed}/{total}, remaining {total-completed}: {dirname(state)} recording/transfer", flush=True)
        d = out / dirname(state)
        d.mkdir(exist_ok=True)
        m = metas.setdefault(state, dict(label=dirname(state), source="Stamp-S3A via Solist I2C/UART relay",
                                         quantize="int16", capture="Stamp-S3A I2S 48k -> FIR 16k, x16 scale",
                                         frames=[]))
        x = solist.measure()
        name = f"frame_{len(m['frames']):06d}.wav"
        write_wav(d / name, x)
        m["frames"].append(dict(wav=name, group=group, time=dt.datetime.now().isoformat(timespec="seconds"),
                                clipped_samples=int(np.sum(np.abs(x) >= 32767))))
        (d / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        completed += 1
        elapsed = time.time() - t0
        print(f"[{args.name}] {completed}/{total}, remaining {total-completed}, elapsed {elapsed/60:.1f} min, ETA {elapsed/completed*(total-completed)/60:.1f} min", flush=True)
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

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
