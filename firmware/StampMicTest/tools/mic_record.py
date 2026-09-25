"""StampMicTest ファームから 16 kHz mono を録音して wav に保存し、レベルと帯域別パワーを表示する。

usage: python firmware/StampMicTest/tools/mic_record.py --port COM13 [--seconds 2] [--out mic.wav]
"""
from __future__ import annotations

import argparse
import time
import wave

import numpy as np
import serial


def record(port: str, seconds: int) -> np.ndarray:
    s = serial.Serial(port, 115200, timeout=1.0)
    try:
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(f"R{seconds}\n".encode())
        t0 = time.time()
        while True:
            line = s.readline()
            if line.startswith(b"PCM16 "):
                n = int(line.split()[1])
                break
            if time.time() - t0 > 5:
                raise TimeoutError("PCM16 header not received")
        data = bytearray()
        while len(data) < n * 2:
            chunk = s.read(n * 2 - len(data))
            if not chunk:
                raise TimeoutError(f"received {len(data)}/{n * 2} bytes")
            data += chunk
        return np.frombuffer(bytes(data), "<i2").copy()
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM13")
    ap.add_argument("--seconds", type=int, default=2)
    ap.add_argument("--out", default="mic.wav")
    a = ap.parse_args()
    x = record(a.port, a.seconds)
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())
    xf = x.astype(np.float64) - x.mean()
    rms = np.sqrt(np.mean(xf ** 2))
    print(f"{len(x)} samples -> {a.out}  rms={20*np.log10(max(rms,1e-9)/32768):.1f} dBFS  "
          f"peak={20*np.log10(max(np.abs(x).max(),1)/32768):.1f} dBFS  dc={x.mean():.1f}")
    P = np.abs(np.fft.rfft(xf * np.hanning(len(xf)))) ** 2
    f = np.fft.rfftfreq(len(xf), 1 / 16000)
    tot = P.sum() + 1e-30
    for lo, hi in ((0, 100), (100, 400), (400, 1000), (1000, 3000), (3000, 8000)):
        m = (f >= lo) & (f < hi)
        print(f"  {lo:5d}-{hi:<5d} Hz: {100 * P[m].sum() / tot:5.1f} %")
    print(f"  peak frequency: {f[1:][np.argmax(P[1:])]:.0f} Hz")


if __name__ == "__main__":
    main()
