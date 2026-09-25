"""samples/uno_q_eval_evening の wav を Stamp に埋め込む clips.bin にまとめる (PlatformIO の pre スクリプト)。

clips.bin: "ICLP" | version u8 | clips u8 | baseline u8 | 0 | samples u16 | 0 u16 |
           class u8 x clips (0xFF = baseline) | 4 byte 境界まで 0 | PCM16 LE (clip 順)
clip 順 = baseline (manifest 順) → 状態 (class 0..31)。

単体実行: python firmware/StampPipelineTest/stamp/tools/make_clips.py
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

try:
    Import("env")  # noqa: F821  (PlatformIO extra_scripts)
    PROJECT = Path(env["PROJECT_DIR"])  # noqa: F821
except NameError:
    PROJECT = Path(__file__).resolve().parent.parent

ROOT = PROJECT.parents[2]
SAMPLES = ROOT / "samples" / "uno_q_eval_evening"
OUT = PROJECT / "data" / "clips.bin"
VERSION = 1


def load(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def build() -> None:
    man = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    clips = [(0xFF, SAMPLES / b["file"]) for b in man["baseline"]]
    clips += [(s["class_id"], SAMPLES / s["file"]) for s in sorted(man["states"], key=lambda s: s["class_id"])]
    pcm = [load(p) for _, p in clips]
    samples = len(pcm[0]) // 2
    assert all(len(p) == samples * 2 for p in pcm)
    head = b"ICLP" + struct.pack("<BBBBHH", VERSION, len(clips), len(man["baseline"]), 0, samples, 0)
    head += bytes(c for c, _ in clips)
    head += b"\0" * (-len(head) % 4)
    data = head + b"".join(pcm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists() or OUT.read_bytes() != data:
        OUT.write_bytes(data)
    print(f"clips.bin: {len(clips)} clips ({len(man['baseline'])} baseline) x {samples} samples, {len(data)} bytes")


build()
