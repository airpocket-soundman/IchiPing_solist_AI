"""PlatformIO pre-script: data/prbs48k.bin = the UNO Q PRBS excitation at 48 kHz, data/prbs_chips.bin = its 16 kHz ±1 chips.

Same computation as IchiPing-UNO-Q uno_q/audio/audio-smoke-test.py prbs16k_data():
±1 chips from random.Random(20260912).getrandbits(1) at 16 kHz (2.0 s), zero-stuffed x3,
72-tap Hann windowed-sinc (7.6 kHz, gain 3), filter delay removed, x amplitude x (2^23-1),
rounded, << 8 (S32 word).  Only the 96000 active samples are stored (int32 LE); the
firmware adds the 0.5 s lead and 1.0 s tail silence.
"""
import math
import random
import struct
from pathlib import Path

SEED, CHIPS, RATE, AMPLITUDE, TAPS_PER_PHASE = 20260912, 32000, 48000, 0.0088, 24


def prbs48k_words(amplitude=AMPLITUDE, seed=SEED):
    rng = random.Random(seed)
    source = [1 if rng.getrandbits(1) else -1 for _ in range(CHIPS)]
    length = 3 * TAPS_PER_PHASE
    centre = (length - 1) / 2
    cutoff = 7_600 / RATE
    kernel = []
    for index in range(length):
        x = index - centre
        sinc = 2 * cutoff if x == 0 else math.sin(2 * math.pi * cutoff * x) / (math.pi * x)
        window = 0.5 - 0.5 * math.cos(2 * math.pi * index / (length - 1))
        kernel.append(3 * sinc * window)
    up = [0.0] * (len(source) * 3 + len(kernel) - 1)
    for index, value in enumerate(source):
        base = index * 3
        for tap, weight in enumerate(kernel):
            up[base + tap] += value * weight
    delay = (len(kernel) - 1) // 2
    active = up[delay:delay + len(source) * 3]
    peak24 = 2 ** 23 - 1
    return [round(v * amplitude * peak24) << 8 for v in active], source


def write(path: Path):
    """prbs48k.bin (96000 x int32) and prbs_chips.bin (32000 chips, bit i of byte i/8 = chip > 0)."""
    words, chips = prbs48k_words()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(words)}i", *words))
    packed = bytearray(len(chips) // 8)
    for i, c in enumerate(chips):
        if c > 0:
            packed[i // 8] |= 1 << (i % 8)
    (path.parent / "prbs_chips.bin").write_bytes(bytes(packed))


try:
    Import("env")  # noqa: F821  (PlatformIO)
    out = Path(env.subst("$PROJECT_DIR")) / "data" / "prbs48k.bin"  # noqa: F821
    if not out.exists() or not (out.parent / "prbs_chips.bin").exists():
        write(out)
except NameError:
    if __name__ == "__main__":
        write(Path(__file__).resolve().parents[1] / "data" / "prbs48k.bin")
