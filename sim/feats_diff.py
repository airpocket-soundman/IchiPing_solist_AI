"""特徴量 = FFT( audio − baseline )  (ユーザー方針 / time-domain baseline 差分)。

diag_align.py で実証済み: PRBS は同一系列がサンプル整列再生されており、
time-domain で baseline(s00000 平均波形) を引くと共通室内応答が 0.3% まで
コヒーレント相殺し、状態変化由来の差だけが残る (分離比 194×)。その差を FFT した
マグニチュードスペクトルを ELM 入力にする。位相情報も差に効くため spectral
magnitude diff (v1 noise_diff) より判別力が高いことを期待。

実機対応: baseline は設置時に一度取得して保持 (Solist-AI の現場学習思想)。励振は
同一MCU・1タイマで DAC送出/ADC取込を駆動しサンプル整列を担保する前提。

出力: nfft=1024 → rFFT 513 → DC 落として 512 bin の log-magnitude。
cross-run 評価では各 run 自身の s00000 を baseline に使う (環境ごとに baseline)。
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from common import (
    CAPTURES_ROOT, CACHE_DIR, DEFAULT_RUNS, N_CLS_14,
    parse_state_label, class_idx_14,
)

NFFT = 1024                 # → 512 bin (DC 落とし後)。AxlCORE の on-chip FFT サイズ想定
HOP = NFFT // 2
BIN_HZ_DIFF = 16000.0 / NFFT   # = 15.625 Hz/bin
DB_FLOOR = -80.0


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def diff_spectrum(x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """diff = x - baseline を Hann/50%overlap で STFT し |FFT| を平均 → 512-bin log-mag。"""
    n = min(len(x), len(baseline))
    d = x[:n] - baseline[:n]
    win = np.hanning(NFFT)
    segs = []
    for start in range(0, n - NFFT + 1, HOP):
        seg = d[start:start + NFFT] * win
        segs.append(np.abs(np.fft.rfft(seg)))
    mag = np.mean(segs, axis=0)            # (513,)
    mag = mag[1:]                          # DC 落とし → 512
    db = 20.0 * np.log10(mag + 1e-9)
    return np.maximum(db, DB_FLOOR).astype(np.float32)


def build_run_cache(run: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{run}__diff_fft.npz"
    if out.exists():
        return out
    run_dir = CAPTURES_ROOT / run
    base_wavs = sorted((run_dir / "s00000").glob("frame_*.wav"))
    if not base_wavs:
        raise FileNotFoundError(f"{run_dir}/s00000 frames not found")
    L = len(load_wav(base_wavs[0]))
    baseline = np.mean(np.stack([load_wav(p)[:L] for p in base_wavs]), axis=0)

    X, y14, y32 = [], [], []
    for state_dir in sorted(run_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        bits = parse_state_label(state_dir.name)
        if bits is None:
            continue
        c14 = class_idx_14(bits)
        c32 = int(bits[0] + bits[1]*2 + bits[2]*4 + bits[3]*8 + bits[4]*16)
        for wav in sorted(state_dir.glob("frame_*.wav")):
            X.append(diff_spectrum(load_wav(wav), baseline))
            y14.append(c14)
            y32.append(c32)
    np.savez_compressed(out, X=np.stack(X), y14=np.array(y14), y32=np.array(y32))
    return out


def load_runs(runs):
    Xs, y14s, y32s, rids = [], [], [], []
    for ri, run in enumerate(runs):
        npz = np.load(build_run_cache(run))
        Xs.append(npz["X"]); y14s.append(npz["y14"]); y32s.append(npz["y32"])
        rids.append(np.full(len(npz["y14"]), ri, dtype=np.int64))
    return (np.concatenate(Xs), np.concatenate(y14s),
            np.concatenate(y32s), np.concatenate(rids))


def bin_to_hz(k: int) -> float:
    return (k + 1) * BIN_HZ_DIFF


if __name__ == "__main__":
    runs = sys.argv[1:] or DEFAULT_RUNS
    X, y14, y32, rid = load_runs(runs)
    print(f"diff_fft 特徴: X={X.shape}  runs={rid.max()+1}  "
          f"mean={X.mean():.2f} std={X.std():.2f}")
    # Fisher 比で情報帯域を確認 (この特徴版)
    from band_analysis import fisher_ratio, informative_band
    F14 = fisher_ratio(X, y14, N_CLS_14)
    seg = X.shape[1] // 32
    print("\n  F14 を 32 区間で集計 (diff_fft の情報集中度):")
    for s in range(32):
        a, b = s*seg, (s+1)*seg
        frac = F14[a:b].sum()/F14.sum()
        print(f"    {bin_to_hz(a):5.0f}-{bin_to_hz(b-1):5.0f}Hz |{'#'*int(frac*200)} {frac:.1%}")
    for keep in (0.90, 0.95, 0.99):
        w, lo, hi, frac = informative_band(F14, keep=keep)
        print(f"  keep≥{keep:.0%}: bins[{lo}:{hi}] width={w} "
              f"({bin_to_hz(lo):.0f}-{bin_to_hz(hi-1):.0f}Hz) fits512={'YES' if w<=512 else 'no'}")
