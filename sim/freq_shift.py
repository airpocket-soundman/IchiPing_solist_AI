"""気温差による周波数シフト対策 (frequency shift) の Solist-AI パイプライン向け実装。

背景 (IchiPing-UNO-Q 2026-09-12): 空調で室温が下がると音速が約0.18 %/°C 変わり、
模型内の共鳴周波数が一様にスケールする。夕方の全閉スペクトルは午後に対して -2.15 % の
周波数シフトで説明でき、UNO Q では学習時に log-PSD を ±3 % シフトする augmentation
(pc/training/dataset.py warp_logmag_psd) で夕方評価が +12 pt 改善した。

Solist-AI 向け特徴 (sim/bench_v612.py) は baseline を **時間波形** で引いてから FFT する
ため、dB スペクトルのシフトをそのまま流用できない。ここでは 2 方式を提供する。

1. IR shift (推奨, 励振 PRBS が既知のデータ用)
   録音 a = prbs ⊛ h と見なし、既知 PRBS で正則化逆畳み込みしてインパルス応答 h を推定、
   h(t) → h(t·(1+ε)) と時間伸縮 (= 周波数 (1+ε) 倍。音速変化そのもの) して再合成する。
   a_w = a + prbs ⊛ (h_w − h) とすることで観測雑音と逆畳み込み誤差を温存する。
   baseline 側はシフトしないので「気温が変わった後に古い baseline で推論する」失敗モードが
   そのまま再現される (UNO Q 方式と同じ考え方)。
2. feature shift (励振が再現不能なデータ用: IchiPing FRDM 世代は xorshift seed 非再現)
   diff スペクトル (dB, 512 bin) を周波数軸で伸縮する。stale-baseline 項は再現されない。

用語: 本リポジトリの「周波数シフト」は周波数軸の (1+ε) 倍スケーリング (比例シフト) を指す。
一定 Hz を足す加算シフトではない (1 kHz で 10 Hz なら 2 kHz で 20 Hz 動く)。対数周波数軸では平行移動になる。
旧称「ワープ」(freq_warp.py, --warp 等) は 2026-09-25 に改名した。

符号: ε > 0 で共鳴が高域へ動く (UNO Q warp_logmag_psd と同じ)。
"""
from __future__ import annotations

import random

import numpy as np

FS = 16_000
FRAME = 32_000          # 2.0 s
NF = 65_536             # 逆畳み込み用 FFT 長 (>= FRAME + IR 長)


def prbs16k(seed: int = 20260912, n: int = FRAME) -> np.ndarray:
    """UNO Q collector の励振 (uno_q/audio/audio-smoke-test.py prbs_source と同一)。"""
    rng = random.Random(seed)
    return np.array([1.0 if rng.getrandbits(1) else -1.0 for _ in range(n)])


class IRShifter:
    """既知励振に対する IR 推定と時間伸縮。

    pre : 逆畳み込み IR の負時間側に残す sample 数 (帯域制限 sinc の pre-ringing 用)
    L   : 正時間側の IR 長 (UNO Q 模型では 10 ms 以内に 97 %, 50 ms で 99.9 %)
    lam : Tikhonov 正則化 (|P|² 平均に対する比)
    """

    def __init__(self, excitation: np.ndarray, pre: int = 256, L: int = 2048, lam: float = 1e-3):
        assert len(excitation) == FRAME
        self.p = excitation.astype(np.float64)
        self.pre, self.L, self.lam = pre, L, lam
        self.P = np.fft.rfft(self.p, NF)
        self._den = np.abs(self.P) ** 2 + lam * np.mean(np.abs(self.P) ** 2)
        self._wiener = np.conj(self.P) / self._den

    def impulse_response(self, a: np.ndarray) -> np.ndarray:
        """t ∈ [-pre, L) の IR (index pre が t=0)。"""
        A = np.fft.rfft(a[:FRAME], NF)
        hf = np.fft.irfft(A * self._wiener, NF)
        return np.concatenate([hf[-self.pre:], hf[:self.L]])

    def reconvolve(self, h: np.ndarray) -> np.ndarray:
        y = np.fft.irfft(self.P * np.fft.rfft(h, NF), NF)
        return np.roll(y, -self.pre)[:FRAME]

    def shift_ir(self, h: np.ndarray, eps: float) -> np.ndarray:
        t = np.arange(len(h), dtype=np.float64) - self.pre
        return np.interp(t * (1.0 + eps), t, h, left=0.0, right=0.0)

    def shift_audio(self, a: np.ndarray, eps: float) -> np.ndarray:
        """周波数を (1+ε) 倍にスケールした録音を返す (ε=0 で恒等)。"""
        if eps == 0.0:
            return a.copy()
        h = self.impulse_response(a)
        return a + self.reconvolve(self.shift_ir(h, eps) - h)

    def reconstruction_snr_db(self, a: np.ndarray) -> float:
        r = self.reconvolve(self.impulse_response(a))
        return float(10 * np.log10(np.sum(a ** 2) / np.sum((a - r) ** 2)))


def shift_db_spectrum(db: np.ndarray, eps: float) -> np.ndarray:
    """dB スペクトル (bin k = (k+1)·fs/NFFT, DC 除外済) を周波数軸で (1+ε) 倍にスケール。
    UNO Q の warp_logmag_psd と同じ規約 (端は保持)。"""
    n = db.shape[-1]
    bins = np.arange(1, n + 1, dtype=np.float64)
    return np.interp(bins / (1.0 + eps), bins, db).astype(db.dtype)


def fit_shift(ref_db: np.ndarray, target_db: np.ndarray, lo: int, hi: int,
             grid=np.linspace(-0.06, 0.06, 241)) -> float:
    """target ≈ shift(ref, ε) となる ε を格子探索 (bin 範囲 [lo,hi) の相関最大)。"""
    best, best_c = 0.0, -np.inf
    for e in grid:
        w = shift_db_spectrum(ref_db, float(e))[lo:hi]
        c = np.corrcoef(w - w.mean(), target_db[lo:hi] - target_db[lo:hi].mean())[0, 1]
        if c > best_c:
            best, best_c = float(e), c
    return best
