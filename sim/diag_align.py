"""時間領域 baseline 差分の前提 (= 時間整列) を診断する。

ユーザー方針: 「音声をそのまま FFT ではなく、baseline との diff を FFT して特徴抽出」。
これは time-domain で x - baseline を取り、その後 FFT する解釈。成立条件は
「同一 PRBS 励振がサンプル単位で整列している」こと。整列していれば baseline 応答が
コヒーレントに相殺し、状態変化由来の差だけが残る。整列していなければ相殺せず、
差分は単なるノイズ増幅になる → その場合は (a) cross-correlation で整列、または
(b) spectral-domain diff (v1 noise_diff) にフォールバック。

ここでは run v12 の s00000 (全閉) と s11111 (全開) で、
  R_self  = ||x_s00000 - baseline||² / ||x_s00000||²   (小さいほどコヒーレント相殺OK)
  R_other = ||x_s11111 - baseline||² / ||x_s11111||²   (R_self より十分大きいほど差分が信号)
を、整列なし / best-lag 整列ありの両方で測る。
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

CAP = Path(r"D:/GitHub/IchiPing/pc/captures/full_32_train_v12")


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def frames(state: str, n=None):
    fs = sorted((CAP / state).glob("frame_*.wav"))
    if n:
        fs = fs[:n]
    return [load_wav(p) for p in fs]


def best_lag_align(x: np.ndarray, ref: np.ndarray, max_lag=64) -> np.ndarray:
    """ref に対し x を ±max_lag の範囲で相互相関最大の lag だけシフト整列。"""
    n = min(len(x), len(ref))
    x, ref = x[:n], ref[:n]
    # 粗く相互相関 (中央付近のみ)
    best, blag = -1e18, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[lag:], ref[:n - lag]
        else:
            a, b = x[:n + lag], ref[-lag:]
        c = float(np.dot(a, b))
        if c > best:
            best, blag = c, lag
    if blag >= 0:
        return np.concatenate([x[blag:], np.zeros(blag)])
    return np.concatenate([np.zeros(-blag), x[:n + blag]])


def resid(xs, baseline, align=False):
    rs = []
    for x in xs:
        n = min(len(x), len(baseline))
        xx = x[:n]
        bb = baseline[:n]
        if align:
            xx = best_lag_align(xx, bb)[:n]
        rs.append(np.sum((xx - bb) ** 2) / (np.sum(xx ** 2) + 1e-12))
    return np.array(rs)


def norm_xcorr_peak(a, b, max_lag=256):
    """正規化相互相関のピーク値と lag。1.0 に近いほど同一波形が整列可能。"""
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    best, blag = -1e18, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:]
        c = float(np.dot(x, y))
        if c > best:
            best, blag = c, lag
    return best / (na * nb + 1e-12), blag


def main():
    s0 = frames("s00000")        # baseline state
    s1 = frames("s11111")        # all-open
    L = len(s0[0])
    print(f"s00000 frames={len(s0)}, len={L}; s11111 frames={len(s1)}")

    # DC と AC の内訳 (DC が L2 を支配していないかの確認)
    x = s0[0]
    dc = x.mean()
    ac = x - dc
    print(f"\n  frame0: mean(DC)={dc:.5f}  RMS_total={np.sqrt((x**2).mean()):.5f}  "
          f"RMS_AC={np.sqrt((ac**2).mean()):.5f}  DC_energy_frac={dc**2/(x**2).mean():.4f}")

    # PRBS が同一系列で整列再生されているか: 正規化相互相関ピーク
    p01, lag01 = norm_xcorr_peak(s0[0], s0[1])
    p0o, lago = norm_xcorr_peak(s0[0], s1[0])
    print(f"  正規化相互相関ピーク: s00000(f0 vs f1)={p01:.3f} @lag {lag01}   "
          f"s00000 vs s11111={p0o:.3f} @lag {lago}")
    print("    (f0 vs f1 が ~1 なら同一PRBS整列再生 → time-domain diff 可。"
          "低いなら毎ショット別系列 → spectral diff 必須)")

    # ---- AC 成分 (DC 除去) でコヒーレンス評価 ----
    def ac_of(xs):
        return [(v[:L] - v[:L].mean()) for v in xs]
    s0a, s1a = ac_of(s0), ac_of(s1)
    half = len(s0a) // 2
    base_ac = np.mean(np.stack(s0a[:half]), axis=0)

    print("\n  === AC 成分での time-domain baseline 差分 ===")
    for align in (False, True):
        tag = "best-lag整列" if align else "整列なし"
        r_self = resid(s0a[half:], base_ac, align)
        r_other = resid(s1a, base_ac, align)
        sep = r_other.mean() / max(r_self.mean(), 1e-9)
        print(f"  [{tag}] R_self={r_self.mean():.3f}  R_other={r_other.mean():.3f}  "
              f"分離比={sep:.2f}")
    print("    (R_self<<1 かつ 分離比>>1 なら AC でもコヒーレント相殺成立)")


if __name__ == "__main__":
    main()
