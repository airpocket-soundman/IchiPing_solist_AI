"""情報帯域の特定 — クラス判別に効く周波数 bin を data から決める。

ユーザー方針: 「情報を含まない高周波・低周波を捨て、残した帯域はネイティブ
FFT 分解能のまま ≤512 点に収める」。一様間引きより周波数分解能を保てる。

判別力の指標として bin ごとの **Fisher 比** (クラス間分散 / クラス内分散) を
14cls ラベルで計算する。F 比が高い bin = クラスで系統的に動く = 情報あり。
累積寄与で「情報の N% を含む最小連続帯域」を求め、それが ≤512 bin なら
ネイティブ分解能のまま採用できる。
"""
from __future__ import annotations

import numpy as np

from common import (
    DEFAULT_RUNS, N_CLS_14, N_BINS, bin_to_hz, hz_to_bin, load_runs,
)


def fisher_ratio(X: np.ndarray, y: np.ndarray, n_cls: int) -> np.ndarray:
    """各 bin の Fisher 判別比 (between-class var / within-class var)。"""
    nb = X.shape[1]
    gmean = X.mean(axis=0)
    between = np.zeros(nb, dtype=np.float64)
    within = np.zeros(nb, dtype=np.float64)
    for c in range(n_cls):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        between += len(Xc) * (mu - gmean) ** 2
        within += ((Xc - mu) ** 2).sum(axis=0)
    return (between / np.maximum(within, 1e-12)).astype(np.float64)


def informative_band(F: np.ndarray, keep: float = 0.95, max_bins: int = 512):
    """F 比の連続窓のうち、累積 F の keep 割合を含む最小連続帯域 [lo, hi)。

    連続帯域に限定する理由: 実機では「FFT 出力の連続スライス」を ELM 入力に
    したいので、飛び地 bin 選択より連続クロップが実装に素直。
    """
    nb = len(F)
    total = F.sum()
    target = keep * total
    best = None  # (width, lo, hi, captured_fraction)
    csum = np.concatenate([[0.0], np.cumsum(F)])  # csum[j]-csum[i] = sum F[i:j]
    for lo in range(nb):
        # 最小幅でtargetに届くhiを二分的に前進 (単調なので線形でも軽い)
        hi = lo
        # まず target を満たす最小 hi を探す
        # csum[hi]-csum[lo] >= target
        need = target + csum[lo]
        hi = int(np.searchsorted(csum, need))
        if hi > nb:
            break
        width = hi - lo
        frac = (csum[hi] - csum[lo]) / total
        if best is None or width < best[0]:
            best = (width, lo, hi, frac)
    return best  # (width, lo, hi, captured_fraction)


def main():
    X, y14, y32, rid = load_runs(DEFAULT_RUNS)
    print(f"loaded X={X.shape} from {rid.max()+1} runs")

    F14 = fisher_ratio(X, y14, N_CLS_14)
    F32 = fisher_ratio(X, y32, 32)

    # 帯域候補を keep 別に表示 (14cls 基準)
    print("\n=== 14cls Fisher 比による情報帯域 (連続クロップ) ===")
    for keep in (0.90, 0.95, 0.99):
        w, lo, hi, frac = informative_band(F14, keep=keep)
        print(f"  keep≥{keep:.0%}: bins[{lo:4d}:{hi:4d}]  width={w:4d}  "
              f"({bin_to_hz(lo):6.0f}–{bin_to_hz(hi-1):6.0f} Hz)  "
              f"captured={frac:.1%}  fits512={'YES' if w<=512 else 'no'}")

    # F 比のピーク帯と端の様子
    topk = np.argsort(F14)[::-1][:10]
    print("\n  top-10 判別 bin (14cls):")
    for k in sorted(topk):
        print(f"    bin{k:4d} ({bin_to_hz(k):6.0f} Hz)  F14={F14[k]:7.2f}  F32={F32[k]:7.2f}")

    # 帯域端のエネルギー寄与プロファイル (32分割の粗いヒストグラム)
    print("\n  F14 を 32 区間で集計 (情報の集中度):")
    seg = N_BINS // 32
    for s in range(32):
        a, b = s * seg, (s + 1) * seg
        frac = F14[a:b].sum() / F14.sum()
        bar = "#" * int(frac * 200)
        print(f"    {bin_to_hz(a):5.0f}-{bin_to_hz(b-1):5.0f}Hz |{bar} {frac:.1%}")

    # 推奨帯域を保存
    w, lo, hi, frac = informative_band(F14, keep=0.95)
    np.savez(Path(__file__).resolve().parent / "_cache" / "band.npz",
             F14=F14, F32=F32, lo=lo, hi=hi, keep=0.95)
    print(f"\n推奨帯域(keep95%): bins[{lo}:{hi}] width={hi-lo} "
          f"({bin_to_hz(lo):.0f}-{bin_to_hz(hi-1):.0f} Hz) を band.npz に保存")


if __name__ == "__main__":
    from pathlib import Path
    main()
