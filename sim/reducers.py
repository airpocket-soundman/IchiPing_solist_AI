"""入力次元リデューサ — ELM 入力 (D≤512) を作る。

ユーザー方針: 情報の無い HF/LF を捨て、残した帯域はネイティブ分解能を維持。
band_crop はそのまま連続クロップ (ネイティブ分解能保持)。512 を超える帯域を
使いたい場合のみ crop_decimate で平均プーリング。
"""
from __future__ import annotations

import numpy as np


def band_crop(X: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """bins [lo, hi) を切り出す (ネイティブ分解能)。hi-lo ≤ 512 を呼び側で担保。"""
    return X[:, lo:hi]


def crop_decimate(X: np.ndarray, lo: int, hi: int, target_D: int) -> np.ndarray:
    """bins [lo,hi) を切り出し target_D へ平均プーリング (帯域 > 512 のとき用)。"""
    Xc = X[:, lo:hi]
    w = Xc.shape[1]
    if w <= target_D:
        return Xc
    edges = np.linspace(0, w, target_D + 1).astype(int)
    return np.stack([Xc[:, edges[i]:edges[i+1]].mean(axis=1)
                     for i in range(target_D)], axis=1)
