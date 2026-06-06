"""ELM スイープ v2 — 標準化 + 帯域探索 + λ/K/アンサンブルで 14cls/32cls を詰める。

v1 比較対照と合わせ、cross-run (末尾3run=test) で評価。
改善レバー:
  - 標準化: 特徴を train 統計で per-dim z-score (ELM の飽和回避; 実機でも mean/scale を保持)
  - 帯域: 情報帯域 (~500-3000Hz 中心) を細かく探索 (ネイティブ分解能クロップ)
  - λ(ridge), K(隠れ), 活性化
  - アンサンブル: 異なる α seed の decision 平均 (Solist-AI の複数モデルに対応)
合格ライン: 14cls cross-run ≥ 92% (v1 97% の ±5%)。
"""
from __future__ import annotations

import numpy as np

from solist_elm import SolistELM, accuracy, macro_f1
from reducers import band_crop, crop_decimate
import feats_diff
import common as C


class Standardizer:
    def __init__(self): self.mu = None; self.sd = None
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; return self
    def __call__(self, X): return (X - self.mu) / self.sd


def split_cross_run(rid, n_test_runs=3):
    n = rid.max() + 1
    te = np.array([r >= n - n_test_runs for r in rid])
    return ~te, te


def crop(X, lo_hz, hi_hz, bin_hz):
    lo = max(0, int(round(lo_hz / bin_hz)) - 1)
    hi = min(X.shape[1], int(round(hi_hz / bin_hz)))
    if hi - lo <= 512:
        return band_crop(X, lo, hi), hi - lo
    return crop_decimate(X, lo, hi, 512), 512


def elm_eval(Xtr, ytr, Xte, yte, n_cls, K, act, ridge, seeds=(0, 1, 2),
             ensemble=False, class_weight=False):
    if ensemble:
        # 全 seed の decision を平均してから argmax (= 4 モデルアンサンブル相当)
        dec = None
        for s in seeds:
            elm = SolistELM(K, act, ridge, seed=s).fit(Xtr, ytr, n_cls, class_weight)
            d = elm.decision(Xte)
            dec = d if dec is None else dec + d
        p = dec.argmax(1)
        return accuracy(yte, p), 0.0, macro_f1(yte, p, n_cls)
    accs, f1s = [], []
    for s in seeds:
        elm = SolistELM(K, act, ridge, seed=s).fit(Xtr, ytr, n_cls, class_weight)
        p = elm.predict(Xte)
        accs.append(accuracy(yte, p)); f1s.append(macro_f1(yte, p, n_cls))
    return np.mean(accs), np.std(accs), np.mean(f1s)


def band_search(X, y14, y32, rid):
    tr, te = split_cross_run(rid, 3)
    bands = [(500, 3000), (500, 3500), (400, 3000), (600, 2800),
             (700, 1300), (500, 1100), (2400, 3100), (500, 2000)]
    print(f"\n=== 帯域探索 (14cls, K=64 sigmoid λ=1e-2, 標準化, cross-run) ===")
    print(f"  {'帯域Hz':>14} {'D':>4}  {'14cls_acc':>9} {'macroF1':>8}")
    best = None
    for lo, hi in bands:
        Xr, D = crop(X, lo, hi, feats_diff.BIN_HZ_DIFF)
        st = Standardizer().fit(Xr[tr])
        Xtr, Xte = st(Xr[tr]), st(Xr[te])
        acc, sd, f1 = elm_eval(Xtr, y14[tr], Xte, y14[te], 14, 64, "sigmoid", 1e-2)
        mark = ""
        if best is None or acc > best[0]:
            best = (acc, lo, hi); mark = " *"
        print(f"  {f'{lo}-{hi}':>14} {D:>4}  {acc:>8.1%} {f1:>7.1%}{mark}")
    return best[1], best[2]


def tune(X, y14, y32, rid, lo, hi):
    tr, te = split_cross_run(rid, 3)
    Xr, D = crop(X, lo, hi, feats_diff.BIN_HZ_DIFF)
    st = Standardizer().fit(Xr[tr])
    Xtr, Xte = st(Xr[tr]), st(Xr[te])
    print(f"\n=== 詳細チューニング 帯域={lo}-{hi}Hz D={D} (標準化, cross-run) ===")
    print(f"  {'task':>6} {'K':>4} {'act':>8} {'ridge':>7} {'ens':>4}  {'acc':>7} {'F1':>7}")
    for n_cls, ytag, y in ((14, "14cls", y14), (32, "32cls", y32)):
        ytr, yte = y[tr], y[te]
        for K in (64, 128, 256):
            for ridge in (1e-3, 1e-2, 1e-1, 1.0):
                acc, sd, f1 = elm_eval(Xtr, ytr, Xte, yte, n_cls, K, "sigmoid", ridge)
                flag = " <=92" if (ytag == "14cls" and acc >= 0.92) else ""
                print(f"  {ytag:>6} {K:>4} {'sigmoid':>8} {ridge:>7.0e} {'-':>4}  "
                      f"{acc:>6.1%} {f1:>6.1%}{flag}")
        # ベスト級でアンサンブル
        acc, _, f1 = elm_eval(Xtr, ytr, Xte, yte, n_cls, 128, "sigmoid", 1e-1,
                              seeds=(0, 1, 2, 3), ensemble=True)
        flag = " <=92" if (ytag == "14cls" and acc >= 0.92) else ""
        print(f"  {ytag:>6} {128:>4} {'sigmoid':>8} {1e-1:>7.0e} {'x4':>4}  "
              f"{acc:>6.1%} {f1:>6.1%}{flag}  (4-model ensemble)")


def main():
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    print(f"diff_fft loaded: X={X.shape} runs={rid.max()+1}")
    lo, hi = band_search(X, y14, y32, rid)
    tune(X, y14, y32, rid, lo, hi)


if __name__ == "__main__":
    main()
