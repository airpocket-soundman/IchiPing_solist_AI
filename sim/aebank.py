"""異常検知のみで分類する — クラス別オートエンコーダ・バンク (argmin 再構成誤差)。

前提: Solist-AI が教師なし異常検知 (オートエンコーダ) しか使えない場合でも分類したい。
方式:
  - クラス c ごとに AE_c を学習 (target = 入力, β_c を最小二乗)。AE_c は c のデータを
    よく再構成し、他クラスでは誤差が大きい。
  - 推論: 全 AE_c の再構成誤差(=異常度) を出し argmin_c を予測クラスとする。
Solist-AI 親和性:
  - α (ランダム射影 = AxlCORE 固定部) は全クラス共有。β_c だけクラス別。
  - 推論は β を差し替えて 14 回スコアリング → argmin。4モデル同時枠の制約は逐次差替で回避。
  - これは「異常度 L = Σ|x − y|²」(appnote) をクラス別に並べたもの。

cross-run (末尾3run hold-out)、診断済 diff_fft 特徴、帯域 400-3000Hz、標準化。
"""
from __future__ import annotations

import numpy as np

import feats_diff
import common as C
from solist_elm import _act, accuracy


class Std:
    def fit(self, X): self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; return self
    def __call__(self, X): return (X - self.mu) / self.sd


def crop(X, lo_hz, hi_hz):
    b = feats_diff.BIN_HZ_DIFF
    lo = max(0, int(round(lo_hz / b)) - 1); hi = min(X.shape[1], int(round(hi_hz / b)))
    return X[:, lo:hi], hi - lo


class AEBank:
    """共有 α + クラス別 β のオートエンコーダ・バンク。"""
    def __init__(self, n_hidden=48, activation="sigmoid", ridge=1e-2, seed=0):
        self.K = n_hidden; self.act = activation; self.ridge = ridge; self.seed = seed
        self.alpha = None; self.bias = None; self.betas = {}

    def _proj(self, X):
        return _act(self.act, X @ self.alpha + self.bias)

    def fit(self, X, y, classes):
        rng = np.random.default_rng(self.seed)
        D = X.shape[1]
        self.alpha = rng.standard_normal((D, self.K)) / np.sqrt(D)
        self.bias = rng.standard_normal(self.K) * 0.1
        for c in classes:
            Xc = X[y == c]
            H = self._proj(Xc)                       # (Nc, K)
            A = H.T @ H + self.ridge * np.eye(self.K)
            self.betas[c] = np.linalg.solve(A, H.T @ Xc)   # (K, D) 再構成
        self.classes = list(classes)
        return self

    def scores(self, X):
        """各クラス AE の再構成誤差 (N, n_cls)。小さいほどそのクラスらしい。"""
        H = self._proj(X)
        S = np.empty((len(X), len(self.classes)))
        for j, c in enumerate(self.classes):
            recon = H @ self.betas[c]
            S[:, j] = ((X - recon) ** 2).mean(axis=1)
        return S

    def predict(self, X):
        S = self.scores(X)
        return np.array(self.classes)[S.argmin(axis=1)]


def vote_eval(score_neg, y, rid, y32, classes, k):
    """soft-vote: 同一(run,状態)の k フレームでスコア合計 → argmin。"""
    classes = np.array(classes)
    correct = tot = 0
    for r in np.unique(rid):
        for s in np.unique(y32[rid == r]):
            idx = np.where((rid == r) & (y32 == s))[0]
            S = score_neg[idx]; Y = y[idx]
            groups = ([np.arange(len(idx))] if k <= 0 else
                      [np.arange(i, min(i+k, len(idx))) for i in range(0, len(idx), k)])
            for g in groups:
                pred = classes[S[g].sum(0).argmin()]
                correct += int(pred == Y[g][0]); tot += 1
    return correct / tot


def main():
    X, y14, y32, rid = feats_diff.load_runs(C.DEFAULT_RUNS)
    n = rid.max() + 1; te = rid >= n - 3; tr = ~te
    Xr, D = crop(X, 400, 3000)
    st = Std().fit(Xr[tr]); Xtr, Xte = st(Xr[tr]), st(Xr[te])
    classes = list(range(14))
    print(f"AE-bank classification (異常検知のみ) D={D} cross-run test=last3runs")
    print(f"  {'K':>4} {'ridge':>7}  {'frame':>7} {'vote5':>7} {'vote-all':>8}")
    for K in (24, 48, 96):
        for ridge in (1e-2, 1e-1):
            bank = AEBank(n_hidden=K, ridge=ridge).fit(Xtr, y14[tr], classes)
            S = bank.scores(Xte)
            yte, r_te, s_te = y14[te], rid[te], y32[te]
            a1 = accuracy(yte, np.array(classes)[S.argmin(1)])
            a5 = vote_eval(S, yte, r_te, s_te, classes, 5)
            aall = vote_eval(S, yte, r_te, s_te, classes, 0)
            star = "  ★100%" if aall >= 0.999 else ""
            print(f"  {K:>4} {ridge:>7.0e}  {a1:>6.1%} {a5:>6.1%} {aall:>7.1%}{star}")


if __name__ == "__main__":
    main()
