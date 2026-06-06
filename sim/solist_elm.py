"""Solist-AI 互換 ELM の純 numpy 参照実装。

ROHM アプリノート (CTD_MBDG_68_001) の構成に忠実:
  - 3 層 FFNN・隠れ層 1 層
  - α (入力→隠れ) は乱数固定、**更新しない**
  - β (隠れ→出力) のみ最小二乗で学習 (= AxlCORE-ODL がオンデバイスで担う部分)
  - 教師あり分類: ターゲット = one-hot、推論は argmax(h·β)
  - 教師なし (異常検知, 参考実装): ターゲット = 入力、異常度 = Σ|x−y|²

実機との対応:
  - 実機の α はチップ/ライブラリが持つ乱数 (こちらでは選べない)。sim では seed を
    変えた複数 α で分散を見ることで「特定 α への過適合でない」ことを担保する。
  - β は (HᵀH+λI)⁻¹HᵀY。λ はリッジ項 (Sim の設定項目に相当)。
  - 入力次元 D ≤ 512 (1 モデル)。D≤64 なら最大 4 モデル → アンサンブル可。
"""
from __future__ import annotations

import numpy as np


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    # 実機 AxlCORE-ODL の活性化 (AnomalyDetection AN 3.1.2):
    if name == "hard_sigmoid":          # ODL_ACTV_SIGMOID: max(0,min(1,0.2x+0.5))
        return np.clip(0.2 * z + 0.5, 0.0, 1.0)
    if name == "hard_tanh":             # ODL_ACTV_TANH: max(-1,min(1,x))
        return np.clip(z, -1.0, 1.0)
    if name == "linear":                # ODL_ACTV_LINEAR
        return z
    raise ValueError(name)


class SolistELM:
    def __init__(self, n_hidden: int = 64, activation: str = "sigmoid",
                 ridge: float = 1e-2, seed: int = 0,
                 alpha_scale: float | None = None):
        self.K = n_hidden
        self.activation = activation
        self.ridge = ridge
        self.seed = seed
        self.alpha_scale = alpha_scale
        self.alpha = None      # (D, K) 乱数固定
        self.bias = None       # (K,)
        self.beta = None       # (K, M)
        self.n_cls = None

    def _project(self, X: np.ndarray) -> np.ndarray:
        return _act(self.activation, X @ self.alpha + self.bias)

    def _init_alpha(self, D: int):
        rng = np.random.default_rng(self.seed)
        # 入力は per-frame 正規化済 (std≈1)。z=Xα の分散を ~1 に保つ初期化。
        scale = self.alpha_scale if self.alpha_scale else 1.0 / np.sqrt(D)
        self.alpha = (rng.standard_normal((D, self.K)) * scale).astype(np.float64)
        self.bias = (rng.standard_normal(self.K) * 0.1).astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray, n_cls: int,
            class_weight: bool = False):
        """教師あり学習。y は整数ラベル (0..n_cls-1)。"""
        self.n_cls = n_cls
        self._init_alpha(X.shape[1])
        H = self._project(X)                       # (N, K)
        Y = np.zeros((len(y), n_cls), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0
        if class_weight:
            # クラス不均衡対策 (appnote の重み付き擬似逆行列に相当)
            freq = np.bincount(y, minlength=n_cls).astype(np.float64)
            w = (1.0 / np.maximum(freq[y], 1.0))
            w *= len(y) / w.sum()
            Hw = H * w[:, None]
            A = H.T @ Hw + self.ridge * np.eye(self.K)
            B = Hw.T @ Y
        else:
            A = H.T @ H + self.ridge * np.eye(self.K)
            B = H.T @ Y
        self.beta = np.linalg.solve(A, B)          # (K, M)
        return self

    def fit_targets(self, X: np.ndarray, Y: np.ndarray):
        """任意の連続ターゲット行列 Y (N,M) に対して β を最小二乗で解く。

        Solist-AI の「教師あり数値出力」モード一般形。one-hot 分類だけでなく
        5bit 回帰 (M=5) など出力次元を抑えた符号化に使う (出力数制限への保険)。
        """
        self.n_cls = Y.shape[1]
        self._init_alpha(X.shape[1])
        H = self._project(X)
        A = H.T @ H + self.ridge * np.eye(self.K)
        self.beta = np.linalg.solve(A, H.T @ Y)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self._project(X) @ self.beta        # (N, M)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision(X).argmax(axis=1)

    def refit_beta(self, X: np.ndarray, y: np.ndarray, class_weight: bool = False):
        """α を固定したまま β のみ再計算 (= オンデバイス再校正のシミュレーション)。"""
        return self.fit_with_fixed_alpha(X, y, self.n_cls, class_weight)

    def fit_with_fixed_alpha(self, X, y, n_cls, class_weight=False):
        # alpha/bias を保ったまま β だけ解き直す
        saved_alpha, saved_bias = self.alpha, self.bias
        D = X.shape[1]
        if saved_alpha is None or saved_alpha.shape[0] != D:
            self._init_alpha(D)
        else:
            self.alpha, self.bias = saved_alpha, saved_bias
        H = self._project(X)
        Y = np.zeros((len(y), n_cls), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0
        A = H.T @ H + self.ridge * np.eye(self.K)
        self.beta = np.linalg.solve(A, H.T @ Y)
        self.n_cls = n_cls
        return self


# --- 評価指標 (sklearn 非依存) --------------------------------------------

def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp == 0:
            f1s.append(0.0 if (fp + fn) > 0 else np.nan)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.nanmean(f1s))
