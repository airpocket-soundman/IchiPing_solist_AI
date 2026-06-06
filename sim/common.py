"""IchiPing→Solist-AI 移植シミュレーション 共通基盤。

v1 リポ (../IchiPing) の features.py / dataset.py を再利用して、PRBS 雑音励振
キャプチャから v1 と完全一致の noise_diff_norm 特徴 (1024-bin log-mag) を取り出す。
ここで作るフレーム特徴キャッシュを ELM スイープが食う。

設計メモ:
- 励振方式は meta.json の pattern.name = "noise_2s_prbs"。よって特徴は noise 系。
  v1 の最良パイプラインが noise_diff_norm (baseline 差分 + per-frame 正規化) なので
  Solist-AI sim もこれを基準にする。実機 (SPI-DAC) では chirp 経路も後で比較可能。
- baseline は run ごとに s00000 平均。dataset.IchiPingDataset がこの規約を実装済み。
- cross-run 評価のため、どの run 由来かを frame ごとに保持する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- v1 リポへのパス -------------------------------------------------------
ICHIPING_ROOT = Path(r"D:/GitHub/IchiPing")
CAPTURES_ROOT = ICHIPING_ROOT / "pc" / "captures"
TRAINING_DIR = ICHIPING_ROOT / "pc" / "training"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# v1 の特徴量・ラベル定義をそのまま使う
from dataset import (  # noqa: E402
    IchiPingDataset,
    CLASS_ORDER_14,
    class_idx_14,
    parse_state_label,
)

N_BINS = 1024          # v1 特徴次元 (rFFT 2048 → DC 落として 1024)
BIN_HZ = 16000.0 / 2048.0   # ≈ 7.8125 Hz/bin。bin k (0-index) の中心 ≈ (k+1)*BIN_HZ
N_CLS_14 = 14
N_CLS_32 = 32

# 学習用キャプチャ run の既定リスト (cross-run 評価で train/test を分ける母集合)
DEFAULT_RUNS = [
    "full_32_train_v2", "full_32_train_v3", "full_32_train_v4",
    "full_32_train_v6", "full_32_train_v7", "full_32_train_v8",
    "full_32_train_v9", "full_32_train_v10", "full_32_train_v11",
    "full_32_train_v12",
]

CACHE_DIR = Path(__file__).resolve().parent / "_cache"


def bin_to_hz(k: int) -> float:
    """0-index bin → 中心周波数[Hz] (DC を落としているので bin0 ≈ BIN_HZ)。"""
    return (k + 1) * BIN_HZ


def hz_to_bin(hz: float) -> int:
    """周波数[Hz] → 最近接 0-index bin。"""
    return int(round(hz / BIN_HZ)) - 1


def build_run_cache(run: str, feature_mode: str = "noise_diff_norm") -> Path:
    """1 run 分の特徴を計算して npz に保存。既にあればスキップ。返り値は npz パス。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{run}__{feature_mode}.npz"
    if out.exists():
        return out

    run_dir = CAPTURES_ROOT / run
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    ds = IchiPingDataset(
        captures_dirs=[run_dir],
        feature_mode=feature_mode,
    )
    n = len(ds)
    X = np.empty((n, N_BINS), dtype=np.float32)
    y14 = np.empty(n, dtype=np.int64)
    y32 = np.empty(n, dtype=np.int64)
    for i in range(n):
        item = ds[i]
        X[i] = item["x"].squeeze(0).numpy()
        y14[i] = int(item["cls_idx_14"])
        y32[i] = int(item["state_idx"])
    np.savez_compressed(out, X=X, y14=y14, y32=y32)
    return out


def load_runs(runs, feature_mode: str = "noise_diff_norm"):
    """複数 run の特徴を結合して返す。

    Returns
    -------
    X    : (N, 1024) float32
    y14  : (N,) int64
    y32  : (N,) int64
    run_id : (N,) int64   各 frame の run インデックス (cross-run 分割用)
    """
    Xs, y14s, y32s, rids = [], [], [], []
    for ri, run in enumerate(runs):
        npz = np.load(build_run_cache(run, feature_mode))
        Xs.append(npz["X"])
        y14s.append(npz["y14"])
        y32s.append(npz["y32"])
        rids.append(np.full(len(npz["y14"]), ri, dtype=np.int64))
    return (np.concatenate(Xs), np.concatenate(y14s),
            np.concatenate(y32s), np.concatenate(rids))


if __name__ == "__main__":
    # スモークテスト: 1 run だけキャッシュして形状を出す
    runs = sys.argv[1:] or DEFAULT_RUNS[:2]
    X, y14, y32, rid = load_runs(runs)
    print(f"runs={runs}")
    print(f"X={X.shape} {X.dtype}  y14={y14.shape}  y32={y32.shape}  runs={rid.max()+1}")
    print(f"14cls hist: {np.bincount(y14, minlength=N_CLS_14)}")
    print(f"32cls present: {np.unique(y32).size} states")
    print(f"feature stats: mean={X.mean():.3f} std={X.std():.3f} "
          f"min={X.min():.2f} max={X.max():.2f}")
