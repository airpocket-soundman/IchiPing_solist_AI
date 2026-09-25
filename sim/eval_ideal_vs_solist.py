"""Same-session baseline + data-derived frequency shift evaluation.

Protocol
--------
* Original IchiPing v6..v12 are grouped into three acquisition days.
* Every run is differenced only against its own s00000 baseline.
* For each outer leave-one-day-out fold, the frequency shift is estimated only
  between the two training days.  Training augmentation spans +/- twice that
  observed shift; the held-out day is never shifted or used for selection.
* Compare a PC 1-D CNN using 1024-bin noise_diff_norm, a PC CNN using the
  current Solist 167-bin frontend, and the Solist-compatible fixed-alpha
  ELM (D=167, m=32).

Run with the IchiPing PyTorch environment, for example:
  D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_ideal_vs_solist.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "sim" / "_cache"
OUT = ROOT / "sim_export" / "solist_ds"
CAPTURES = Path(r"D:/GitHub/IchiPing/pc/captures")
ALPHA32 = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float64)

BIN_HZ = 16_000.0 / 1024
LO = max(0, int(round(400 / BIN_HZ)) - 1)
HI = min(512, int(round(3000 / BIN_HZ)))
HIRES_LO = int(round(400 / (16_000.0 / 2048))) - 1
HIRES_HI = int(round(3000 / (16_000.0 / 2048)))
FRDM_DAYS = {
    "2026-05-30": (6, 7, 8),
    "2026-05-31": (9, 10),
    "2026-06-01": (11, 12),
}


@dataclass(frozen=True)
class ElmConfig:
    scale: float
    ridge: float


def self_cache(run: int) -> Path:
    candidates = [
        CACHE / f"v612_full_32_train_v{run}__bl_self_v{run}.npz",
        CACHE / f"v612_full_32_train_v{run}__bl_v{run}.npz",
        CACHE / f"v612_full_32_train_v{run}__bl__v{run}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"self-baseline cache not found for v{run}")


def load_run(run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(self_cache(run))
    return (z["X"].astype(np.float32), z["y32"].astype(np.int64),
            np.full(len(z["y32"]), run, dtype=np.int64))


def parse_state(name: str) -> int | None:
    if not (name.startswith("s") and len(name) == 6 and set(name[1:]) <= {"0", "1"}):
        return None
    return sum(int(bit) << i for i, bit in enumerate(name[1:]))


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def log_psd_1024(audio: np.ndarray) -> np.ndarray:
    nfft, hop = 2048, 1024
    starts = range(0, len(audio) - nfft + 1, hop)
    seg = np.stack([audio[s:s + nfft] for s in starts]) * np.hanning(nfft)
    power = np.mean(np.abs(np.fft.rfft(seg, axis=1)) ** 2, axis=0)[1:]
    return np.maximum(10.0 * np.log10(power + 1e-12), -80.0).astype(np.float32)


def load_run_hires(run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = CACHE / f"ideal_noise_diff_norm_v{run}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"], np.full(len(z["y"]), run, dtype=np.int64)
    root = CAPTURES / f"full_32_train_v{run}"
    baseline_paths = sorted((root / "s00000").glob("frame_*.wav"))
    baseline = np.mean(np.stack([log_psd_1024(load_wav(p)) for p in baseline_paths]), axis=0)
    X, y = [], []
    for state_dir in sorted(root.iterdir()):
        cls = parse_state(state_dir.name)
        if cls is None:
            continue
        for wav_path in sorted(state_dir.glob("frame_*.wav")):
            feat = log_psd_1024(load_wav(wav_path)) - baseline
            feat = (feat - feat.mean()) / (feat.std() + 1e-6)
            X.append(feat.astype(np.float32)); y.append(cls)
    X, y = np.stack(X), np.asarray(y, dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y)
    print(f"cached hires self-baseline v{run}: {X.shape}", flush=True)
    return X, y, np.full(len(y), run, dtype=np.int64)


def load_days(days: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = [load_run(run) for day in days for run in FRDM_DAYS[day]]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def load_days_hires(days: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = [load_run_hires(run) for day in days for run in FRDM_DAYS[day]]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def shift_spectrum(X: np.ndarray, eps: float) -> np.ndarray:
    """Scale the frequency axis by 1+eps; X is DC-excluded 512-bin dB."""
    bins = np.arange(1, X.shape[1] + 1, dtype=np.float64)
    xp = bins / (1.0 + eps)
    lo = np.clip(np.floor(xp).astype(np.int64) - 1, 0, X.shape[1] - 1)
    hi = np.clip(lo + 1, 0, X.shape[1] - 1)
    w = xp - np.floor(xp)
    return (X[:, lo] * (1.0 - w) + X[:, hi] * w).astype(np.float32)


def day_class_means(day: str) -> np.ndarray:
    X, y, _ = load_days_hires((day,))
    return np.stack([X[y == c].mean(0) for c in range(32)])


def corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return np.sum(A * B, axis=1) / den


def estimate_shift(day_a: str, day_b: str) -> dict:
    """Estimate target(day_b) ~= shift(source(day_a), eps) from class means."""
    A = day_class_means(day_a)
    B = day_class_means(day_b)
    grid = np.linspace(-0.06, 0.06, 241)
    objective = []
    per_class = np.empty((len(grid), 32), np.float64)
    for i, eps in enumerate(grid):
        c = corr_rows(shift_spectrum(A, float(eps))[:, HIRES_LO:HIRES_HI],
                      B[:, HIRES_LO:HIRES_HI])
        per_class[i] = c
        objective.append(float(np.median(c)))
    best_i = int(np.argmax(objective))
    class_eps = grid[np.argmax(per_class, axis=0)]
    return {
        "day_a": day_a,
        "day_b": day_b,
        "global_shift": float(grid[best_i]),
        "median_class_shift": float(np.median(class_eps)),
        "class_shift_iqr": [float(np.percentile(class_eps, 25)),
                            float(np.percentile(class_eps, 75))],
        "median_shape_correlation": float(objective[best_i]),
    }


def augment(X: np.ndarray, y: np.ndarray, groups: np.ndarray, delta: float):
    # Keep original data and mix observed-scale and 2x extrapolated shifts.
    eps_values = (0.0, -delta, delta, -2.0 * delta, 2.0 * delta)
    Xs = [X if eps == 0 else shift_spectrum(X, eps) for eps in eps_values]
    # noise_diff_norm remains shape-only after interpolation.
    if X.shape[1] == 1024:
        Xs = [(z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6) for z in Xs]
    return np.concatenate(Xs), np.tile(y, len(eps_values)), np.tile(groups, len(eps_values))


def macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    values = []
    for c in range(32):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    pred = scores.argmax(1)
    votes = []
    for run in np.unique(groups):
        for c in range(32):
            idx = np.where((groups == run) & (y == c))[0]
            if len(idx):
                votes.append(int(scores[idx].sum(0).argmax() == c))
    return {
        "frame_accuracy": float(np.mean(pred == y)),
        "macro_f1": macro_f1(pred, y),
        "state_vote_accuracy": float(np.mean(votes)),
    }


def split_train_validation(y: np.ndarray, groups: np.ndarray, seed: int = 20260925):
    """Hold out 20% within every run/class; no held-out-day samples are used."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), dtype=bool)
    for run in np.unique(groups):
        for c in range(32):
            idx = np.where((groups == run) & (y == c))[0]
            rng.shuffle(idx)
            val[idx[:max(1, len(idx) // 5)]] = True
    return ~val, val


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def onehot(y: np.ndarray) -> np.ndarray:
    z = np.zeros((len(y), 32), np.float64)
    z[np.arange(len(y)), y] = 1.0
    return z


def fit_elm(X: np.ndarray, y: np.ndarray, cfg: ElmConfig):
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    H = hard_sigmoid((((X - mu) / sd) * cfg.scale) @ ALPHA32)
    beta = np.linalg.solve(H.T @ H + cfg.ridge * np.eye(32), H.T @ onehot(y))
    return mu, sd, beta


def elm_scores(model, X: np.ndarray, cfg: ElmConfig) -> np.ndarray:
    mu, sd, beta = model
    H = hard_sigmoid((((X - mu) / sd) * cfg.scale) @ ALPHA32)
    return H @ beta


def select_elm(X: np.ndarray, y: np.ndarray, groups: np.ndarray, delta: float) -> ElmConfig:
    tr, va = split_train_validation(y, groups)
    Xa, ya, _ = augment(X[tr], y[tr], groups[tr], delta)
    candidates = [ElmConfig(s, r) for s in (0.1, 0.15, 0.25, 0.5, 1.0)
                  for r in (0.01, 0.1, 1.0, 10.0)]
    return max(candidates, key=lambda cfg: macro_f1(
        elm_scores(fit_elm(Xa[:, LO:HI], ya, cfg), X[va, LO:HI], cfg).argmax(1), y[va]))


def eval_elm(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             Xe: np.ndarray, ye: np.ndarray, ge: np.ndarray, delta: float) -> tuple[dict, ElmConfig]:
    cfg = select_elm(X, y, groups, delta)
    Xa, ya, _ = augment(X, y, groups, delta)
    model = fit_elm(Xa[:, LO:HI], ya, cfg)
    return metrics(elm_scores(model, Xe[:, LO:HI], cfg), ye, ge), cfg


def compress_hires_167(X: np.ndarray, mode: str) -> np.ndarray:
    """Reduce the 1024-bin shape feature to the Solist input width."""
    source = X if mode == "fullband" else X[:, HIRES_LO:HIRES_HI]
    xp = np.linspace(0.0, source.shape[1] - 1.0, 167)
    lo = np.floor(xp).astype(np.int64)
    hi = np.minimum(lo + 1, source.shape[1] - 1)
    w = xp - lo
    return (source[:, lo] * (1.0 - w) + source[:, hi] * w).astype(np.float32)


def eval_elm_hires(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   Xe: np.ndarray, ye: np.ndarray, ge: np.ndarray, delta: float) -> tuple[dict, str, ElmConfig]:
    tr, va = split_train_validation(y, groups)
    Xa, ya, _ = augment(X[tr], y[tr], groups[tr], delta)
    candidates = [(mode, ElmConfig(s, r)) for mode in ("fullband", "400-3000Hz")
                  for s in (0.1, 0.15, 0.25, 0.5, 1.0) for r in (0.01, 0.1, 1.0, 10.0)]
    mode, cfg = max(candidates, key=lambda item: macro_f1(
        elm_scores(fit_elm(compress_hires_167(Xa, item[0]), ya, item[1]),
                   compress_hires_167(X[va], item[0]), item[1]).argmax(1), y[va]))
    Xall, yall, _ = augment(X, y, groups, delta)
    model = fit_elm(compress_hires_167(Xall, mode), yall, cfg)
    out = metrics(elm_scores(model, compress_hires_167(Xe, mode), cfg), ye, ge)
    return out, mode, cfg


def eval_cnn(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             Xe: np.ndarray, ye: np.ndarray, ge: np.ndarray, delta: float,
             crop167: bool, epochs: int, seed: int, xl1024: bool = False,
             train_views: list[np.ndarray] | None = None) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Run with the IchiPing/pc PyTorch virtual environment") from exc

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = split_train_validation(y, groups, seed)
    if train_views is None:
        train_views = [X]
    augmented = [augment(view[tr], y[tr], groups[tr], delta) for view in train_views]
    Xtr = np.concatenate([a[0] for a in augmented])
    ytr = np.concatenate([a[1] for a in augmented])
    Xva, yva = X[va], y[va]
    if crop167:
        Xtr, Xva, Xe = Xtr[:, LO:HI], Xva[:, LO:HI], Xe[:, LO:HI]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xva = ((Xva - mu) / sd).astype(np.float32)
    Xen = ((Xe - mu) / sd).astype(np.float32)

    class ShapeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            if xl1024:
                self.net = nn.Sequential(
                    nn.Conv1d(1, 32, 16, stride=4), nn.BatchNorm1d(32), nn.ReLU(),
                    nn.Conv1d(32, 64, 8, stride=4), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Conv1d(64, 128, 4, stride=2), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Flatten(), nn.Dropout(0.4), nn.Linear(128 * 30, 32),
                )
            else:
                self.net = nn.Sequential(
                    nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
                    nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Dropout(0.25), nn.Linear(128 * 42, 32),
                )
        def forward(self, z):
            return self.net(z)

    model = ShapeCNN().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr[:, None, :]), torch.from_numpy(ytr)),
                        batch_size=256, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_f1, best_loss, stale = None, -1.0, math.inf, 0
    va_tensor = torch.from_numpy(Xva[:, None, :]).to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_logits = model(va_tensor)
            val_loss = float(loss_fn(va_logits, torch.from_numpy(yva).to(device)).item())
            pred = va_logits.argmax(1).cpu().numpy()
        f1 = macro_f1(pred, yva)
        if val_loss < best_loss - 1e-5:
            best_f1 = f1; best_loss = val_loss; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 12:
            break
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(Xen[:, None, :]).to(device)).cpu().numpy()
    out = metrics(scores, ye, ge)
    out.update({"validation_macro_f1": best_f1, "validation_loss": best_loss,
                "epochs_trained": epoch + 1,
                "parameters": int(sum(p.numel() for p in model.parameters())),
                "device": str(device)})
    return out


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def pct2(x: float) -> str:
    return f"{100*x:.2f}%"


def write_report(result: dict) -> None:
    lines = [
        "# Same-baseline周波数シフト: PC理想モデル vs Solist-AI ELM",
        "",
        "各runは自身の起動時baselineだけで差分化した。外側leave-one-day-outの学習2日間から",
        "日間周波数シフトを推定し、その絶対値の2倍までを学習augmentationに使用した。",
        "評価日はシフト推定・モデル選択・標準化のいずれにも使用していない。",
        "",
        "| 未知日 | 学習日間シフト | aug範囲 | モデル | frame | macro F1 | state vote |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for day, fold in result["folds"].items():
        shift = fold["shift"]["global_shift"]
        delta = fold["augmentation_delta"]
        for model, m in fold["models"].items():
            lines.append(f"| {day} | {pct2(shift)} | ±{pct2(2*delta)} | {model} | "
                         f"{pct(m['frame_accuracy'])} | {pct(m['macro_f1'])} | "
                         f"{pct(m['state_vote_accuracy'])} |")
    lines += [
        "",
        "## 平均とSolist化による低下",
        "",
        "| モデル | frame | macro F1 | state vote | PC XL/1024からのframe低下 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, m in result["means"].items():
        lines.append(f"| {model} | {pct(m['frame_accuracy'])} | {pct(m['macro_f1'])} | "
                     f"{pct(m['state_vote_accuracy'])} | {pct(m['frame_drop_from_pc_ideal'])} |")
    ideal = result["means"]["PC CNN XL/1024"]
    legacy = result["means"]["Solist ELM m32/D167"]
    pc167 = result["means"]["PC CNN 167"]
    shape = result["means"]["Solist ELM m32/shape167"]
    lines += [
        "",
        "## 結論",
        "",
        f"- PC理想モデルはframe {pct(ideal['frame_accuracy'])} / macro F1 {pct(ideal['macro_f1'])}。",
        f"- 現行Solist ELMはframe {pct(legacy['frame_accuracy'])} / macro F1 {pct(legacy['macro_f1'])}で、PC理想モデルからframe {pct(ideal['frame_accuracy']-legacy['frame_accuracy'])}低下した。",
        f"- 低下はほぼ半分ずつで、1024→現行167特徴への変更がframe {pct(ideal['frame_accuracy']-pc167['frame_accuracy'])}、PC167 CNN→m=32 ELMがさらに{pct(pc167['frame_accuracy']-legacy['frame_accuracy'])}。",
        f"- 1024-bin形状を単純圧縮したshape167 ELMはframe {pct(shape['frame_accuracy'])}で、現行ELMとの差は{pct(shape['frame_accuracy']-legacy['frame_accuracy'])}。単純補間だけではCNNの局所形状表現を移せない。",
    ]
    lines += [
        "",
        "## 注意",
        "",
        "- Original世代は励振PRBSを再現できないため、同日baseline差分後のスペクトルをシフトした。PC側は1024-bin PSD差分、legacy Solist側は512-bin時間波形差分で、別日のbaselineは使っていない。",
        "- PC CNN XL/1024はUNO Qで実績のあるnoise_diff_norm特徴とXL構造による性能上限。PC CNN 167とlegacy ELMは現行の時間波形差分400–3000 Hz特徴を使う。",
        "- shape167 ELMは1024-bin noise_diff_normを167点へ補間圧縮し、公式Sim由来の同じ固定α・m=32へ入力した可能性評価。圧縮帯域は学習内validationだけで選択した。",
        "- 日数は3日だけなので信頼区間は広い。最終判断にはStamp-S3Aで最低3日、可能なら5日以上の収録が必要。",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IDEAL_VS_SOLIST.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "IDEAL_VS_SOLIST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    days = tuple(FRDM_DAYS)
    result = {"protocol": {"baseline": "same run only", "classes": 32,
                           "augmentation": "0, +/- observed day shift, +/- 2x observed day shift"},
              "folds": {}}
    for fold_i, holdout in enumerate(days):
        train_days = tuple(d for d in days if d != holdout)
        shift = estimate_shift(train_days[0], train_days[1])
        delta = abs(shift["global_shift"])
        X, y, g = load_days(train_days)
        Xe, ye, ge = load_days((holdout,))
        Xh, yh, gh = load_days_hires(train_days)
        Xhe, yhe, ghe = load_days_hires((holdout,))
        elm, cfg = eval_elm(X, y, g, Xe, ye, ge, delta)
        elm["config"] = asdict(cfg)
        shape_elm, shape_mode, shape_cfg = eval_elm_hires(Xh, yh, gh, Xhe, yhe, ghe, delta)
        shape_elm["compression"] = shape_mode
        shape_elm["config"] = asdict(shape_cfg)
        models = {
            "PC CNN XL/1024": eval_cnn(Xh, yh, gh, Xhe, yhe, ghe, delta, False,
                                        args.epochs, args.seed + fold_i, xl1024=True),
            "PC CNN 167": eval_cnn(X, y, g, Xe, ye, ge, delta, True, args.epochs, args.seed + 10 + fold_i),
            "Solist ELM m32/shape167": shape_elm,
            "Solist ELM m32/D167": elm,
        }
        result["folds"][holdout] = {"training_days": train_days, "shift": shift,
                                    "augmentation_delta": delta, "models": models}
        print(holdout, "shift", shift, "models", models, flush=True)
    ideal = np.mean([f["models"]["PC CNN XL/1024"]["frame_accuracy"] for f in result["folds"].values()])
    result["means"] = {}
    for name in next(iter(result["folds"].values()))["models"]:
        vals = [f["models"][name] for f in result["folds"].values()]
        frame = float(np.mean([v["frame_accuracy"] for v in vals]))
        result["means"][name] = {
            "frame_accuracy": frame,
            "macro_f1": float(np.mean([v["macro_f1"] for v in vals])),
            "state_vote_accuracy": float(np.mean([v["state_vote_accuracy"] for v in vals])),
            "frame_drop_from_pc_ideal": float(ideal - frame),
        }
    write_report(result)
    print(f"wrote {OUT / 'IDEAL_VS_SOLIST.md'}")


if __name__ == "__main__":
    main()
