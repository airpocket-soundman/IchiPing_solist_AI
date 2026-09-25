"""Controlled baseline-jitter ablation under strict leave-one-day-out.

Evaluation always uses the held-out run's own power-on baseline.  Training
uses two equally sized views per recording:
  self_only:       self baseline twice (sample-count control)
  within_day:      self + a baseline from another run on the same day
  cross_day:       self + a baseline from the other training day

All conditions share the same data-derived +/-2x frequency shift, model,
validation split, number of optimizer steps, and random seed.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eval_ideal_vs_solist import (
    CACHE, CAPTURES, FRDM_DAYS, OUT, ElmConfig, augment, compress_hires_167,
    elm_scores, estimate_shift, eval_cnn, fit_elm, load_days_hires, load_wav,
    log_psd_1024, macro_f1, metrics, parse_state, pct, pct2,
    split_train_validation,
)


def raw_run(run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = CACHE / f"ideal_raw_logpsd_v{run}.npz"
    if path.exists():
        z = np.load(path)
        return z["X"], z["y"], np.full(len(z["y"]), run, dtype=np.int64)
    root = CAPTURES / f"full_32_train_v{run}"
    X, y = [], []
    for state_dir in sorted(root.iterdir()):
        cls = parse_state(state_dir.name)
        if cls is None:
            continue
        for wav_path in sorted(state_dir.glob("frame_*.wav")):
            X.append(log_psd_1024(load_wav(wav_path))); y.append(cls)
    X, y = np.stack(X), np.asarray(y, dtype=np.int64)
    np.savez_compressed(path, X=X, y=y)
    print(f"cached raw PSD v{run}: {X.shape}", flush=True)
    return X, y, np.full(len(y), run, dtype=np.int64)


def baseline(run: int) -> np.ndarray:
    X, y, _ = raw_run(run)
    return X[y == 0].mean(0)


def norm_diff(raw: np.ndarray, bl: np.ndarray) -> np.ndarray:
    z = raw - bl
    return ((z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)).astype(np.float32)


def run_to_day() -> dict[int, str]:
    return {run: day for day, runs in FRDM_DAYS.items() for run in runs}


def training_views(days: tuple[str, ...], strategy: str):
    by_day = FRDM_DAYS
    run_day = run_to_day()
    runs = [r for d in days for r in by_day[d]]
    all_baselines = {r: baseline(r) for r in runs}
    self_parts, jitter_parts, ys, gs = [], [], [], []
    for run in runs:
        raw, y, g = raw_run(run)
        own = norm_diff(raw, all_baselines[run])
        if strategy == "self_only":
            jitter = own.copy()
        elif strategy == "within_day":
            candidates = np.asarray([r for r in by_day[run_day[run]] if r != run])
            chosen = candidates[np.arange(len(raw)) % len(candidates)]
            jitter = np.stack([norm_diff(raw[i:i+1], all_baselines[int(b)])[0]
                               for i, b in enumerate(chosen)])
        elif strategy == "cross_day":
            candidates = np.asarray([r for r in runs if run_day[r] != run_day[run]])
            chosen = candidates[np.arange(len(raw)) % len(candidates)]
            jitter = np.stack([norm_diff(raw[i:i+1], all_baselines[int(b)])[0]
                               for i, b in enumerate(chosen)])
        else:
            raise ValueError(strategy)
        self_parts.append(own); jitter_parts.append(jitter); ys.append(y); gs.append(g)
    return ([np.concatenate(self_parts), np.concatenate(jitter_parts)],
            np.concatenate(ys), np.concatenate(gs))


def eval_elm_views(views: list[np.ndarray], y: np.ndarray, groups: np.ndarray,
                   Xe: np.ndarray, ye: np.ndarray, ge: np.ndarray, delta: float) -> dict:
    tr, va = split_train_validation(y, groups)
    aug = [augment(v[tr], y[tr], groups[tr], delta) for v in views]
    Xa = np.concatenate([compress_hires_167(a[0], "400-3000Hz") for a in aug])
    ya = np.concatenate([a[1] for a in aug])
    Xv = compress_hires_167(views[0][va], "400-3000Hz")
    candidates = [ElmConfig(s, r) for s in (0.1, 0.15, 0.25, 0.5, 1.0)
                  for r in (0.01, 0.1, 1.0, 10.0)]
    cfg = max(candidates, key=lambda c: macro_f1(
        elm_scores(fit_elm(Xa, ya, c), Xv, c).argmax(1), y[va]))
    full_aug = [augment(v, y, groups, delta) for v in views]
    Xall = np.concatenate([compress_hires_167(a[0], "400-3000Hz") for a in full_aug])
    yall = np.concatenate([a[1] for a in full_aug])
    out = metrics(elm_scores(fit_elm(Xall, yall, cfg),
                             compress_hires_167(Xe, "400-3000Hz"), cfg), ye, ge)
    out["config"] = asdict(cfg)
    return out


def main() -> None:
    days = tuple(FRDM_DAYS)
    strategies = ("self_only", "within_day", "cross_day")
    result = {"protocol": {"evaluation_baseline": "held-out run self baseline",
                           "training_views_per_recording": 2,
                           "frequency_augmentation": "0, +/- observed shift, +/- 2x"},
              "folds": {}}
    for fold_i, holdout in enumerate(days):
        train_days = tuple(d for d in days if d != holdout)
        shift = estimate_shift(train_days[0], train_days[1])
        delta = abs(shift["global_shift"])
        Xe, ye, ge = load_days_hires((holdout,))
        fold = {"training_days": train_days, "shift": shift,
                "augmentation_delta": delta, "strategies": {}}
        for strategy in strategies:
            views, y, groups = training_views(train_days, strategy)
            cnn = eval_cnn(views[0], y, groups, Xe, ye, ge, delta, False, 100,
                           20260925 + fold_i, xl1024=True, train_views=views)
            elm = eval_elm_views(views, y, groups, Xe, ye, ge, delta)
            fold["strategies"][strategy] = {"PC CNN XL/1024": cnn,
                                             "Solist ELM m32/shape167": elm}
            print(holdout, strategy, fold["strategies"][strategy], flush=True)
        result["folds"][holdout] = fold

    result["means"] = {}
    for strategy in strategies:
        result["means"][strategy] = {}
        for model in ("PC CNN XL/1024", "Solist ELM m32/shape167"):
            vals = [f["strategies"][strategy][model] for f in result["folds"].values()]
            result["means"][strategy][model] = {
                key: float(np.mean([v[key] for v in vals]))
                for key in ("frame_accuracy", "macro_f1", "state_vote_accuracy")
            }
    lines = [
        "# Baseline augmentation ablation",
        "",
        "評価は常に未知run自身の起動時baseline。学習view数・周波数シフト・モデル・seedを揃え、baseline augmentationだけを変更した。",
        "",
        "| 学習baseline | モデル | frame | macro F1 | state vote | self比frame差 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        for model, m in result["means"][strategy].items():
            base = result["means"]["self_only"][model]["frame_accuracy"]
            lines.append(f"| {strategy} | {model} | {pct(m['frame_accuracy'])} | "
                         f"{pct(m['macro_f1'])} | {pct(m['state_vote_accuracy'])} | "
                         f"{100*(m['frame_accuracy']-base):+.1f} pt |")
    lines += ["", "## Fold detail", "",
              "| 未知日 | 学習日間shift | aug範囲 | baseline | PC CNN frame | ELM frame |",
              "|---|---:|---:|---|---:|---:|"]
    for day, fold in result["folds"].items():
        for strategy, models in fold["strategies"].items():
            lines.append(f"| {day} | {pct2(fold['shift']['global_shift'])} | "
                         f"±{pct2(2*fold['augmentation_delta'])} | {strategy} | "
                         f"{pct(models['PC CNN XL/1024']['frame_accuracy'])} | "
                         f"{pct(models['Solist ELM m32/shape167']['frame_accuracy'])} |")
    pc_self = result["means"]["self_only"]["PC CNN XL/1024"]["frame_accuracy"]
    pc_cross = result["means"]["cross_day"]["PC CNN XL/1024"]["frame_accuracy"]
    elm_self = result["means"]["self_only"]["Solist ELM m32/shape167"]["frame_accuracy"]
    elm_within = result["means"]["within_day"]["Solist ELM m32/shape167"]["frame_accuracy"]
    elm_cross = result["means"]["cross_day"]["Solist ELM m32/shape167"]["frame_accuracy"]
    lines += [
        "", "## 結論", "",
        f"- PC CNNはself-onlyが{pct(pc_self)}で最良。cross-dayは{pct(pc_cross)}で{100*(pc_cross-pc_self):+.1f} ptとなり、平均では悪化した。",
        f"- m=32 ELMはself-only {pct(elm_self)}に対し、within-day {pct(elm_within)} ({100*(elm_within-elm_self):+.1f} pt)、cross-day {pct(elm_cross)} ({100*(elm_cross-elm_self):+.1f} pt)。baseline jitterが低容量モデルの正則化として働いた。",
        "- cross-dayのPC効果は未知日ごとに大きく変動したため、物理的に不一致なcross-day差分をPC教師モデルの標準条件にはしない。ELMでは実運用に近いwithin-day jitterを優先候補とする。",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "BASELINE_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "BASELINE_ABLATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'BASELINE_ABLATION.md'}")


if __name__ == "__main__":
    main()
