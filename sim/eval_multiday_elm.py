"""旧cross-baseline条件を記録する探索スクリプト（運用汎化の採用判定には使わない）。

注意: この版は学習時に別日／別セッションのbaselineを差し引いており、起動時に
その日のbaselineを取得する実運用と一致しない。MULTIDAY_ELMの数値は比較参考値であり、
same-session baselineで再構築した評価に置き換える必要がある。

対象:
- Original IchiPing: full_32_train_v6..v12 を 2026-05-30/05-31/06-01 の
  3 日にまとめ、leave-one-day-out で評価する。
- Arduino IchiPing (UNO Q): 8 学習セッションで学習し、学習に使っていない
  gray/evening/survey/crowd の4セットで評価する。ハイパーパラメータは
  session1/4/6 の leave-one-session-out だけで決め、評価4セットを見て選ばない。
- UNO Q単独に加え、Original由来のFRDM学習データを足した既存の
  unoq+frdm_ir2 cacheも同じ未使用4セットで評価する。

Original の学習側だけは、holdout 日の baseline を一切使わず、学習日に属する
baseline を cross-baseline として使う。評価は各 run 自身の baseline を使う。

出力:
  sim_export/solist_ds/MULTIDAY_ELM.json
  sim_export/solist_ds/MULTIDAY_ELM.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "sim" / "_cache"
OUT = ROOT / "sim_export" / "solist_ds"
ALPHA32 = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float64)

BIN_HZ = 16_000.0 / 1024
LO = max(0, int(round(400 / BIN_HZ)) - 1)
HI = min(512, int(round(3000 / BIN_HZ)))
D = HI - LO
A_MAX = 0.205
RIDGE_ELM = 0.1

FRDM_DAYS = {
    "2026-05-30": (6, 7, 8),
    "2026-05-31": (9, 10),
    "2026-06-01": (11, 12),
}
FRDM_BASELINE_REP = {"2026-05-30": 6, "2026-05-31": 9, "2026-06-01": 11}
UNO_EVALS = ("unoq_gray", "unoq_evening", "unoq_survey", "unoq_crowd")


@dataclass(frozen=True)
class Config:
    kind: str
    hidden: int = 0
    scale: float = 1.0
    ridge: float = 1.0

    @property
    def name(self) -> str:
        if self.kind == "linear":
            return f"linear-ridge-lam{self.ridge:g}"
        alpha = "sim" if self.hidden == 32 else "random"
        return f"elm-m{self.hidden}-{alpha}-s{self.scale:g}"


def configs() -> list[Config]:
    out = [Config("linear", ridge=x) for x in (0.1, 1.0, 10.0, 100.0)]
    for m in (32, 64, 128):
        out += [Config("elm", hidden=m, scale=s, ridge=RIDGE_ELM)
                for s in (0.15, 0.25, 0.5, 1.0)]
    return out


def config_families(items: list[Config]) -> dict[str, list[Config]]:
    out = {"linear": [], "elm_m32": [], "elm_m64": [], "elm_m128": []}
    for cfg in items:
        key = "linear" if cfg.kind == "linear" else f"elm_m{cfg.hidden}"
        out[key].append(cfg)
    return {k: v for k, v in out.items() if v}


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def onehot(y: np.ndarray, n: int = 32) -> np.ndarray:
    z = np.zeros((len(y), n), np.float64)
    z[np.arange(len(y)), y] = 1.0
    return z


def alpha_for(m: int) -> np.ndarray:
    if m == 32:
        return ALPHA32
    return np.random.default_rng(1).uniform(-A_MAX, A_MAX, (D, m))


def fit_scores(X: np.ndarray, y: np.ndarray, Xe: np.ndarray, cfg: Config) -> np.ndarray:
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    Xen = (Xe - mu) / sd
    if cfg.kind == "linear":
        A = np.hstack((Xn, np.ones((len(Xn), 1))))
        Ae = np.hstack((Xen, np.ones((len(Xen), 1))))
        reg = cfg.ridge * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ onehot(y))
        return Ae @ beta
    alpha = alpha_for(cfg.hidden)
    H = hard_sigmoid((Xn * cfg.scale) @ alpha)
    beta = np.linalg.solve(H.T @ H + cfg.ridge * np.eye(cfg.hidden), H.T @ onehot(y))
    return hard_sigmoid((Xen * cfg.scale) @ alpha) @ beta


def metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    pred = scores.argmax(1)
    recalls = [float((pred[y == c] == c).mean()) for c in range(32) if np.any(y == c)]
    f1s = []
    for c in range(32):
        if not np.any(y == c):
            continue
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    votes = []
    for g in np.unique(groups):
        for c in range(32):
            idx = np.where((groups == g) & (y == c))[0]
            if len(idx):
                votes.append(int(scores[idx].sum(0).argmax() == c))
    return {
        "frame_accuracy": float((pred == y).mean()),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "state_vote_accuracy": float(np.mean(votes)),
    }


def shift_spectrum_batch(X: np.ndarray, eps: float) -> np.ndarray:
    bins = np.arange(1, X.shape[1] + 1, dtype=np.float64)
    xp = bins / (1.0 + eps)
    lo = np.floor(xp).astype(int) - 1
    hi = np.clip(lo + 1, 0, X.shape[1] - 1)
    lo = np.clip(lo, 0, X.shape[1] - 1)
    w = xp - np.floor(xp)
    return (X[:, lo] * (1.0 - w) + X[:, hi] * w).astype(np.float32)


def frdm_cache(run: int, baseline: int) -> Path:
    candidates = sorted(CACHE.glob(f"v612_full_32_train_v{run}__bl*{baseline}.npz"))
    candidates = [p for p in candidates if "self" not in p.name]
    if not candidates:
        raise FileNotFoundError(f"cache missing for run v{run}, baseline v{baseline}")
    return candidates[0]


def load_frdm(run: int, baseline: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(frdm_cache(run, baseline))
    return z["X"].astype(np.float32), z["y32"].astype(np.int64)


def frdm_training(days: tuple[str, ...], shift: bool) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    baselines = [FRDM_BASELINE_REP[d] for d in days]
    for day in days:
        for run in FRDM_DAYS[day]:
            for baseline in baselines:
                X, y = load_frdm(run, baseline)
                Xs.append(X); ys.append(y)
                if shift:
                    Xs.extend((shift_spectrum_batch(X, -0.03), shift_spectrum_batch(X, 0.03)))
                    ys.extend((y, y))
    X = np.concatenate(Xs)[:, LO:HI]
    return X.astype(np.float64), np.concatenate(ys)


def frdm_test(day: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, gs = [], [], []
    for run in FRDM_DAYS[day]:
        X, y = load_frdm(run, run)
        Xs.append(X[:, LO:HI]); ys.append(y); gs.append(np.full(len(y), run))
    return np.concatenate(Xs).astype(np.float64), np.concatenate(ys), np.concatenate(gs)


def select_frdm_config(train_days: tuple[str, ...], shift: bool, candidates: list[Config]) -> Config:
    scores = {c.name: [] for c in candidates}
    for val_day in train_days:
        inner_days = tuple(d for d in train_days if d != val_day)
        X, y = frdm_training(inner_days, shift)
        Xe, ye, ge = frdm_test(val_day)
        for cfg in candidates:
            scores[cfg.name].append(metrics(fit_scores(X, y, Xe, cfg), ye, ge)["macro_f1"])
    return max(candidates, key=lambda c: np.mean(scores[c.name]))


def eval_frdm(candidates: list[Config]) -> dict:
    result = {}
    days = tuple(FRDM_DAYS)
    for recipe, shift in (("cross_baseline", False), ("cross_baseline+freq_shift", True)):
        result[recipe] = {}
        for family, family_cfgs in config_families(candidates).items():
            folds = {}
            for holdout in days:
                train_days = tuple(d for d in days if d != holdout)
                cfg = select_frdm_config(train_days, shift, family_cfgs)
                X, y = frdm_training(train_days, shift)
                Xe, ye, ge = frdm_test(holdout)
                folds[holdout] = {"config": cfg.name, **metrics(fit_scores(X, y, Xe, cfg), ye, ge)}
                print("FRDM", recipe, family, holdout, folds[holdout], flush=True)
            result[recipe][family] = {
                "folds": folds,
                "mean_frame_accuracy": float(np.mean([v["frame_accuracy"] for v in folds.values()])),
                "mean_macro_recall": float(np.mean([v["macro_recall"] for v in folds.values()])),
                "mean_macro_f1": float(np.mean([v["macro_f1"] for v in folds.values()])),
                "mean_state_vote_accuracy": float(np.mean([v["state_vote_accuracy"] for v in folds.values()])),
            }
    return result


def uno_validation_rows(z, run: int) -> np.ndarray:
    # session1/4/6 are the three sessions whose own baseline is present.
    domain = z["domain"] if "domain" in z.files else np.zeros(len(z["run_id"]), dtype=int)
    return ((domain == 0) & (z["run_id"] == run) &
            (z["bl_id"] == run) & (z["eps"] == 0))


def select_uno_config(z, candidates: list[Config]) -> Config:
    cv = {c.name: [] for c in candidates}
    domain = z["domain"] if "domain" in z.files else np.zeros(len(z["run_id"]), dtype=int)
    for val_run in (0, 3, 5):
        # UNOのvalidation sessionだけを除外する。併用条件のFRDM run_idは
        # UNOと番号が重なるため、domainを見ずに除外してはいけない。
        train = ~((domain == 0) & (z["run_id"] == val_run))
        val = uno_validation_rows(z, val_run)
        X, y = z["X"][train].astype(np.float64), z["y32"][train]
        Xe, ye = z["X"][val].astype(np.float64), z["y32"][val]
        ge = np.full(len(ye), val_run)
        for cfg in candidates:
            cv[cfg.name].append(metrics(fit_scores(X, y, Xe, cfg), ye, ge)["macro_f1"])
    return max(candidates, key=lambda c: np.mean(cv[c.name]))


def eval_uno(candidates: list[Config], quick: bool) -> dict:
    result = {}
    variants = ("unoq_none", "unoq_ir2", "unoq+frdm_ir2")
    if quick:
        candidates = [c for c in candidates if c.kind == "linear" or c.hidden in (32, 64)]
    for variant in variants:
        z = np.load(CACHE / f"solist_ds_{variant}.npz")
        result[variant] = {}
        for family, family_cfgs in config_families(candidates).items():
            cfg = select_uno_config(z, family_cfgs)
            X, y = z["X"].astype(np.float64), z["y32"]
            sets = {}
            for e in UNO_EVALS:
                Xe, ye = z[f"eval_{e}_X"].astype(np.float64), z[f"eval_{e}_y32"]
                sets[e] = metrics(fit_scores(X, y, Xe, cfg), ye, np.zeros(len(ye), dtype=int))
            result[variant][family] = {
                "config": cfg.name,
                "sets": sets,
                "mean_frame_accuracy": float(np.mean([v["frame_accuracy"] for v in sets.values()])),
                "mean_macro_recall": float(np.mean([v["macro_recall"] for v in sets.values()])),
                "mean_macro_f1": float(np.mean([v["macro_f1"] for v in sets.values()])),
                "mean_state_vote_accuracy": float(np.mean([v["state_vote_accuracy"] for v in sets.values()])),
            }
            print("UNO", variant, family, cfg.name, result[variant][family], flush=True)
    return result


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def write_report(result: dict) -> None:
    lines = [
        "# 旧cross-baseline探索結果（運用汎化の採用判定には使用不可）",
        "",
        "> **重要:** 学習特徴に別日／別セッションのbaseline差分が含まれる。実運用では起動時に当日のbaselineを取得するため、この条件は不一致であり、以下の数値を汎化性能として採用しない。`audio_day - baseline_same_session`で特徴を作り直し、audioとbaselineを同じ率でシフトする再評価が必要。",
        "",
        "評価日／評価セットはハイパーパラメータ選択にも使用していない。Original IchiPing は日単位の",
        "leave-one-day-out、UNO Q は8学習セッションと4評価セットを分離した。値は frame accuracy /",
        "macro F1 / 32状態別スコア合算精度。モデル選択も内側holdoutのmacro F1で行った。",
        "",
        "## Original IchiPing: leave-one-day-out",
        "",
        "| 学習条件 | holdout日 | 選択モデル | frame | macro F1 | state vote |",
        "|---|---|---|---:|---:|---:|",
    ]
    for recipe, families in result["original_ichiping"].items():
        for family, rr in families.items():
            for day, v in rr["folds"].items():
                lines.append(f"| {recipe}/{family} | {day} | {v['config']} | {pct(v['frame_accuracy'])} | "
                             f"{pct(v['macro_f1'])} | {pct(v['state_vote_accuracy'])} |")
            lines.append(f"| **{recipe}/{family} 平均** | — | — | **{pct(rr['mean_frame_accuracy'])}** | "
                         f"**{pct(rr['mean_macro_f1'])}** | **{pct(rr['mean_state_vote_accuracy'])}** |")
    lines += [
        "",
        "## Arduino IchiPing / UNO Q: 未使用時刻・雑音セット",
        "",
        "| 学習条件 | 評価セット | 選択モデル | frame | macro F1 | state vote |",
        "|---|---|---|---:|---:|---:|",
    ]
    for variant, families in result["arduino_ichiping_uno_q"].items():
        for family, rr in families.items():
            for name, v in rr["sets"].items():
                lines.append(f"| {variant}/{family} | {name} | {rr['config']} | {pct(v['frame_accuracy'])} | "
                             f"{pct(v['macro_f1'])} | {pct(v['state_vote_accuracy'])} |")
            lines.append(f"| **{variant}/{family} 平均** | — | {rr['config']} | **{pct(rr['mean_frame_accuracy'])}** | "
                         f"**{pct(rr['mean_macro_f1'])}** | **{pct(rr['mean_state_vote_accuracy'])}** |")
    lines += [
        "",
        "## 結論と当初想定との差",
        "",
        "- 当初の99.1%はv12から各状態10フレームを校正用に取り、同じv12の残りを評価した現地校正値であり、日跨ぎ汎化値ではない。",
        "- 当初のfactory 32cls約92%もv6-v11学習→v12評価で、v11とv12は同じ2026-06-01収録である。今回の日単位holdoutとは難易度が異なる。",
        "- 厳密なOriginal日単位holdoutでは、実αのm=32 ELMは平均frame 60.1% / macro F1 54.4%。m=128でも64.0% / 60.5%で、95%級の汎化は確認できない。",
        "- OriginalをUNO Qへ加えると、未使用4条件のframe / macro F1平均はm=32で34.9% / 26.8%→37.6% / 29.2%、m=128で51.2% / 43.4%→58.7% / 51.9%。複数日データは有効だが、単純混合だけでは不十分。",
        "- 一律±3% shiftはOriginal m=32のframe / macro F1を60.1% / 54.4%→46.1% / 38.4%へ悪化させ、UNO QのIR shiftも34.9% / 26.8%→28.7% / 20.1%。温度シフト対策自体ではなく、現行D=167特徴への適用方法と分布設定が合っていない。",
        "- βだけを学習するELMでは固定ランダムαが捨てた識別情報を復元できない。パラメータが小さいことは実装上の長所だが、十分な汎化性能の根拠にはならない。",
        "",
        "## 次の改善実験（優先順）",
        "",
        "1. UNO Qで効果が確認済みの1024-bin `noise_diff_norm` と、実測シフト分布に基づく周波数伸縮を移植し、最後に167次元へ圧縮する。",
        "2. ランダムαを固定したままβだけを学ぶ条件と、教師ありで学習した167→32/64射影（またはteacherからの蒸留）を同じ日holdoutで比較する。",
        "3. 日・時刻・baseline・機器をgroupとして分離し、augmentation強度とELM scale/ridgeをnested CVで選ぶ。評価日は最後まで触らない。",
        "4. 最終Stamp-S3A収録で最低3日、各日複数時刻・baseline×3を取得し、未知日macro F1を採用判定にする。少量の現地校正あり／なしを別指標で管理する。",
    ]
    lines += [
        "",
        "## 解釈上の注意",
        "",
        "- Originalのfrequency shiftはPRBS seedを再現できない世代のため、時間波形ではなく512-bin差分スペクトルを±3%シフトした。",
        "- UNO Qの`unoq_ir2`は既知PRBSからIRを推定して時間伸縮後に再合成する、より物理的なaugmentationである。",
        "- `unoq+frdm_ir2`はUNO Q学習データへOriginal由来FRDMデータを加え、同じUNO Q未使用4セットで評価した条件である。",
        "- m=64/128のαは実機未プローブのため一様乱数による可能性評価。m=32だけが公式Simから採取した実αである。",
        "- UNO Qは複数セッション・複数時刻だが同一日。厳密な日跨ぎ評価はOriginal側だけである。",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MULTIDAY_ELM.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "MULTIDAY_ELM.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="UNO Qのm=128候補を省略")
    args = ap.parse_args()
    candidates = configs()
    result = {
        "protocol": {
            "feature": "FFT(audio-baseline), 1024-point Hann, magnitude mean, log20, 400-3000Hz, D=167",
            "classes": 32,
            "selection": "nested day/session holdout; final eval sets unused for selection",
        },
        "original_ichiping": eval_frdm(candidates),
        "arduino_ichiping_uno_q": eval_uno(candidates, args.quick),
    }
    write_report(result)
    print(f"wrote {OUT / 'MULTIDAY_ELM.md'}")


if __name__ == "__main__":
    main()
