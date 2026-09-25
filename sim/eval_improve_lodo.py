"""Original IchiPing 日単位 LODO で改良候補を比較する (self-baseline, CPU のみ)。

比較軸
  特徴 : D167 (時間波形差分 400–3000 Hz, 現行) / N1024 (1024-bin noise_diff_norm) / N333 (その 400–3000 Hz)
  モデル: ELM m=32 公式Simα (D167 のみ実機α) / ELM m=128 乱数α / 線形 ridge
  校正 : なし (factory) / 評価日の別 run から N frame/class で β 再計算 (Solist ODL 相当)
         → 同日の別 run で評価。校正 run と評価 run は必ず別 run (同一 run 内の分割はしない)。

厳密性
  * 外側 = 日単位 leave-one-day-out。評価日は学習・標準化・ハイパラ選択に使わない。
  * ハイパラは学習 2 日の内側 LODO (日単位 2 fold) の macro F1 平均で選ぶ。
  * 周波数シフト augmentation は使わない (RUN_SHIFT.md: Original 3 日の温度差は約 1°C)。
  * 校正の β 再計算方式 (校正のみ / factory+校正混合) は両方報告し、評価で選ばない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ideal_vs_solist import (FRDM_DAYS, HIRES_LO, HIRES_HI, LO, HI, ALPHA32,  # noqa: E402
                                  load_run, load_run_hires, hard_sigmoid)
from make_solist_dataset import class14  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
DAYS = tuple(FRDM_DAYS)
N_CAL = 10


def to14(y32: np.ndarray) -> np.ndarray:
    return np.array([class14([(int(v) >> k) & 1 for k in range(5)]) for v in y32])


def onehot(y, C):
    z = np.zeros((len(y), C)); z[np.arange(len(y)), y] = 1.0; return z


def macro_f1(pred, y, C):
    f = []
    for c in range(C):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def rand_alpha(D, m, seed, scale=0.205):
    return np.random.default_rng(seed).uniform(-scale, scale, (D, m))


# ------------------------------------------------------------------ models
class Elm:
    def __init__(self, alpha, s, lam):
        self.alpha, self.s, self.lam = alpha, s, lam

    def hidden(self, X):
        return hard_sigmoid(((X - self.mu) / self.sd * self.s) @ self.alpha)

    def fit(self, X, y, C):
        self.mu, self.sd, self.C = X.mean(0), X.std(0) + 1e-6, C
        H = self.hidden(X)
        self.G, self.B = H.T @ H, H.T @ onehot(y, C)            # factory 統計を保持 (混合校正用)
        self.beta = np.linalg.solve(self.G + self.lam * np.eye(len(self.G)), self.B)
        return self

    def recal(self, Xc, yc, mode):
        """β のみ再計算 (α・標準化は factory のまま = 実機 ODL と同じ自由度)。"""
        H = self.hidden(Xc); G, B = H.T @ H, H.T @ onehot(yc, self.C)
        if mode == "mix":                                         # factory 全体と校正を同じ総重みで混合
            w = self.G.trace() / max(G.trace(), 1e-12)
            G, B = self.G + w * G, self.B + w * B
        beta = np.linalg.solve(G + self.lam * np.eye(len(G)), B)
        new = Elm(self.alpha, self.s, self.lam); new.__dict__.update(self.__dict__); new.beta = beta
        return new

    def scores(self, X):
        return self.hidden(X) @ self.beta


class Linear:
    """z-score → ridge → one-hot 回帰。校正は最終層 W の再計算 (比較用、Solist では CPU 実装)。"""
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y, C):
        self.mu, self.sd, self.C = X.mean(0), X.std(0) + 1e-6, C
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        self.G, self.B = Z.T @ Z, Z.T @ onehot(y, C)
        self.W = np.linalg.solve(self.G + self.lam * np.eye(len(self.G)), self.B)
        return self

    def recal(self, Xc, yc, mode):
        Z = np.c_[(Xc - self.mu) / self.sd, np.ones(len(Xc))]
        G, B = Z.T @ Z, Z.T @ onehot(yc, self.C)
        if mode == "mix":
            w = self.G.trace() / max(G.trace(), 1e-12)
            G, B = self.G + w * G, self.B + w * B
        new = Linear(self.lam); new.__dict__.update(self.__dict__)
        new.W = np.linalg.solve(G + self.lam * np.eye(len(G)), B)
        return new

    def scores(self, X):
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.W


# ------------------------------------------------------------------ data
def load(day, feat):
    parts = []
    for run in FRDM_DAYS[day]:
        if feat == "D167":
            X, y, g = load_run(run); X = X[:, LO:HI]
        else:
            X, y, g = load_run_hires(run)
            if feat == "N333":
                X = X[:, HIRES_LO:HIRES_HI]
        parts.append((X.astype(np.float64), y, g))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def load_days(days, feat):
    parts = [load(d, feat) for d in days]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# 初回 (s≥0.1, λ≤1 / 線形 λ≤1e3) では選択値が格子端に張り付いたため拡張
ELM_GRID = [(s, l) for s in (0.03, 0.06, 0.1, 0.25, 0.5, 1.0) for l in (0.1, 1.0, 10.0, 100.0)]
LIN_GRID = [(l,) for l in (10, 100, 1e3, 1e4, 1e5)]

MODELS = {
    # name: (features, factory(params), param grid)
    "ELM m32 Simα / D167 (現行)": ("D167", lambda p: Elm(ALPHA32, *p),
                                   ELM_GRID),
    "線形 ridge / D167":          ("D167", lambda p: Linear(*p), LIN_GRID),
    "ELM m32 乱数α / N333":        ("N333", lambda p: Elm(rand_alpha(HIRES_HI - HIRES_LO, 32, 1), *p),
                                   ELM_GRID),
    "ELM m128 乱数α / N333":       ("N333", lambda p: Elm(rand_alpha(HIRES_HI - HIRES_LO, 128, 1), *p),
                                   ELM_GRID),
    "線形 ridge / N333":          ("N333", lambda p: Linear(*p), LIN_GRID),
    "線形 ridge / N1024":         ("N1024", lambda p: Linear(*p), LIN_GRID),
}


def select(make, grid, days, feat, C, lab):
    """学習日の内側 LODO で macro F1 最大のパラメータ。"""
    best = None
    for p in grid:
        f = []
        for d in days:
            tr = tuple(x for x in days if x != d)
            X, y, _ = load_days(tr, feat); Xv, yv, _ = load(d, feat)
            m = make(p).fit(X, lab(y), C)
            f.append(macro_f1(m.scores(Xv).argmax(1), lab(yv), C))
        if best is None or np.mean(f) > best[0]:
            best = (float(np.mean(f)), p)
    return best[1]


def metrics(S, y, C, y32):
    pred = S.argmax(1)
    out = {"frame": float(np.mean(pred == y)), "macro_f1": macro_f1(pred, y, C)}
    if C == 32:
        out["as14"] = float(np.mean(to14(pred) == to14(y32)))
    return out


def cal_split(y, g, run, rng):
    """run から class 毎に N_CAL frame (先頭) を校正用に取る。"""
    idx = []
    for c in np.unique(y):
        ci = np.where((g == run) & (y == c))[0]
        idx += list(ci[:N_CAL])
    return np.array(idx)


def main():
    res = {"n_cal": N_CAL, "folds": {}}
    rng = np.random.default_rng(0)
    for C, lab in ((32, lambda y: y), (14, to14)):
        for name, (feat, make, grid) in MODELS.items():
            per_fold = []
            for hold in DAYS:
                train = tuple(d for d in DAYS if d != hold)
                p = select(make, grid, train, feat, C, lab)
                X, y, _ = load_days(train, feat)
                model = make(p).fit(X, lab(y), C)
                Xe, ye, ge = load(hold, feat)
                runs = list(FRDM_DAYS[hold])
                fac = metrics(model.scores(Xe), lab(ye), C, ye)
                # 校正: 同日の別 run で校正 → 評価 run は校正に使わない
                cal = {"cal_only": [], "mix": [], "none_same_test": []}
                for rc in runs:
                    ci = cal_split(ye, ge, rc, rng)
                    for rt in runs:
                        if rt == rc:
                            continue
                        ti = np.where(ge == rt)[0]
                        cal["none_same_test"].append(metrics(model.scores(Xe[ti]), lab(ye[ti]), C, ye[ti]))
                        for mode in ("cal_only", "mix"):
                            mm = model.recal(Xe[ci], lab(ye[ci]), mode)
                            cal[mode].append(metrics(mm.scores(Xe[ti]), lab(ye[ti]), C, ye[ti]))
                cal = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in cal.items()}
                per_fold.append({"holdout": hold, "params": list(p), "factory": fac, **cal})
                print(f"{C}cls {name} {hold} p={p} factory={fac['frame']:.3f} "
                      f"cal_only={cal['cal_only']['frame']:.3f} mix={cal['mix']['frame']:.3f}", flush=True)
            res["folds"][f"{C}cls|{name}"] = per_fold
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IMPROVE_LODO.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)


def write_md(res):
    def mean(folds, key, m):
        v = [f[key][m] for f in folds if m in f[key]]
        return float(np.mean(v)) if v else float("nan")
    p = lambda v: "—" if v != v else f"{v*100:.1f}%"
    L = ["# 改良候補の比較 (Original 日単位 LODO, self-baseline)", "",
         "外側=日単位 leave-one-day-out 3 fold 平均。ハイパラは学習 2 日の内側 LODO で選択。周波数シフト aug なし。",
         f"校正 = 評価日の別 run から {res['n_cal']} frame/class で最終層 (ELM は β) を再計算し、同日の別 run で評価 "
         "(全 run 順序ペアの平均)。「校正なし(同評価集合)」は校正列と同じ評価 run での factory 値。", "",
         "| クラス | 特徴 / モデル | factory frame | factory macroF1 | →14cls換算 | 校正なし(同評価集合) | 校正のみ β | factory+校正 混合 β |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, folds in res["folds"].items():
        C, name = key.split("|")
        L.append(f"| {C} | {name} | {p(mean(folds,'factory','frame'))} | {p(mean(folds,'factory','macro_f1'))} | "
                 f"{p(mean(folds,'factory','as14'))} | {p(mean(folds,'none_same_test','frame'))} | "
                 f"{p(mean(folds,'cal_only','frame'))} | {p(mean(folds,'mix','frame'))} |")
    L += ["", "注: N333/N1024 の乱数α ELM は実機未プローブ α (一様乱数) による可能性評価。D167 の m32 のみ公式Sim実α。",
          "線形 ridge の校正は Solist の β 再計算とは別機構 (MCU CPU 実装想定) の参考値。"]
    (OUT / "IMPROVE_LODO.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
