"""Solist-AI (ELM, α固定, hidden≤64, bf16) の制約下で精度向上策を一括比較する。

評価: UNO Q eval 4 セット (gray/evening/survey/crowd, 各セット自前 baseline 校正) の 14cls frame 精度。
学習: sim/_cache/solist_ds_<variant>.npz (make_solist_dataset.py の出力)。
出力: sim_export/solist_ds/IMPROVE.md

usage: python sim/improve_experiments.py [--variant unoq_ir2] [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "sim" / "_cache"
OUT = ROOT / "sim_export" / "solist_ds"
ALPHA32 = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float32)   # Sim seed1 (167,32), U(-0.205,0.205)
EVALS = ("unoq_gray", "unoq_evening", "unoq_survey", "unoq_crowd")
A_MAX = 0.205


def hs(z):
    return np.clip(0.2 * z + 0.5, 0.0, 1.0)


def onehot(y, C):
    M = np.zeros((len(y), C), np.float32); M[np.arange(len(y)), y] = 1.0; return M


def solve(H, Y, lam):
    return np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1], dtype=H.dtype), H.T @ Y)


def rand_alpha(D, m, seed):
    return np.random.default_rng(seed).uniform(-A_MAX, A_MAX, (D, m)).astype(np.float32)


class Data:
    def __init__(self, variant):
        z = np.load(CACHE / f"solist_ds_{variant}.npz")
        self.X, self.y14, self.y32 = z["X"], z["y14"], z["y32"]
        self.ev = {e: (z[f"eval_{e}_X"], z[f"eval_{e}_y14"], z[f"eval_{e}_y32"]) for e in EVALS}


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def run_elm(Xtr, ytr, evs, alpha, s, lam, C=14, bias=None):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    H = hs(((Xtr - mu) / sd * s) @ alpha)
    beta = solve(H, onehot(ytr, C), lam)
    res = {}
    for e, (Xe, ye, _) in evs.items():
        res[e] = acc(hs(((Xe - mu) / sd * s) @ alpha) @ beta, ye)
    return res


def run_linear(Xtr, ytr, evs, lam, C=14):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1), np.float32)])
    W = solve(Xn, onehot(ytr, C), lam)
    return {e: acc(np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1), np.float32)]) @ W, ye) for e, (Xe, ye, _) in evs.items()}


def bin_avg(X, k):
    n = (X.shape[1] // k) * k
    return X[:, :n].reshape(len(X), -1, k).mean(2)


def frame_norm(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)


def pca(Xtr, k):
    mu = Xtr.mean(0); C = np.cov((Xtr - mu).T)
    w, V = np.linalg.eigh(C); return mu, V[:, ::-1][:, :k].astype(np.float32)


def lda(Xtr, ytr, k, reg=1e-2):
    mu = Xtr.mean(0); D = Xtr.shape[1]
    Sw = np.zeros((D, D)); Sb = np.zeros((D, D))
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]; mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc); Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    Sw /= len(Xtr); Sb /= len(Xtr)
    Sw += reg * np.trace(Sw) / D * np.eye(D)
    w, V = np.linalg.eigh(np.linalg.solve(Sw, Sb))
    return mu, V[:, ::-1][:, :k].astype(np.float32)


def fmt(res):
    m = np.mean([res[e] for e in EVALS])
    return " | ".join(f"{res[e]:.1%}" for e in EVALS) + f" | **{m:.1%}**", m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="unoq_ir2")
    ap.add_argument("--compare-variants", default="unoq_none,unoq_ir2,unoq_feat2,unoq+frdm_ir2,unoq_none_blall")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sections", default="ABCDEFGIJK")
    a = ap.parse_args()
    d = Data(a.variant)
    X, y = d.X, d.y14
    lines = [f"# 精度向上実験 (14cls frame 精度, 学習 variant `{a.variant}`, 評価 = UNO Q 4 セット自前 baseline)", "",
             "| 実験 | gray | evening | survey | crowd | 平均 |", "|---|---|---|---|---|---|"]
    rows = []

    secs = a.sections
    OUT.mkdir(parents=True, exist_ok=True)

    def add(name, res):
        s_, m = fmt(res); rows.append((m, name)); lines.append(f"| {name} | {s_} |"); print(f"{m:.1%}  {name}", flush=True)
        (OUT / f"IMPROVE_{secs}.md").write_text("\n".join(lines), encoding="utf-8")

    t0 = time.time()
    # A. baseline: Sim α m=32, s/λ grid
    for s in (0.15, 0.25, 0.5, 1.0, 2.0) if "A" in secs else ():
        for lam in (0.01, 0.1, 1.0, 10.0):
            add(f"A. ELM m=32 Simα s={s} λ={lam}", run_elm(X, y, d.ev, ALPHA32, s, lam))
    # B. hidden size (random α, mean of 3 seeds)
    for m in (32, 48, 64, 128, 256) if "B" in secs else ():
        for s in (0.25, 0.5):
            res = {e: np.mean([run_elm(X, y, d.ev, rand_alpha(167, m, sd), s, 0.1)[e] for sd in (1, 2, 3)]) for e in EVALS}
            add(f"B. ELM m={m} 乱数α(3seed平均) s={s} λ=0.1", res)
    # C. bin averaging (D reduction) with m=32/64
    for k in (2, 3, 4) if "C" in secs else ():
        Xk = bin_avg(X, k); evk = {e: (bin_avg(Xe, k), ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        Dk = Xk.shape[1]
        for m in (32, 64):
            for s in (0.25, 0.5, 1.0):
                res = {e: np.mean([run_elm(Xk, y, evk, rand_alpha(Dk, m, sd), s, 0.1)[e] for sd in (1, 2)]) for e in EVALS}
                add(f"C. bin平均 k={k} (D={Dk}) ELM m={m} s={s}", res)
        add(f"C. bin平均 k={k} (D={Dk}) 線形ridge λ=1", run_linear(Xk, y, evk, 1.0))
    # D. per-frame normalization
    if "D" in secs:
        Xf = frame_norm(X); evf = {e: (frame_norm(Xe), ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for s in (0.25, 0.5, 1.0):
            add(f"D. frame正規化 + ELM m=32 Simα s={s} λ=0.1", run_elm(Xf, y, evf, ALPHA32, s, 0.1))
        add("D. frame正規化 + ELM m=64 乱数α s=0.5", {e: np.mean([run_elm(Xf, y, evf, rand_alpha(167, 64, sd), 0.5, 0.1)[e] for sd in (1, 2)]) for e in EVALS})
        add("D. frame正規化 + 線形ridge λ=1", run_linear(Xf, y, evf, 1.0))
    # E. linear ridge (no hidden layer)
    for lam in (0.1, 1.0, 10.0, 100.0) if "E" in secs else ():
        add(f"E. 線形ridge D=167 λ={lam}", run_linear(X, y, d.ev, lam))
    # F. PCA / LDA front-end + ELM m=32 (Sim α rows) / linear
    for k in (16, 32, 48) if "F" in secs else ():
        mu, V = pca(X, k); Xp = (X - mu) @ V; evp = {e: ((Xe - mu) @ V, ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for s in (0.25, 0.5, 1.0):
            add(f"F. PCA{k} + ELM m=32 Simα[:{k}] s={s}", run_elm(Xp, y, evp, ALPHA32[:k], s, 0.1))
        add(f"F. PCA{k} + 線形ridge λ=1", run_linear(Xp, y, evp, 1.0))
    for k in (13,) if "F" in secs else ():
        mu, V = lda(X, y, k); Xl = (X - mu) @ V; evl = {e: ((Xe - mu) @ V, ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for s in (0.25, 0.5, 1.0, 2.0):
            add(f"F. LDA{k} + ELM m=32 Simα[:{k}] s={s}", run_elm(Xl, y, evl, ALPHA32[:k], s, 0.1))
        add(f"F. LDA{k} + 線形ridge λ=1", run_linear(Xl, y, evl, 1.0))
        cent = np.stack([Xl[y == c].mean(0) for c in range(14)])
        add(f"F. LDA{k} + 最近傍重心", {e: acc(-((Xe_[:, None, :] - cent[None]) ** 2).sum(2), ye) for e, (Xe_, ye, _) in evl.items()})
    # F2. PC 学習の線形前段 (ridge スコア k=14 / LDA13 拡張) → ELM (Stamp または CPU で射影, ODL は β のみ)
    if "F" in secs:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
        W = solve(Xn, onehot(y, 14), 10.0)
        proj = lambda Xe: np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1), np.float32)]) @ W
        Xs = proj(X); evs = {e: (proj(Xe), ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for m, al in ((32, ALPHA32[:14]), (64, rand_alpha(14, 64, 1))):
            for s in (0.5, 1.0, 2.0, 4.0):
                add(f"F2. 線形ridge14スコア前段 + ELM m={m} s={s}", run_elm(Xs, y, evs, al, s, 0.1))
        add("F2. 線形ridge14スコア前段 + 線形ridge (=2段線形)", run_linear(Xs, y, evs, 1.0))
        Wl = W[:-1]; V2 = np.linalg.svd(Wl, full_matrices=False)[0][:, :13]   # ridge 重みの列空間 (13 dims)
        Xr = ((X - mu) / sd) @ V2; evr = {e: (((Xe - mu) / sd) @ V2, ye, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for s in (0.5, 1.0, 2.0):
            add(f"F2. ridge重み部分空間13 + ELM m=32 Simα[:13] s={s}", run_elm(Xr, y, evr, ALPHA32[:13], s, 0.1))
    # G. dataset variants (m=32 Sim α, best s from A) and ensemble
    best_s = 0.25
    for v in a.compare_variants.split(",") if "G" in secs else ():
        if not (CACHE / f"solist_ds_{v}.npz").exists():
            continue
        dv = Data(v)
        add(f"G. variant `{v}` ELM m=32 Simα s={best_s}", run_elm(dv.X, dv.y14, d.ev, ALPHA32, best_s, 0.1))
        add(f"G. variant `{v}` 線形ridge λ=1", run_linear(dv.X, dv.y14, d.ev, 1.0))
    # J. 2-instance ensemble (score average) m=32 x2
    mu, sd = X.mean(0), X.std(0) + 1e-6
    res = {}
    for e, (Xe, ye, _) in (d.ev.items() if "J" in secs else ()):
        P = 0
        for al in (ALPHA32, rand_alpha(167, 32, 7)):
            beta = solve(hs(((X - mu) / sd * best_s) @ al), onehot(y, 14), 0.1)
            P = P + hs(((Xe - mu) / sd * best_s) @ al) @ beta
        res[e] = acc(P, ye)
    if "J" in secs:
        add(f"J. 2インスタンス集約 (m=32×2) s={best_s}", res)
    # I. few-shot on-site update: k frames per 32-state from the eval set's first frames, test on the rest
    for k in (1, 2, 3) if "I" in secs else ():
        for w in (1, 20):
            res = {}
            for e, (Xe, ye, y32) in d.ev.items():
                cal = np.concatenate([np.where(y32 == s_)[0][:k] for s_ in np.unique(y32)])
                rest = np.setdiff1d(np.arange(len(ye)), cal)
                Xtr2 = np.concatenate([X] + [Xe[cal]] * w); ytr2 = np.concatenate([y] + [ye[cal]] * w)
                r = run_elm(Xtr2, ytr2, {e: (Xe[rest], ye[rest], y32[rest])}, ALPHA32, best_s, 0.1)
                res[e] = r[e]
            add(f"I. 現地校正 {k}frame/状態 (重み×{w}) + ELM m=32 Simα s={best_s}", res)
        res = {}
        for e, (Xe, ye, y32) in d.ev.items():
            cal = np.concatenate([np.where(y32 == s_)[0][:k] for s_ in np.unique(y32)])
            rest = np.setdiff1d(np.arange(len(ye)), cal)
            res[e] = run_linear(np.concatenate([X] + [Xe[cal]] * 20), np.concatenate([y] + [ye[cal]] * 20), {e: (Xe[rest], ye[rest], y32[rest])}, 1.0)[e]
        add(f"I. 現地校正 {k}frame/状態 (重み×20) + 線形ridge", res)
    # K. 32cls で上位手法を確認
    if "K" in secs:
        ev32 = {e: (Xe, y32, y32) for e, (Xe, ye, y32) in d.ev.items()}
        add("K(32cls). ELM m=32 Simα s=0.5", run_elm(X, d.y32, ev32, ALPHA32, 0.5, 0.1, C=32))
        add("K(32cls). ELM m=64 乱数α s=0.25", run_elm(X, d.y32, ev32, rand_alpha(167, 64, 1), 0.25, 0.1, C=32))
        for lam in (1.0, 10.0):
            add(f"K(32cls). 線形ridge D=167 λ={lam}", run_linear(X, d.y32, ev32, lam, C=32))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)]); W = solve(Xn, onehot(d.y32, 32), 10.0)
        proj = lambda Xe: np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1), np.float32)]) @ W
        Xs = proj(X); evs = {e: (proj(Xe), y32, y32) for e, (Xe, ye, y32) in d.ev.items()}
        for s in (1.0, 2.0):
            add(f"K(32cls). 線形ridge32スコア前段 + ELM m=32 Simα[:32] s={s}", run_elm(Xs, d.y32, evs, ALPHA32[:32], s, 0.1, C=32))
    # L. 実機想定の最終形: PC 学習の線形前段 W (Stamp/CPU で計算, 固定) + ODL ELM (β のみ現地校正)
    if "L" in secs:
        def vote(P, y, y32):
            ok = tot = 0
            for st in np.unique(y32):
                idx = np.where(y32 == st)[0]; ok += int(P[idx].sum(0).argmax() == y[idx][0]); tot += 1
            return ok / tot
        for vname in ("unoq_ir2", "unoq+frdm_ir2"):
            if not (CACHE / f"solist_ds_{vname}.npz").exists():
                continue
            dv = Data(vname)
            for C, ycol, ev_y in ((14, "y14", 1), (32, "y32", 2)):
                Xv = dv.X; yv = getattr(dv, ycol)
                mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
                Xn = np.hstack([(Xv - mu) / sd, np.ones((len(Xv), 1), np.float32)])
                W = solve(Xn, onehot(yv, C), 10.0)
                proj = lambda Xe: np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1), np.float32)]) @ W
                Xs = proj(Xv)
                for m, al in ((32, ALPHA32[:C] if C <= 32 else rand_alpha(C, 32, 1)), (64, rand_alpha(C, 64, 1))):
                    for s in (1.0, 2.0, 4.0):
                        for k in (0, 1, 2):
                            res = {}; resv = {}
                            for e, (Xe, ye14, ye32) in d.ev.items():
                                ye = ye14 if C == 14 else ye32
                                Xe_s = proj(Xe)
                                if k > 0:
                                    cal = np.concatenate([np.where(ye32 == st)[0][:k] for st in np.unique(ye32)])
                                    rest = np.setdiff1d(np.arange(len(ye)), cal)
                                    if len(rest) == 0:
                                        res[e] = np.nan; resv[e] = np.nan; continue
                                    Xtr2 = np.concatenate([Xs] + [Xe_s[cal]] * 20); ytr2 = np.concatenate([yv] + [ye[cal]] * 20)
                                else:
                                    rest = np.arange(len(ye)); Xtr2, ytr2 = Xs, yv
                                H = hs((Xtr2 * s) @ al); beta = solve(H, onehot(ytr2, C), 0.1)
                                P = hs((Xe_s[rest] * s) @ al) @ beta
                                res[e] = acc(P, ye[rest]); resv[e] = vote(P, ye[rest], ye32[rest])
                            name = f"L({C}cls). `{vname}` ridge{C}前段 + ELM m={m} s={s} 校正{k}frame/状態"
                            add(name, res)
                            sv, mv = fmt(resv); lines.append(f"| ↳ 状態別投票 | {sv} |")
    lines += ["", f"実行時間 {time.time()-t0:.0f} s", "", "## 上位10", ""]
    for m, name in sorted(rows, reverse=True)[:10]:
        lines.append(f"- {m:.1%}  {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"IMPROVE_{secs}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
