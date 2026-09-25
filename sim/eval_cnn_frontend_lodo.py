"""学習済み CNN 前段 + Solist ELM ヘッド (β 現地校正) の Original 日単位 LODO 評価。

構成: 1024-bin noise_diff_norm → Conv 3 層 (CNN XL と同じ) → Linear 3840→E (ボトルネック, ReLU)
      → [PC] Linear E→32  /  [Solist] ELM (入力 E, 乱数α m, hard sigmoid, β)
前段 (Conv + ボトルネック) は学習日だけで学習して凍結し、ELM β を factory / 現地校正で求める。
校正プロトコルは eval_improve_lodo.py と同じ (評価日の別 run から 10 frame/class、同日別 run で評価)。

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_cnn_frontend_lodo.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ideal_vs_solist import FRDM_DAYS, split_train_validation  # noqa: E402
from eval_improve_lodo import (DAYS, Elm, Linear, load, load_days, metrics, cal_split,  # noqa: E402
                               rand_alpha, to14)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
EMB = 64
SEEDS = (0, 1, 2)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(1, 32, 16, stride=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 8, stride=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(128 * 30, EMB), nn.ReLU())
        self.head = nn.Linear(EMB, C)

    def forward(self, x):
        return self.head(self.front(x))


def train_front(X, y, g, C, seed, epochs=100):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    tr, va = split_train_validation(y if C == 32 else y, g, seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32)[:, None, :])
    Xt, Xv = f(X[tr]), f(X[va]).to(DEV)
    yt, yv = torch.from_numpy(y[tr]), torch.from_numpy(y[va]).to(DEV)
    net = Net(C).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best, best_loss, stale = None, 1e9, 0
    g_ = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g_)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lossf(net(Xt[b].to(DEV)), yt[b].to(DEV)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), yv))
        if vl < best_loss - 1e-5:
            best_loss, stale = vl, 0
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    net.load_state_dict(best); net.eval()

    def embed(A):
        with torch.no_grad():
            Z = torch.cat([net.front(f(A[i:i + 1024]).to(DEV)).cpu() for i in range(0, len(A), 1024)])
            return Z.numpy().astype(np.float64)

    def logits(A):
        with torch.no_grad():
            return torch.cat([net(f(A[i:i + 1024]).to(DEV)).cpu() for i in range(0, len(A), 1024)]).numpy()
    return embed, logits


def main():
    res = {"emb": EMB, "seeds": SEEDS, "rows": {}}
    for C, lab in ((32, lambda v: v), (14, to14)):
        rows = {}
        for hold in DAYS:
            train = tuple(d for d in DAYS if d != hold)
            X, y32, g = load_days(train, "N1024")
            Xe, ye32, ge = load(hold, "N1024")
            y, ye = lab(y32), lab(ye32)
            runs = list(FRDM_DAYS[hold])
            for seed in SEEDS:
                embed, logits = train_front(X, y, g, C, seed)
                Z, Ze = embed(X), embed(Xe)
                heads = {
                    "PC CNN head (factory)": None,
                    "ELM m32 乱数α on emb": Elm(rand_alpha(EMB, 32, seed + 1), 0.5, 1.0).fit(Z, y, C),
                    "ELM m64 乱数α on emb": Elm(rand_alpha(EMB, 64, seed + 1), 0.5, 1.0).fit(Z, y, C),
                    "線形 ridge on emb": Linear(10.0).fit(Z, y, C),
                }
                for name, h in heads.items():
                    S = logits(Xe) if h is None else h.scores(Ze)
                    r = {"factory": metrics(S, ye, C, ye32)}
                    if h is not None:
                        for mode in ("cal_only", "mix"):
                            v = []
                            for rc in runs:
                                ci = cal_split(ye32, ge, rc, None)
                                hm = h.recal(Ze[ci], ye[ci], mode)
                                for rt in runs:
                                    if rt != rc:
                                        ti = np.where(ge == rt)[0]
                                        v.append(metrics(hm.scores(Ze[ti]), ye[ti], C, ye32[ti]))
                            r[mode] = {m: float(np.mean([q[m] for q in v])) for m in v[0]}
                    rows.setdefault(name, []).append(r)
                print(f"{C}cls {hold} seed{seed} CNN={metrics(logits(Xe), ye, C, ye32)['frame']:.3f}", flush=True)
        res["rows"][f"{C}cls"] = rows
    (OUT / "CNN_FRONTEND_LODO.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    p = lambda v: f"{v*100:.1f}%"
    L = ["# 学習済み CNN 前段 + Solist ELM ヘッド (Original 日単位 LODO)", "",
         f"前段 = CNN XL の Conv 3 層 + ボトルネック {EMB} 次元 (学習日のみで学習・凍結)。3 fold × seed {len(SEEDS)} 平均。",
         "ヘッドのハイパラは固定 (ELM s=0.5 λ=1, 線形 λ=10; 評価日で選択していない)。校正は評価日の別 run 10 frame/class → 同日別 run。",
         "CNN の early stopping は学習日内の frame 分割 (handoff と同じ) で、やや楽観側。", "",
         "| クラス | ヘッド | factory frame | factory macroF1 | →14cls換算 | 校正のみ | factory+校正 混合 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for C, rows in res["rows"].items():
        for name, rs in rows.items():
            m = lambda k, q: float(np.mean([r[k][q] for r in rs if k in r and q in r[k]])) if any(k in r and q in r[k] for r in rs) else float("nan")
            fmt = lambda v: "—" if v != v else p(v)
            L.append(f"| {C} | {name} | {fmt(m('factory','frame'))} | {fmt(m('factory','macro_f1'))} | "
                     f"{fmt(m('factory','as14'))} | {fmt(m('cal_only','frame'))} | {fmt(m('mix','frame'))} |")
    (OUT / "CNN_FRONTEND_LODO.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
