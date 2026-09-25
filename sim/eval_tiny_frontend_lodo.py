"""ML63Q2557 単体に載る軽量 CNN 前段 + Solist ELM ヘッドの Original 日単位 LODO 評価。

前提構成: Stamp-S3A は I2S スピーカ出力 (PRBS 再生) のみ。ML63Q2557 が FFT(HW) → log-PSD 差分
(noise_diff_norm, 400–3000 Hz 333 bin) → 畳み込み前段 (M0+ CPU, int8 想定) → ELM ヘッド + β 現地校正 (AxlCORE)。
制約: Flash 256KB / SRAM 16KB (AI RAM m32≈9.6KB を含む) / Cortex-M0+ 48MHz FPU 無。

各前段について パラメータ数 (int8 Flash), ピーク活性化 (int8, 層の入力+出力の最大), MAC 数を併記する。
学習・校正プロトコルは eval_cnn_frontend_lodo.py と同じ。

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_tiny_frontend_lodo.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ideal_vs_solist import FRDM_DAYS, split_train_validation  # noqa: E402
from eval_improve_lodo import DAYS, Elm, load, load_days, metrics, cal_split, rand_alpha, to14  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
SEEDS = (0, 1, 2)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_IN = 333

# (name, conv 層 [(out_ch, kernel, stride)], 埋め込み次元 E, flatten→E の全結合を使うか)
ARCHS = [
    ("T8-16-32 FC32",  [(8, 9, 2), (16, 7, 2), (32, 5, 2)], 32, True),
    ("T8-16-32 GAP",   [(8, 9, 2), (16, 7, 2), (32, 5, 2)], 32, False),
    ("T16-32-32 s4 FC32", [(16, 9, 4), (32, 7, 2), (32, 5, 2)], 32, True),
    ("T8-16 s4 FC32",  [(8, 9, 4), (16, 7, 4)], 32, True),
    ("T16-32-64 FC64", [(16, 9, 2), (32, 7, 2), (64, 5, 2)], 64, True),
]


def out_len(L, k, s):
    return (L - k) // s + 1


class Tiny(nn.Module):
    def __init__(self, convs, emb, fc, C):
        super().__init__()
        layers, ch, L = [], 1, D_IN
        self.cost = {"params": 0, "macs": 0, "peak_act": D_IN}
        for oc, k, s in convs:
            layers += [nn.Conv1d(ch, oc, k, stride=s), nn.BatchNorm1d(oc), nn.ReLU()]
            Lo = out_len(L, k, s)
            self.cost["params"] += ch * oc * k + oc            # BN は推論時に畳み込みへ融合
            self.cost["macs"] += oc * Lo * ch * k
            self.cost["peak_act"] = max(self.cost["peak_act"], ch * L + oc * Lo)
            ch, L = oc, Lo
        if fc:
            layers += [nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch * L, emb), nn.ReLU()]
            self.cost["params"] += ch * L * emb + emb
            self.cost["macs"] += ch * L * emb
            self.cost["peak_act"] = max(self.cost["peak_act"], ch * L + emb)
        else:
            layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
            emb = ch
        self.front = nn.Sequential(*layers)
        self.head = nn.Linear(emb, C)
        self.emb = emb

    def forward(self, x):
        return self.head(self.front(x))


def train(arch, X, y, g, C, seed, epochs=150):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    tr, va = split_train_validation(y, g, seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32)[:, None, :])
    Xt, Xv = f(X[tr]), f(X[va]).to(DEV)
    yt, yv = torch.from_numpy(y[tr]), torch.from_numpy(y[va]).to(DEV)
    net = Tiny(*arch[1:], C).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best, best_loss, stale = None, 1e9, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
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
            if stale >= 15:
                break
    net.load_state_dict(best); net.eval()

    def run(A, fn):
        with torch.no_grad():
            return torch.cat([fn(f(A[i:i + 2048]).to(DEV)).cpu() for i in range(0, len(A), 2048)]).numpy().astype(np.float64)
    return net, (lambda A: run(A, net.front)), (lambda A: run(A, net))


def main():
    res = {"seeds": SEEDS, "archs": {}}
    for arch in ARCHS:
        cost = Tiny(*arch[1:], 32).cost
        entry = {"cost": cost, "rows": {}}
        for C, lab in ((32, lambda v: v), (14, to14)):
            rows = {}
            for hold in DAYS:
                train_days = tuple(d for d in DAYS if d != hold)
                X, y32, g = load_days(train_days, "N333")
                Xe, ye32, ge = load(hold, "N333")
                y, ye = lab(y32), lab(ye32)
                runs = list(FRDM_DAYS[hold])
                for seed in SEEDS:
                    net, embed, logits = train(arch, X, y, g, C, seed)
                    Z, Ze = embed(X), embed(Xe)
                    heads = {"CNN head": None,
                             "ELM m32": Elm(rand_alpha(net.emb, 32, seed + 1), 0.5, 1.0).fit(Z, y, C),
                             "ELM m64": Elm(rand_alpha(net.emb, 64, seed + 1), 0.5, 1.0).fit(Z, y, C)}
                    for name, h in heads.items():
                        S = logits(Xe) if h is None else h.scores(Ze)
                        r = {"factory": metrics(S, ye, C, ye32)}
                        if h is not None:
                            v = []
                            for rc in runs:
                                ci = cal_split(ye32, ge, rc, None)
                                hm = h.recal(Ze[ci], ye[ci], "mix")
                                v += [metrics(hm.scores(Ze[ge == rt]), ye[ge == rt], C, ye32[ge == rt])
                                      for rt in runs if rt != rc]
                            r["mix"] = {m: float(np.mean([q[m] for q in v])) for m in v[0]}
                        rows.setdefault(name, []).append(r)
            entry["rows"][f"{C}cls"] = rows
            m = lambda n, k: np.mean([r[k]["frame"] for r in rows[n]])
            print(f"{arch[0]} {C}cls CNN={m('CNN head','factory'):.3f} ELM32={m('ELM m32','factory'):.3f}"
                  f"→cal {m('ELM m32','mix'):.3f} ELM64={m('ELM m64','factory'):.3f}→cal {m('ELM m64','mix'):.3f} {cost}", flush=True)
        res["archs"][arch[0]] = entry
    (OUT / "TINY_FRONTEND_LODO.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)


def write_md(res):
    p = lambda v: f"{v*100:.1f}%"
    L = ["# ML63Q2557 単体向け 軽量 CNN 前段 + Solist ELM ヘッド (Original 日単位 LODO)", "",
         "入力 = noise_diff_norm 400–3000 Hz 333 bin。前段は学習日のみで学習・凍結。3 fold × seed 3 平均。",
         "ELM は乱数α (s=0.5, λ=1 固定)。校正 = 評価日の別 run 10 frame/class を factory と混合して β 再計算 → 同日別 run。",
         "Flash/活性化は int8 想定 (BN は畳み込みに融合)。M0+ 時間は 1 MAC ≈ 10 cycle @48MHz の目安。", "",
         "| 前段 | Flash (int8) | ピーク活性化 | MAC | M0+目安 | 32cls CNN | 32cls ELM m32 → 校正 | 32cls ELM m64 → 校正 | 14cls ELM m32 → 校正 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in res["archs"].items():
        c = e["cost"]
        m = lambda C, h, k: float(np.mean([r[k]["frame"] for r in e["rows"][C][h]]))
        L.append(f"| {name} | {c['params']/1024:.1f} KB | {c['peak_act']/1024:.2f} KB | {c['macs']/1e3:.0f}k | "
                 f"{c['macs']*10/48e6*1e3:.0f} ms | {p(m('32cls','CNN head','factory'))} | "
                 f"{p(m('32cls','ELM m32','factory'))} → {p(m('32cls','ELM m32','mix'))} | "
                 f"{p(m('32cls','ELM m64','factory'))} → {p(m('32cls','ELM m64','mix'))} | "
                 f"{p(m('14cls','ELM m32','factory'))} → {p(m('14cls','ELM m32','mix'))} |")
    L += ["", "参考 (CNN_FRONTEND_LODO.md): CNN XL 前段 (1024 bin, 約30万 param, 2.3M MAC) + ELM m64 = 88.3% → 校正 94.2%。",
          "現行 ELM m32 / D167 = 66.3% → 校正 69.8% (IMPROVE_LODO.md)。"]
    (OUT / "TINY_FRONTEND_LODO.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
