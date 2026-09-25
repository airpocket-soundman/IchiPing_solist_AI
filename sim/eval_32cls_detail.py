"""32 クラス分類の詳細成績 (Original 日単位 LODO)。

対象: 推奨構成 T8-16-32 FC32 前段 + ELM m32 (factory / 同日別 run 10 frame/class 混合校正)、
      比較用に T16-32-64 FC64 + ELM m64 と現行 ELM m32 Simα / D167。
指標: frame 精度, macro F1, 状態投票精度 (評価 run×状態ごとにスコア合算), 14cls 換算,
      fold 別, 状態別正解率, 主な混同ペア, 誤りのうち 14cls 等価内の割合。
プロトコルは eval_tiny_frontend_lodo.py / eval_improve_lodo.py と同じ (seed 0,1,2)。

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/eval_32cls_detail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ideal_vs_solist import FRDM_DAYS, ALPHA32  # noqa: E402
from eval_improve_lodo import DAYS, Elm, load, load_days, cal_split, rand_alpha, to14, macro_f1  # noqa: E402
from eval_tiny_frontend_lodo import ARCHS, SEEDS, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds"
C = 32


def label(c: int) -> str:
    return "s" + "".join(str((c >> k) & 1) for k in range(5))       # a b c AB BC


def evaluate(S, y, g):
    pred = S.argmax(1)
    votes = [int(S[(g == r) & (y == c)].sum(0).argmax() == c)
             for r in np.unique(g) for c in range(C) if np.any((g == r) & (y == c))]
    return {"frame": float(np.mean(pred == y)), "macro_f1": macro_f1(pred, y, C),
            "vote": float(np.mean(votes)), "as14": float(np.mean(to14(pred) == to14(y))),
            "pred": pred, "y": y}


def calibrated(h, Ze, ye, ge, runs):
    """同日の別 run で校正した β で、各評価 run を採点 (run 順序ペア全部)。"""
    out = []
    for rc in runs:
        hm = h.recal(Ze[cal_split(ye, ge, rc, None)], ye[cal_split(ye, ge, rc, None)], "mix")
        for rt in runs:
            if rt != rc:
                m = ge == rt
                out.append(evaluate(hm.scores(Ze[m]), ye[m], ge[m]))
    return out


def main():
    configs = {}
    arch_by = {a[0]: a for a in ARCHS}
    for arch_name, m in (("T8-16-32 FC32", 32), ("T16-32-64 FC64", 64)):
        for hold in DAYS:
            train_days = tuple(d for d in DAYS if d != hold)
            X, y, g = load_days(train_days, "N333")
            Xe, ye, ge = load(hold, "N333")
            for seed in SEEDS:
                net, embed, _ = train(arch_by[arch_name], X, y, g, C, seed)
                Z, Ze = embed(X), embed(Xe)
                h = Elm(rand_alpha(net.emb, m, seed + 1), 0.5, 1.0).fit(Z, y, C)
                base = f"{arch_name} + ELM m{m}"
                configs.setdefault(f"{base} (校正なし)", []).append((hold, [evaluate(h.scores(Ze), ye, ge)]))
                configs.setdefault(f"{base} (現地校正)", []).append((hold, calibrated(h, Ze, ye, ge, list(FRDM_DAYS[hold]))))
            print(arch_name, hold, flush=True)
    # 現行: ELM m32 公式Simα / D167 (eval_improve_lodo の内側 LODO 選択値を fold 毎に使用)
    sel = json.loads((OUT / "IMPROVE_LODO.json").read_text(encoding="utf-8"))["folds"]["32cls|ELM m32 Simα / D167 (現行)"]
    for f in sel:
        hold, (s, lam) = f["holdout"], f["params"]
        train_days = tuple(d for d in DAYS if d != hold)
        X, y, _ = load_days(train_days, "D167"); Xe, ye, ge = load(hold, "D167")
        h = Elm(ALPHA32, s, lam).fit(X, y, C)
        configs.setdefault("現行 ELM m32 Simα / D167 (校正なし)", []).append((hold, [evaluate(h.scores(Xe), ye, ge)]))
        configs.setdefault("現行 ELM m32 Simα / D167 (現地校正)", []).append((hold, calibrated(h, Xe, ye, ge, list(FRDM_DAYS[hold]))))
    report(configs)


def report(configs):
    p = lambda v: f"{v*100:.1f}%"
    summary, detail = [], {}
    L = ["# 32 クラス分類の詳細成績 (Original 日単位 LODO)", "",
         "外側 = 日単位 leave-one-day-out。前段構成は seed 0–2 の平均。現地校正 = 評価日の別 run 10 frame/class を"
         " factory と混合して β 再計算し、同日の別 run で評価 (run 順序ペア平均)。",
         "総合の各指標は fold 平均 (fold 内は seed・run ペア平均)。誤り内訳のみ全評価 frame を集計。",
         "状態投票 = 評価 run×状態ごとに全 frame のスコアを合算して判定 (1 run あたり 32 判定)。", "",
         "## 総合", "",
         "| 構成 | frame | macro F1 | 状態投票 | 14cls換算 | 誤りのうち14cls等価内 |", "|---|---:|---:|---:|---:|---:|"]
    for name, items in configs.items():
        ev = [e for _, es in items for e in es]
        y = np.concatenate([e["y"] for e in ev]); pr = np.concatenate([e["pred"] for e in ev])
        err = pr != y
        within = float(np.mean(to14(pr[err]) == to14(y[err]))) if err.any() else float("nan")
        fold_means = lambda k: [float(np.mean([e[k] for e in es])) for _, es in items]
        mean = lambda k: float(np.mean(fold_means(k)))           # fold 平均 (run 数の多い日に偏らない)
        L.append(f"| {name} | {p(mean('frame'))} | {p(mean('macro_f1'))} | {p(mean('vote'))} | {p(mean('as14'))} | {p(within)} |")
        folds = {}
        for hold, es in items:
            folds.setdefault(hold, []).extend(es)
        cls_acc = [float(np.mean(pr[y == c] == c)) for c in range(C)]
        conf = np.zeros((C, C), int)
        np.add.at(conf, (y, pr), 1)
        pairs = sorted(((conf[a, b] / conf[a].sum(), a, b) for a in range(C) for b in range(C) if a != b), reverse=True)[:8]
        detail[name] = {"folds": {h: {k: float(np.mean([e[k] for e in es])) for k in ("frame", "macro_f1", "vote", "as14")}
                                   for h, es in folds.items()},
                        "class_acc": cls_acc, "top_confusions": [(label(a), label(b), float(r)) for r, a, b in pairs],
                        "err_within14": within}
    L += ["", "## fold 別 (frame / macro F1 / 状態投票)", "",
          "| 構成 | " + " | ".join(DAYS) + " |", "|---|" + "---:|" * len(DAYS)]
    for name, d in detail.items():
        L.append(f"| {name} | " + " | ".join(
            f"{p(d['folds'][h]['frame'])} / {p(d['folds'][h]['macro_f1'])} / {p(d['folds'][h]['vote'])}" for h in DAYS) + " |")
    L += ["", "## 状態別 frame 正解率", "",
          "状態ラベルは a b c AB BC (1=開)。", "",
          "| 状態 | " + " | ".join(detail) + " |", "|---|" + "---:|" * len(detail)]
    for c in range(C):
        L.append(f"| {label(c)} | " + " | ".join(p(d["class_acc"][c]) for d in detail.values()) + " |")
    L += ["", "## 主な混同 (正解 → 予測: その正解状態の frame に占める割合)", ""]
    for name, d in detail.items():
        L.append(f"- **{name}**: " + ", ".join(f"{a}→{b} {r*100:.0f}%" for a, b, r in d["top_confusions"]))
    (OUT / "DETAIL_32CLS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "DETAIL_32CLS.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
