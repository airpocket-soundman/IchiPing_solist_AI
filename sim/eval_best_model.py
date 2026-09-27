"""手持ちの全データ (FRDM / UNO Q / このハード Stamp-S3A) から、このハードで最も汎化する事前学習モデルを探す。

評価 (時間帯ブロック hold-out): このハードのセッションを時間帯ブロックに分け、1 ブロックを丸ごと評価に回し
残りで学習する (隣の時間帯が学習に入らない分、セッション単位の leave-one-out より厳しい)。
  noon    2026-09-26 s1-s5   (11:45-13:40)
  evening 2026-09-26 s6-s9   (16:24-17:56)
  night   2026-09-26 s10-s11 (21:24-22:10, ε ≈ -0.15%)
  morning 2026-09-27 s1-s2   (07:12-08:20)
指標は実機と同じ int8 前段 + bf16 ELM 参照計算 (mcu_reference) の 32 クラス / 14 クラス換算。ブロック平均と最悪ブロック。
評価フレームは各セッション自身の baseline で差分 (実機の運用と同じ)。

学習の条件:
  --data     stamp | unoq (UNO Q train 8 + Stamp) | all (FRDM 全 run + UNO Q train 8 + Stamp)
  --shift    比例周波数シフト aug の最大 |ε| (0 = なし, 0.02 = ±2%)
  --xbase K  クロスベースライン: 学習フレームを自セッションの baseline に加え、同じ世代 (FRDM / UNO Q / Stamp) の
             学習側の別セッション K 本の baseline でも差分して増やす (baseline のばらつきへの頑健化)
  --arch     small (実機と同じ, int8 ≈42 KB) | b1 (≈174 KB)
  --stamp-rep Stamp の学習フレームを N 倍に複製 (他世代に対する比重)
前段の早期終了・ELM ハイパラは学習側 Stamp セッション 1 本 (最新) を検証に使う。

結果は sim_export/solist_ds/best_model/results.jsonl に 1 行ずつ追記する。
usage: python sim/eval_best_model.py --data unoq --shift 0.02 --xbase 0 --arch small --name base
       python sim/eval_best_model.py --report   (results.jsonl を BEST_MODEL.md にまとめる)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_full_data as efd  # noqa: E402
from eval_full_data import (CACHE, FRDM_RUNS, UNOQ_TRAIN, load_wav, norm_diff, run_path, seg_power,  # noqa: E402
                            state_frames, to_db)
from emit_frontend_model import ARCHS, ELM_HIDDEN, C, build_candidate, elm_input, emb_norm, mcu_reference  # noqa: E402
from eval_stamp_mix import weighted_beta  # noqa: E402
from eval_stamp_session import class14  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sim_export" / "solist_ds" / "best_model"
C14 = np.array([class14(c) for c in range(32)])
S26 = lambda i: f"stamp_20260926_s{i}_wav"
S27 = lambda i: f"stamp_20260927_s{i}_wav"
BLOCKS = {
    "noon": [S26(i) for i in range(1, 6)],
    "evening": [S26(i) for i in range(6, 10)],
    "night": [S26(10), S26(11)],
    "morning": [S27(1), S27(2)],
}
STAMP_ALL = [r for b in BLOCKS.values() for r in b]


def family(r: str) -> str:
    return "stamp" if r.startswith("stamp_") else ("unoq" if r.startswith("uno_q") else "frdm")


def run_db(name: str) -> dict:
    """フレーム毎の dB スペクトル (1024 bin, baseline 差分前) と baseline dB。キャッシュする。"""
    cp = CACHE / f"db_{name}.npz"
    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    sf = state_frames(run_path(name))
    db, y, isb = [], [], []
    for c, ws, bs in sf:
        for w, b in zip(ws, bs):
            db.append(to_db(seg_power(load_wav(w)).mean(0)).astype(np.float32)); y.append(c); isb.append(b)
    db, y, isb = np.stack(db), np.array(y), np.array(isb)
    d = {"db": db[~isb], "y": y[~isb], "base": db[isb].mean(0)}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cp, **d)
    print(f"  cached dB {name}: {d['db'].shape}", flush=True)
    return d


def feats(d: dict, base: np.ndarray) -> np.ndarray:
    return norm_diff(d["db"], base).astype(np.float16)


def build_train(runs, xbase, rng, stamp_rep=1):
    X, y = [], []
    for r in runs:
        d = run_db(r)
        bases = [d["base"]]
        if xbase > 0:
            others = [o for o in runs if o != r and family(o) == family(r)]
            for o in rng.choice(others, size=min(xbase, len(others)), replace=False) if others else []:
                bases.append(run_db(str(o))["base"])
        rep = stamp_rep if family(r) == "stamp" else 1
        for b in bases:
            for _ in range(rep):
                X.append(feats(d, b)); y.append(d["y"])
    return np.concatenate(X), np.concatenate(y)


def score(P, y):
    p = P.argmax(1)
    return float(np.mean(p == y)), float(np.mean(C14[p] == C14[y]))


def run_config(a):
    efd.SHIFT_MAX = a.shift
    spec, emb_dim = ARCHS[a.arch]
    pre = {"stamp": [], "unoq": UNOQ_TRAIN, "all": FRDM_RUNS + UNOQ_TRAIN}[a.data]
    res = {}
    t0 = time.time()
    for blk, held in BLOCKS.items():
        rng = np.random.default_rng(0)
        train_stamp = [r for r in STAMP_ALL if r not in held]
        val = train_stamp[-1]                                   # 学習側で最新の Stamp セッション
        tr_runs = pre + train_stamp
        fit_runs = [r for r in tr_runs if r != val]
        X, y = build_train(tr_runs, a.xbase, rng, a.stamp_rep)
        Xf, yf = build_train(fit_runs, a.xbase, np.random.default_rng(0), a.stamp_rep)
        dv = run_db(val)
        Xv, yv = feats(dv, dv["base"]), dv["y"]
        cands = [build_candidate(spec, emb_dim, X, Xf, yf, Xv, yv, sd)
                 for sd in range(a.seed_offset, a.seed_offset + max(a.seeds, a.ensemble))]
        if a.ensemble > 1:
            # 前段 K 個 (seed 違い) の埋め込みを並べて 1 つの ELM (入力 32K ≤ 167) に入れる
            ms = cands[:a.ensemble]
            emb = lambda A: np.concatenate([q["embed"](A) for q in ms], axis=1)
            E_fit, E_val = emb(Xf), emb(Xv)
            best = None
            for s_elm in (0.1, 0.15, 0.25, 0.5):
                mul, add = emb_norm(E_fit, s_elm)
                for lam in (0.1, 1.0, 10.0):
                    b = weighted_beta([elm_input(E_fit, mul, add)], [yf], [1.0], lam)
                    acc = float(np.mean(mcu_reference(elm_input(E_val, mul, add), b).argmax(1) == yv))
                    if best is None or acc > best[0]:
                        best = (acc, s_elm, lam)
            val_acc, s_elm, lam = best
        else:
            m = max(cands, key=lambda q: q["val_acc"])
            emb = m["embed"]
            val_acc, s_elm, lam = m["val_acc"], m["s_elm"], m["lam"]
        E = emb(X)
        mul, add = emb_norm(E, s_elm)
        beta = weighted_beta([elm_input(E, mul, add)], [y], [1.0], lam)
        per = {}
        for r in held:
            d = run_db(r)
            Xe = feats(d, d["base"]).astype(np.float32)
            per[r] = score(mcu_reference(elm_input(emb(Xe), mul, add), beta), d["y"])
        res[blk] = dict(acc32=float(np.mean([v[0] for v in per.values()])),
                        acc14=float(np.mean([v[1] for v in per.values()])), sessions=per,
                        train_frames=int(len(y)), val=val, val_acc=val_acc)
        print(f"  [{a.name}] hold-out {blk:8s} 32cls={res[blk]['acc32']:.3f} 14cls={res[blk]['acc14']:.3f} "
              f"(train {len(y)} frames, {time.time() - t0:.0f} s)", flush=True)
    a32 = [v["acc32"] for v in res.values()]
    a14 = [v["acc14"] for v in res.values()]
    row = dict(name=a.name, data=a.data, shift=a.shift, xbase=a.xbase, arch=a.arch, seeds=a.seeds, seed_offset=a.seed_offset, ensemble=a.ensemble,
               stamp_rep=a.stamp_rep, mean32=float(np.mean(a32)), worst32=float(np.min(a32)),
               mean14=float(np.mean(a14)), worst14=float(np.min(a14)), blocks=res,
               seconds=round(time.time() - t0))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{a.name}] mean 32cls {row['mean32']:.3f} worst {row['worst32']:.3f} | 14cls {row['mean14']:.3f}", flush=True)


def report():
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    L = ["# このハードで最も汎化する事前学習モデルの探索", "",
         "時間帯ブロック hold-out (1 ブロックを丸ごと評価、残りで学習)。int8 前段 + bf16 ELM (実機と同じ参照計算)。",
         "ブロック: 昼 = 09-26 s1-s5, 夕方 = s6-s9, 夜 = s10-s11, 翌朝 = 09-27 s1-s2。", "",
         "| 名前 | データ | シフト | クロスBL | 前段 | Stamp比重 | 昼 | 夕方 | 夜 | 翌朝 | **平均 32cls** | 最悪 | 平均 14cls |",
         "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    dname = {"stamp": "Stamp のみ", "unoq": "UNO Q+Stamp", "all": "FRDM+UNO Q+Stamp"}
    for r in rows:
        b = r["blocks"]
        L.append(f"| {r['name']} | {dname[r['data']]} | ±{r['shift'] * 100:g}% | {r['xbase']} | {r['arch']} | ×{r['stamp_rep']} | "
                 + " | ".join(f"{b[k]['acc32'] * 100:.1f}" for k in BLOCKS)
                 + f" | **{r['mean32'] * 100:.1f}%** | {r['worst32'] * 100:.1f}% | {r['mean14'] * 100:.1f}% |")
    groups = {}
    for r in rows:
        k = (r["data"], r["shift"], r["xbase"], r["arch"], r["stamp_rep"], r["seeds"], r.get("ensemble", 1))
        groups.setdefault(k, []).append(r)
    L += ["", "## 同じ構成の繰り返し (seed の開始番号違い) の平均", "",
          "| データ | シフト | クロスBL | 前段 | Stamp比重 | seed数 | アンサンブル | 回数 | 昼 | 夕方 | 夜 | 翌朝 | **平均 32cls** "
          "| 最悪ブロック平均 | 回ごとの平均 32cls |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, g in sorted(groups.items(), key=lambda kv: -np.mean([r["mean32"] for r in kv[1]])):
        blk = {b: np.mean([r["blocks"][b]["acc32"] for r in g]) for b in BLOCKS}
        L.append(f"| {dname[k[0]]} | ±{k[1] * 100:g}% | {k[2]} | {k[3]} | ×{k[4]} | {k[5]} | {k[6]} | {len(g)} | "
                 + " | ".join(f"{blk[b] * 100:.1f}" for b in BLOCKS)
                 + f" | **{np.mean([r['mean32'] for r in g]) * 100:.1f}%** | {np.mean([r['worst32'] for r in g]) * 100:.1f}% | "
                 + ", ".join(f"{r['mean32'] * 100:.1f}" for r in g) + " |")
    (OUT.parent / "BEST_MODEL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="unoq", choices=("stamp", "unoq", "all"))
    ap.add_argument("--shift", type=float, default=0.02)
    ap.add_argument("--xbase", type=int, default=0)
    ap.add_argument("--arch", default="small", choices=tuple(ARCHS))
    ap.add_argument("--stamp-rep", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1, help="前段の学習 seed 数 (検証精度最大を採用)")
    ap.add_argument("--seed-offset", type=int, default=0, help="seed の開始番号 (同じ構成の繰り返し評価用)")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="前段を K 個 (seed 違い) 学習し、埋め込みを並べて 1 つの ELM に入れる (K x 32 <= 167)")
    ap.add_argument("--name", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    a.name = a.name or f"{a.data}_s{a.shift:g}_x{a.xbase}_{a.arch}_r{a.stamp_rep}"
    run_config(a)


if __name__ == "__main__":
    main()
