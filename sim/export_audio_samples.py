"""実機試験の自己テストの元になった音声 (UNO Q eval evening) をリポジトリへ抜き出し、特徴計算を検証する。

抜き出すもの (samples/uno_q_eval_evening/):
  baseline/   起動時 baseline (全閉, meta group=="baseline") 10 frame
  sXXXXX/     各状態の先頭 (baseline を除く) 1 frame = firmware 自己テストの 32 ケース
  manifest.json  元ファイル・状態・クラス番号・収録時刻
検証: 抜き出した wav だけから N333 特徴を計算し、board_model_frontend_32cls.npz の標準化・量子化を掛けた int8 入力が
      firmware/IchiPingInference/generated/ichiping_model.h の自己テスト入力と一致するか確認する。

実行: python sim/export_audio_samples.py [--verify-only]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_data import BLO, BHI, UNOQ, load_wav, seg_power, to_db, norm_diff  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = UNOQ / "uno_q_eval_20260912_evening_wav"
DST = ROOT / "samples" / "uno_q_eval_evening"
HEADER = ROOT / "firmware" / "IchiPingInference" / "generated" / "ichiping_model.h"
MODEL = ROOT / "sim_export" / "solist_ds" / "board_model_frontend_32cls.npz"


def state_class(name: str) -> int:
    return sum(int(b) << k for k, b in enumerate(name[1:]))       # bit k = a b c AB BC


def export():
    if DST.exists():
        shutil.rmtree(DST)
    manifest = dict(source=str(SRC).replace("\\", "/"), sample_rate=16000, frame_seconds=2.0,
                    note="UNO Q (INMP441 / MAX98357A), PRBS 励振, 16 kHz mono int16。状態名は扉 a b c AB BC (1=開)。",
                    baseline=[], states=[])
    for d in sorted(p for p in SRC.iterdir() if p.is_dir() and re.fullmatch(r"s[01]{5}", p.name)):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        frames = [f for f in meta["frames"] if "wav" in f]
        if d.name == "s00000":
            for f in (f for f in frames if f.get("group") == "baseline"):
                (DST / "baseline").mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / f["wav"], DST / "baseline" / f["wav"])
                manifest["baseline"].append(dict(file=f"baseline/{f['wav']}", source=f"{d.name}/{f['wav']}", time=f.get("time")))
        first = next(f for f in frames if f.get("group") != "baseline")
        (DST / d.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / first["wav"], DST / d.name / first["wav"])
        manifest["states"].append(dict(state=d.name, class_id=state_class(d.name), file=f"{d.name}/{first['wav']}",
                                       source=f"{d.name}/{first['wav']}", time=first.get("time")))
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    n = len(manifest["baseline"]) + len(manifest["states"])
    print(f"exported {n} wav -> {DST}")


def header_cases() -> np.ndarray:
    h = HEADER.read_text(encoding="ascii")
    body = re.search(r"ichi_model_cases\[ICHI_MODEL_CASE_COUNT\]\[ICHI_FRONT_INPUT_SIZE\] = \{(.*?)\};", h, re.S).group(1)
    return np.array([int(v) for v in body.replace("\n", " ").split(",") if v.strip()], np.int8).reshape(32, -1)


def verify():
    manifest = json.loads((DST / "manifest.json").read_text(encoding="utf-8"))
    base_db = np.mean([to_db(seg_power(load_wav(DST / b["file"])).mean(0)) for b in manifest["baseline"]], axis=0)
    m = np.load(MODEL)
    cases = header_cases()
    ok = 0
    for s in sorted(manifest["states"], key=lambda s: s["class_id"]):
        # 学習・参照入力は float16 キャッシュ (eval_full_data.build_run) を経由しているので同じ丸めを通す
        x = norm_diff(to_db(seg_power(load_wav(DST / s["file"])).mean(0)), base_db).astype(np.float16)
        xs = (x[BLO:BHI].astype(np.float32) - m["in_mu"]) / m["in_sd"]
        q = np.clip(np.round(xs / m["s_in"]), -127, 127).astype(np.int8)
        same = np.array_equal(q, cases[s["class_id"]])
        ok += same
        if not same:
            d = np.abs(q.astype(int) - cases[s["class_id"]].astype(int))
            print(f"  {s['state']} (class {s['class_id']}): mismatch {int((d > 0).sum())} values, max |Δ| {d.max()}")
    print(f"feature check: {ok}/32 self-test inputs reproduced from samples/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    if not ap.parse_args().verify_only:
        export()
    verify()
