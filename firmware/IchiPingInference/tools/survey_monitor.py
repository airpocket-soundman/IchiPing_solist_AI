"""実機 32 状態サーベイ (build.ps1 -Main ichi_survey_main) の結果を UART で受信・集計する。

Solist がサーボで全閉 → baseline → 32 状態を順に作り、Stamp で PRBS 再生・録音した音から
その場で推論して結果を送る。本スクリプトは受信して正解率・状態別結果・混同を
docs/board_survey.{md,json} に書く。

UART フレーム (ichi_protocol.h 形式):
  0x40 SURVEY_START  rounds u8 | baseline u8 | states u8 | calibrate u8
  0x41 SURVEY_BASE   index u8 | exponent u8 | transfer ms u32
  0x42 SURVEY_RESULT state u8 | round u8 | pred u8 | pred_factory u8 | transfer ms u32 | feature ms u16 |
                     infer ms u16 | outputs float32 x 32   (pred: current beta, pred_factory: factory head)
  0x43 SURVEY_DONE   correct u16 | total u16 | aborted u8 | seconds u16 | correct_factory u16
  0x45 CAL_STEP      state u8 | window u8 | pred before update u8 | 0 | transfer ms u32
  0x46 CAL_DONE      correct-before u16 | samples u16 | ok u8 | seconds u16
  0x44 SURVEY_ERROR  stage u8 | state u8

usage: python firmware/IchiPingInference/tools/survey_monitor.py --port COM3 [--timeout 3600]
       (起動すると PC から 'G' を送ってサーベイを開始する。--wait-exec なら EXEC を押して開始)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from ichi_serial import decode_frame  # noqa: E402

START, BASE, RESULT, DONE, ERROR, CAL_STEP, CAL_DONE = 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46
STAGES = {1: "Stamp 応答なし", 2: "測定開始", 3: "測定待ち", 4: "PCM 読み出し", 5: "推論", 6: "サーボ", 7: "校正の学習"}


def label(c: int) -> str:
    """"h" + 0/1 in the TFT order C, BC, B, AB, A (1 = OPEN), e.g. h01101."""
    return "h" + "".join(str((c >> k) & 1) for k in (2, 4, 1, 3, 0))


def class14(c: int) -> int:
    a, b, cc, ab, bc = [(c >> k) & 1 for k in range(5)]
    if ab == 0:
        return 0 if a == 0 else 1
    if bc == 0:
        return 2 + a + 2 * b
    return 6 + a + 2 * b + 4 * cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "board_survey"))
    ap.add_argument("--wait-exec", action="store_true", help="PC から開始せず、EXEC が押されるのを待つ")
    a = ap.parse_args()
    s = serial.Serial(a.port, 115200, timeout=0.2)
    buf = bytearray()
    results, bases, info, done, cal_steps, cal_done = [], [], {}, None, [], None
    t0 = time.time()
    last_start = -1e9
    print("EXEC を押すとサーベイを開始します" if a.wait_exec else "PC からサーベイを開始します ('G' を送信)")
    try:
        while time.time() - t0 < a.timeout and done is None:
            if not a.wait_exec and not info and time.time() - last_start > 3.0:
                s.write(b"G")                                   # 開始要求 (START を受けるまで 3 秒ごとに再送)
                last_start = time.time()
            buf += s.read(4096)
            while b"\x00" in buf:
                chunk, _, rest = bytes(buf).partition(b"\x00")
                buf = bytearray(rest)
                fr = decode_frame(chunk) if chunk else None
                if fr is None:
                    continue
                p = fr.payload
                if fr.type == START:
                    info = dict(repeats=p[0], baseline=p[1], states=p[2], calibrate=bool(p[3]) if len(p) > 3 else False,
                                started=dt.datetime.now().isoformat(timespec="seconds"))
                    results.clear(); bases.clear(); cal_steps.clear()
                    print(f"開始: 状態 {p[2]} × {p[0]} 回, baseline {p[1]} frame, 校正 {'あり' if info['calibrate'] else 'なし'}")
                elif fr.type == BASE:
                    idx, exp_, ms = struct.unpack_from("<BBI", p)
                    bases.append(dict(index=idx, exponent=exp_, transfer_ms=ms))
                    print(f"  baseline {idx + 1}: transfer {ms} ms, exp {exp_}")
                elif fr.type == RESULT:
                    st, rep, pred, pfac, tr_ms, ft_ms, inf_ms = struct.unpack_from("<BBBBIHH", p)
                    out = np.frombuffer(p, "<f4", count=32, offset=12).tolist()
                    results.append(dict(state=st, rep=rep, pred=pred, pred_factory=pfac, transfer_ms=tr_ms,
                                        feature_ms=ft_ms, infer_ms=inf_ms, outputs=out))
                    mark = lambda v: "○" if v == st else ("△" if class14(v) == class14(st) else "×")
                    acc = np.mean([r["pred"] == r["state"] for r in results])
                    accf = np.mean([r["pred_factory"] == r["state"] for r in results])
                    print(f"  {label(st)}: 校正後 {label(pred)} {mark(pred)} / 工場 {label(pfac)} {mark(pfac)}"
                          f"   累積 校正後 {acc:.1%} / 工場 {accf:.1%} ({len(results)})", flush=True)
                elif fr.type == CAL_STEP:
                    st, w, pred, _, ms = struct.unpack_from("<BBBBI", p)
                    cal_steps.append(dict(state=st, window=w, pred_before=pred, transfer_ms=ms))
                    if w == 0:
                        print(f"  校正 {len(cal_steps) // 5 + 1:2d}/32 {label(st)} (転送 {ms} ms)", flush=True)
                elif fr.type == CAL_DONE:
                    c, n, ok, sec = struct.unpack_from("<HHBH", p)
                    cal_done = dict(correct_before=c, samples=n, ok=bool(ok), seconds=sec)
                    print(f"校正{'完了' if ok else '失敗/中断'}: 更新前の正解 {c}/{n}, {sec} 秒", flush=True)
                elif fr.type == DONE:
                    correct, total, aborted, secs = struct.unpack_from("<HHBH", p)
                    done = dict(correct=correct, total=total, aborted=bool(aborted), seconds=secs)
                    if len(p) >= 9:
                        done["correct_factory"] = struct.unpack_from("<H", p, 7)[0]
                elif fr.type == ERROR:
                    print(f"  エラー: {STAGES.get(p[0], p[0])} state {label(p[1])}")
    finally:
        s.close()
    if not results:
        print("結果なし"); return

    y = np.array([r["state"] for r in results]); pr = np.array([r["pred"] for r in results])
    pf = np.array([r.get("pred_factory", r["pred"]) for r in results])
    acc32 = float(np.mean(pr == y)); acc14 = float(np.mean([class14(p) == class14(t) for p, t in zip(pr, y)]))
    acc32f = float(np.mean(pf == y)); acc14f = float(np.mean([class14(p) == class14(t) for p, t in zip(pf, y)]))
    per_state = {label(c): dict(n=int((y == c).sum()), correct=int(((y == c) & (pr == c)).sum()),
                                correct_factory=int(((y == c) & (pf == c)).sum()),
                                preds=[label(int(v)) for v in pr[y == c]],
                                preds_factory=[label(int(v)) for v in pf[y == c]]) for c in range(32) if (y == c).any()}
    summary = dict(info=info, done=done, calibration=cal_done, calibration_steps=cal_steps,
                   accuracy32=acc32, accuracy14=acc14, accuracy32_factory=acc32f, accuracy14_factory=acc14f,
                   n=len(results), baseline=bases, per_state=per_state, results=results,
                   transfer_ms_mean=float(np.mean([r["transfer_ms"] for r in results])))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    cal = info.get("calibrate", False) or bool(cal_steps) or cal_done is not None
    L = [f"# 実機 32 状態サーベイ ({info.get('started', '')})", "",
         "Solist-AI 単体推論 (Stamp で PRBS 再生・INMP441 録音 → I2C → Solist で N333 特徴 → int8 CNN 前段 → ELM)。",
         f"状態はサーボで作り、各状態 {info.get('repeats', '?')} 回。baseline は開始時の全閉 {info.get('baseline', '?')} frame。"]
    if cal:
        L += ["現地校正あり: baseline の後、32 状態 × 6 秒 PRBS × 5 窓 (1 秒ずらし) で AxlCORE 上の ELM β を逐次学習 (OS-ELM)。",
              "同じ測定に対して、校正後の β (AxlCORE) と工場の β (CPU で同じ計算) の両方で推論した。",
              f"校正: 更新前の推論の正解 {cal_done['correct_before']}/{cal_done['samples']}, {cal_done['seconds']} 秒" if cal_done else "校正: 記録なし"]
    L += ["", "| モデル | 32 クラス | 14 クラス換算 |", "|---|---:|---:|",
          f"| {'校正後' if cal else '工場'} | **{acc32:.1%}** ({int((pr == y).sum())}/{len(y)}) | {acc14:.1%} |"]
    if cal:
        L += [f"| 工場 (同じ測定) | {acc32f:.1%} ({int((pf == y).sum())}/{len(y)}) | {acc14f:.1%} |"]
    L += ["", f"所要時間: {done['seconds'] if done else '-'} s, 1 frame の転送 平均 {summary['transfer_ms_mean']:.0f} ms", "",
          "| 状態 | 正解 | 予測 |" + (" 工場の予測 |" if cal else ""), "|---|---:|---|" + ("---|" if cal else "")]
    L += [f"| {k} | {v['correct']}/{v['n']} | {', '.join(v['preds'])} |" + (f" {', '.join(v['preds_factory'])} |" if cal else "")
          for k, v in per_state.items()]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n32 クラス {acc32:.1%} / 14 クラス換算 {acc14:.1%}"
          + (f"   (工場: {acc32f:.1%} / {acc14f:.1%})" if cal else "") + f"  -> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
