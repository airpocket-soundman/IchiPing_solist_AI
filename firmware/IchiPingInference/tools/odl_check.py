"""AxlCORE の実機学習 (ODL_StartTrain / OS-ELM) を PC の同じ計算と比べる (build.ps1 -Main ichi_odl_test_main)。

ファームは起動 3 秒後に「工場 beta + P0 を読み込み → 読み出し」「自己テストのケースで 1 回学習 → 読み出し」を
数回繰り返し、beta と P を UART で送る。本スクリプトは受信して、PC 上の OS-ELM (float, 同じ初期値・同じ入力)
と各ステップで比べる。

usage: python firmware/IchiPingInference/tools/odl_check.py --port COM3   (起動してから書き込む / リセットする)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "generated"
sys.path.insert(0, str(HERE))
from ichi_serial import decode_frame  # noqa: E402


def bf16(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def c_array(text, name):
    body = re.search(name + r"\[[^\]]*\](?:\[[^\]]*\])? = \{(.*?)\};", text, re.S).group(1)
    vals = [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
    return np.array([int(v, 16) if v.startswith("0x") else int(v) for v in vals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    model_h = (GEN / "ichiping_model.h").read_text(encoding="ascii")
    prior_h = (GEN / "ichi_calib_prior.h").read_text(encoding="ascii")
    m = int(re.search(r"#define ICHI_MODEL_HIDDEN_SIZE (\d+)", model_h).group(1))
    beta0 = bf16(c_array(model_h, "ichi_model_beta") & 0xFFFF).reshape(m, -1)
    p0 = bf16(c_array(prior_h, "ichi_calib_p0") & 0xFFFF).reshape(m, m)
    alpha = bf16(c_array(prior_h, "ichi_calib_alpha") & 0xFFFF).reshape(-1, m)

    s = serial.Serial(a.port, 115200, timeout=0.2)
    buf, rows, steps, done, t0, last_rx = bytearray(), {}, {}, False, time.time(), None
    print("受信待ち (Solist をリセットするか書き込むと 3 秒後に始まります)")
    while not done and time.time() - t0 < a.timeout:
        chunk = s.read(4096)
        if chunk:
            last_rx = time.time()
        elif last_rx is not None and time.time() - last_rx > 5.0:
            break                                   # DONE が壊れて届かなくても、受信が止まったら解析する
        buf += chunk
        while b"\x00" in buf:
            chunk, _, rest = bytes(buf).partition(b"\x00")
            buf = bytearray(rest)
            fr = decode_frame(chunk) if chunk else None
            if fr is None:
                continue
            p = fr.payload
            if fr.type == 0x60:
                rows[(p[0], p[1], p[2])] = bf16(np.frombuffer(p, "<u2", count=m, offset=3))
            elif fr.type == 0x61:
                steps[p[0]] = dict(case=p[1], cls=p[2], pred=p[3], out=np.frombuffer(p, "<f4", count=m, offset=4),
                                   x=bf16(np.frombuffer(p, "<u2", count=alpha.shape[0], offset=4 + 4 * m)))
            elif fr.type == 0x62:
                done = True
    s.close()
    if not rows:
        print("何も受信できませんでした"); return
    if not done:
        print(f"DONE は届きませんでした (受信行 {len(rows)}, ステップ {len(steps)})。届いた分で比べます")

    def mat(k, step):
        # 欠けた行は NaN (差の max には nanmax を使う)
        return np.stack([rows.get((k, step, r), np.full(m, np.nan)) for r in range(m)])

    beta, P = beta0.copy(), p0.copy()
    print(f"step 0 (読み込み直後): beta 差 max {np.abs(mat(0, 0) - beta0).max():.3g}, P 差 max {np.abs(mat(1, 0) - p0).max():.3g}"
          f"  (P trace 実機 {np.trace(mat(1, 0)):.3g} / PC {np.trace(p0):.3g})")
    for step in sorted(steps):
        st = steps[step]
        h = np.clip(0.2 * (st["x"] @ alpha) + 0.5, 0, 1)   # 実機が計算した ELM 入力を使う
        t = np.zeros(beta.shape[1]); t[st["cls"]] = 1
        Ph = P @ h
        P = P - np.outer(Ph, Ph) / (1.0 + h @ Ph)
        d_pc = np.outer(P @ h, t - h @ beta)
        beta = beta + d_pc
        bd, pd = mat(0, step), mat(1, step)
        y_pc = h @ beta
        print(f"step {step}: case {st['case']} (class {st['cls']})  推論 実機 {st['pred']} / PC {int(np.argmax(y_pc))}")
        print(f"    beta: 差 max {np.abs(bd - beta).max():.3g}, 1 回の変化量 max 実機 "
              f"{np.abs(bd - mat(0, step - 1)).max():.3g} / PC {np.abs(d_pc).max():.3g}")
        print(f"    P   : trace 実機 {np.trace(pd):.4g} / PC {np.trace(P):.4g}, 差 max {np.abs(pd - P).max():.3g}")
        print(f"    出力: 正解クラス 実機 {st['out'][st['cls']]:.3f} / PC {y_pc[st['cls']]:.3f}, "
              f"差 max {np.abs(st['out'] - y_pc).max():.3g}")
    print("\n実機の P が学習で減っていなければ P の更新が行われていない。beta の変化量が PC より桁違いに大きければ、"
          "P の解釈 (スケール・並び) が違う。")


if __name__ == "__main__":
    main()
