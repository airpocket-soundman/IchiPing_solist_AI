"""実機アクセラレータが seed/scaleAlpha から生成する α を、単位ベクトル入力で直接読み出す。

前提: `python sim/emit_board_model.py --probe <ni> --hidden <m>` で生成したファーム (β=単位行列, 出力=hidden)
       を書き込んであること。hidden の hard_sigmoid 出力 h_j = clip(0.2·z_j + 0.5) がそのまま返る。
方法: x = ±c·e_i を AI_INFER で送り、α_ij = (h_j(+c) − h_j(−c)) / (0.4·c)。|0.2·c·α| < 0.5 となる c を使う。
出力: sim_export/alpha_probe/alpha_ni<ni>_m<m>.npz (alpha, ni, m, c) と Sim 採取 α との比較。

usage: C:/ProgramData/anaconda3/python.exe firmware/IchiPingInference/tools/probe_alpha.py --port COM3 --ni 167 --m 32 [--c 8]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from board_test import Board, AI_INFER  # noqa: E402
sys.path.insert(0, r"D:/GitHub/acrylic_pan/pc")
from acrylic_pan_monitor import protocol as P  # noqa: E402


def to_bf16_bits(v):
    raw = np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32)
    return ((raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--ni", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--c", type=float, default=8.0)
    a = ap.parse_args()
    fmt = struct.Struct(f"<BBH{a.m}f")
    b = Board(a.port)

    def infer(x):
        fr = b.request(AI_INFER, to_bf16_bits(x).astype("<u2").tobytes())
        if int(fr.message_type) != P.MessageType.AI_RESULT or len(fr.payload) != fmt.size:
            raise RuntimeError(f"unexpected reply type=0x{int(fr.message_type):02X} len={len(fr.payload)} (expect {fmt.size})")
        return np.array(fmt.unpack(fr.payload)[3:], np.float32)

    try:
        h0 = infer(np.zeros(a.ni, np.float32))
        print(f"zero input -> hidden mean={h0.mean():.4f} (expect 0.5), min={h0.min():.4f} max={h0.max():.4f}")
        alpha = np.zeros((a.ni, a.m), np.float32)
        sat = 0
        for i in range(a.ni):
            x = np.zeros(a.ni, np.float32); x[i] = a.c
            hp = infer(x); hm = infer(-x)
            sat += int(((hp <= 0) | (hp >= 1) | (hm <= 0) | (hm >= 1)).any())
            alpha[i] = (hp - hm) / (0.4 * a.c)
            if (i + 1) % 20 == 0:
                print(f"  probed {i+1}/{a.ni}", flush=True)
    finally:
        b.close()
    out = ROOT / "sim_export" / "alpha_probe"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"alpha_ni{a.ni}_m{a.m}.npz", alpha=alpha, ni=a.ni, m=a.m, c=a.c, h0=h0)
    print(f"alpha probed: shape={alpha.shape} max|α|={np.abs(alpha).max():.4f} mean|α|={np.abs(alpha).mean():.4f} saturated probes={sat}")
    cap = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float32)
    k = min(a.ni, cap.shape[0]); mm = min(a.m, cap.shape[1])
    d = np.abs(alpha[:k, :mm] - cap[:k, :mm])
    print(f"vs Sim capture rows[:{k}] cols[:{mm}]: max|Δ|={d.max():.4f} mean|Δ|={d.mean():.4f} corr={np.corrcoef(alpha[:k,:mm].ravel(), cap[:k,:mm].ravel())[0,1]:.4f}")
    # generator-order hypotheses
    stream_r = cap.flatten(); stream_c = cap.T.flatten()
    for name, cand in (("capture row-major stream -> col-major fill", stream_r[:a.ni*mm].reshape(mm, a.ni).T if a.ni*mm <= stream_r.size else None),
                       ("capture col-major stream -> row-major fill", stream_c[:a.ni*mm].reshape(a.ni, mm) if a.ni*mm <= stream_c.size else None),
                       ("capture col-major stream -> col-major fill", stream_c[:a.ni*mm].reshape(mm, a.ni).T if a.ni*mm <= stream_c.size else None)):
        if cand is not None:
            print(f"  {name}: corr={np.corrcoef(alpha[:, :mm].ravel(), cand.ravel())[0,1]:.4f}")
    print(f"-> {out / f'alpha_ni{a.ni}_m{a.m}.npz'}")


if __name__ == "__main__":
    main()
