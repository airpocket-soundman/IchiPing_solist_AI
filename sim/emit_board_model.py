"""実機 (ML63Q2557 / AxlCORE-ODL) 向けモデル生成の共通関数 (bf16 変換、hard sigmoid、C 配列出力)。

emit_frontend_model.py / emit_calibration_prior.py / eval_odl_calibration.py などが import する。
- α: Solist-AI Sim seed=1 の実 α (sim_export/_alpha32_sim.npy, 167×32, |α|≤0.205)。
  実機では ODL_SetWeightAlpha は無効 (stub) で、seed=1・scaleAlpha=0x3E52 からアクセラレータが
  同じ α を再生成する (167 入力で実機と一致を確認済み)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "firmware" / "IchiPingInference" / "generated"
ALPHA = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float32)   # (167,32)
SCALE_ALPHA_BF16 = 0x3E52      # 0.205078125 (Sim seed=1 α の最大値と一致)

def to_bf16_bits(v):
    raw = np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32)
    return ((raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(b):
    return (np.asarray(b, np.uint32) << 16).view(np.float32)


def q(v):
    return from_bf16_bits(to_bf16_bits(v))


def hard_sigmoid(z):
    return np.clip(0.2 * z + 0.5, 0.0, 1.0)


def mcu_reference(x_bf16_float, alpha, beta):
    """bf16 境界 + float32 累積の参照 (acrylic_pan dummy_model_pipeline.mcu_reference と同じ)。"""
    h = q(hard_sigmoid(x_bf16_float.astype(np.float32) @ q(alpha)))
    return q(h @ q(beta))


def c_array(name, bits, decl=None, cols=12):
    flat = np.asarray(bits).reshape(-1)
    lines = [f"    " + ", ".join(f"0x{int(v):04X}" for v in flat[i:i + cols]) + "," for i in range(0, len(flat), cols)]
    return f"static const int16_t {decl or f'{name}[{len(flat)}]'} = {{\n" + "\n".join(lines) + "\n};\n"
