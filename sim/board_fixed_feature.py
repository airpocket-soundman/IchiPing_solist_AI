"""ML63Q2557 上で N333 特徴を計算する固定小数点パイプラインの PC 参照実装 (firmware と同じ整数演算)。

firmware/IchiPingInference/src/ichi_feature.c と 1 対 1 に対応させる:
  PCM16 2 s (32000) → 2048 点 Hann (Q15) 窓, hop 1024 の 30 区間
  → 実 FFT 2048 = 複素 FFT 1024 (int16 ブロック浮動小数点, 段毎に条件付き 1/2) + 分離
  → パワー (float32) を 1024 bin 積算 → 平均 → dB (max(10log10(P+1e-12), -80))
  → baseline dB (int16, 1/256 dB) との差 → 1024 bin の平均・標準偏差で正規化 → float16 丸め
  → bin 50..383 (334) を標準化・int8 量子化 (board_model_frontend_32cls.npz の in_mu/in_sd/s_in)

実行: python sim/board_fixed_feature.py [--emit]
  samples/uno_q_eval_evening で自己テスト入力と比較する。--emit で次も書き出す:
  firmware/IchiPingInference/generated/ichi_feature_tables.h   Q15 窓・余弦表と入力標準化 (C 用)
  firmware/IchiPingInference/generated/pipeline_expected.json  クリップ毎の期待値 (tools/pipeline_monitor.py 用)
"""
from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "uno_q_eval_evening"
MODEL = ROOT / "sim_export" / "solist_ds" / "board_model_frontend_32cls.npz"
GEN = ROOT / "firmware" / "IchiPingInference" / "generated"
HEADER = GEN / "ichiping_model.h"

NFFT, HOP, NC = 2048, 1024, 1024          # real FFT length, hop, complex FFT length
BLO, BHI = 50, 384
BASE_Q = 256.0                            # baseline dB fixed point (1/256 dB)
PEAK_S0, PEAK_S1 = 13573, 27146           # 32767 / (1 + sqrt 2) and x2: stage shift 0 / 1 / 2 bits

# Q15 tables (firmware: const int16_t in flash)
WIN_Q15 = np.round(np.hanning(NFFT) * 32767.0).astype(np.int64)[:NFFT // 2]          # symmetric half
COS_Q15 = np.round(np.cos(2 * np.pi * np.arange(NFFT // 2) / NFFT) * 32767.0).astype(np.int64)   # cos(2πk/2048), k<1024


def cos_q(k: int) -> int:      # cos(2π k / 2048), k in [0, 2048)
    k %= NFFT
    if k < NFFT // 2:
        return int(COS_Q15[k])
    return -int(COS_Q15[k - NFFT // 2])


def sin_q(k: int) -> int:      # sin(2π k / 2048) = cos(2π (k - 512) / 2048)
    return cos_q(k - NFFT // 4)


def rshift_round(v, s: int):
    return (v + (1 << (s - 1))) >> s if s > 0 else v


BITREV = np.array([int(f"{i:010b}"[::-1], 2) for i in range(NC)])
COS_FULL = np.array([cos_q(k) for k in range(NFFT)], np.int64)
SIN_FULL = np.array([sin_q(k) for k in range(NFFT)], np.int64)
WIN_FULL = np.concatenate([WIN_Q15, WIN_Q15[::-1]])
STAGES = []
_half = 1
while _half < NC:
    _k = np.arange(NC // 2) % _half
    _grp = np.arange(NC // 2) // _half
    _a = _grp * 2 * _half + _k
    STAGES.append((_a, _a + _half, COS_FULL[_k * (NC // _half)], SIN_FULL[_k * (NC // _half)]))
    _half *= 2


def fft1024_bfp(re: np.ndarray, im: np.ndarray):
    """Radix-2 DIT complex FFT on int16 data with block floating point (vectorized, integer exact).
    Stage: the butterfly outputs are shifted right by 0/1/2 bits (round half up) by the stage peak. Returns (re, im, exp)."""
    re = re[BITREV].astype(np.int64)
    im = im[BITREV].astype(np.int64)
    exp = 0
    for a, b, c, sn in STAGES:
        peak = max(int(np.abs(re).max()), int(np.abs(im).max()))
        s = 0 if peak < PEAK_S0 else (1 if peak < PEAK_S1 else 2)   # |a| + sqrt2|b| must stay in int16
        exp += s
        tr = (re[b] * c + im[b] * sn + (1 << 14)) >> 15
        ti = (im[b] * c - re[b] * sn + (1 << 14)) >> 15
        ar, ai = re[a], im[a]
        re[a], im[a] = rshift_round(ar + tr, s), rshift_round(ai + ti, s)
        re[b], im[b] = rshift_round(ar - tr, s), rshift_round(ai - ti, s)
        assert max(int(np.abs(re).max()), int(np.abs(im).max())) <= 32767, "int16 overflow"
    return re, im, exp


def segment_power(seg: np.ndarray) -> np.ndarray:
    """int16[2048] -> float32[1024] power of bins 1..1024 (same scale as PC: samples/32768, float Hann)."""
    xw = (seg.astype(np.int64) * WIN_FULL + (1 << 14)) >> 15
    re, im, exp = fft1024_bfp(xw[0::2], xw[1::2])
    k = np.arange(1, NC + 1)
    kk, nk = k % NC, (NC - k) % NC
    ar, ai, br, bi = re[kk], im[kk], re[nk], -im[nk]
    er, ei, dr, di = ar + br, ai + bi, ar - br, ai - bi
    c, sn = COS_FULL[k], SIN_FULL[k]
    orr, oi = di, -dr
    wr = (orr * c + oi * sn + (1 << 14)) >> 15
    wi = (oi * c - orr * sn + (1 << 14)) >> 15
    xr = (er + wr).astype(np.float32)
    xi = (ei + wi).astype(np.float32)
    scale = np.float32(2.0 ** (2 * exp) / (32768.0 ** 2))
    return ((xr * xr + xi * xi) * np.float32(0.25)) * scale


def frame_db(pcm: np.ndarray) -> np.ndarray:
    acc = np.zeros(NC, np.float32)
    nseg = 0
    for s in range(0, len(pcm) - NFFT + 1, HOP):
        acc += segment_power(pcm[s:s + NFFT])
        nseg += 1
    mean = acc / np.float32(nseg)
    return np.maximum(np.float32(10.0) * np.log10(mean + np.float32(1e-12)), np.float32(-80.0)).astype(np.float32)


def load(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def baseline_q(dbs: list[np.ndarray]) -> np.ndarray:
    """int16 baseline: every baseline frame adds round_half_even(dB * 256 / n) (no extra RAM on the MCU)."""
    base = np.zeros(NC, np.int32)
    n = np.float32(len(dbs))
    for db in dbs:
        base += np.round((db * np.float32(BASE_Q)) / n).astype(np.int32)
    assert np.abs(base).max() <= 32767
    return base.astype(np.int16)


def features_int8(db: np.ndarray, base_q: np.ndarray, m) -> np.ndarray:
    d = db - base_q.astype(np.float32) / np.float32(BASE_Q)
    d64 = d.astype(np.float64)
    mu64 = 0.0
    for v in d64:                                   # sequential double sums, as in C
        mu64 += v
    mu64 /= NC
    var64 = 0.0
    for v in d64:
        var64 += (v - mu64) * (v - mu64)
    mu = np.float32(mu64)
    sd = np.float32(np.sqrt(var64 / NC))
    x = ((d - mu) / (sd + np.float32(1e-6))).astype(np.float16).astype(np.float32)
    xs = (x[BLO:BHI] - m["in_mu"]) / m["in_sd"]
    return np.clip(np.round(xs / np.float32(m["s_in"])), -127, 127).astype(np.int8)


def header_cases() -> np.ndarray:
    h = HEADER.read_text(encoding="ascii")
    body = re.search(r"ichi_model_cases\[ICHI_MODEL_CASE_COUNT\]\[ICHI_FRONT_INPUT_SIZE\] = \{(.*?)\};", h, re.S).group(1)
    return np.array([int(v) for v in body.replace("\n", " ").split(",") if v.strip()], np.int8).reshape(32, -1)


def bf16(v):
    raw = np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32)
    return (((raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16).astype(np.uint32) << 16).view(np.float32)


def infer(xq: np.ndarray, m) -> np.ndarray:
    """int8 N333 (N, 334) -> 32 outputs.  Same math as sim/emit_frontend_model.py int_forward + mcu_reference."""
    h = xq[:, None, :].astype(np.int32)
    for i in range(3):
        k, st = int(m[f"conv{i}_k"]), int(m[f"conv{i}_s"])
        lo = (h.shape[2] - k) // st + 1
        idx = np.arange(lo)[:, None] * st + np.arange(k)[None, :]
        acc = np.einsum("nckl,ock->nol", h[:, :, idx].transpose(0, 1, 3, 2), m[f"conv{i}_wq"].astype(np.int32))
        v = acc.astype(np.float32) * m[f"conv{i}_M"][None, :, None] + m[f"conv{i}_B"][None, :, None]
        h = np.where(v <= 0, 0, np.where(v >= np.float32(126.5), 127, np.floor(v + np.float32(0.5)))).astype(np.int32)
    acc = h.reshape(len(h), -1) @ m["fc_wq"].T.astype(np.int32)
    e = np.maximum(acc.astype(np.float32) * m["fc_M"] + m["fc_B"], np.float32(0))
    zp = np.zeros((len(e), 167), np.float32)
    zp[:, :32] = bf16(e * m["emb_mul"] + m["emb_add"])
    hid = bf16(np.clip(0.2 * (zp @ bf16(m["alpha"])) + 0.5, 0.0, 1.0))
    return bf16(hid @ bf16(m["beta"]))


def emit_tables(m) -> None:
    def arr(ctype, name, vals, fmt, per=12):
        body = ",\n".join("    " + ", ".join(fmt(v) for v in vals[i:i + per]) for i in range(0, len(vals), per))
        return f"static const {ctype} {name}[{len(vals)}] = {{\n{body}\n}};\n"
    f32 = lambda v: f"{float(np.float32(v))!r}f".replace("inf", "INFINITY")
    lines = ["/* Generated by sim/board_fixed_feature.py --emit.  Do not edit. */",
             "#ifndef ICHI_FEATURE_TABLES_H", "#define ICHI_FEATURE_TABLES_H", "#include <stdint.h>", "",
             f"#define ICHI_FEAT_BIN_LO {BLO}", f"#define ICHI_FEAT_BIN_HI {BHI}",
             f"#define ICHI_FEAT_PEAK_S0 {PEAK_S0}", f"#define ICHI_FEAT_PEAK_S1 {PEAK_S1}",
             f"#define ICHI_FEAT_IN_SCALE {f32(m['s_in'])}", "",
             "/* Hann window (np.hanning(2048), symmetric) x 32767, first half. */",
             arr("int16_t", "ichi_win_q15", [int(v) for v in WIN_Q15], str),
             "/* cos(2*pi*k/2048) x 32767, k = 0..1023. */",
             arr("int16_t", "ichi_cos_q15", [int(v) for v in COS_Q15], str),
             "/* Input standardization of the front-end (board_model_frontend_32cls.npz). */",
             arr("float", "ichi_feat_in_mu", m["in_mu"], f32, 6),
             arr("float", "ichi_feat_in_sd", m["in_sd"], f32, 6),
             "#endif", ""]
    (GEN / "ichi_feature_tables.h").write_text("\n".join(lines), encoding="ascii")


def main():
    # float reference = eval_full_data.seg_power / to_db (copied: that module needs torch)
    def seg_power(a):
        seg = np.stack([a[s:s + NFFT] for s in range(0, len(a) - NFFT + 1, HOP)]) * np.hanning(NFFT)
        return (np.abs(np.fft.rfft(seg, axis=1)) ** 2)[:, 1:]

    def to_db(p):
        return np.maximum(10 * np.log10(p + 1e-12), -80.0)

    man = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    m = np.load(MODEL)
    cases = header_cases()
    base_dbs = [frame_db(load(SAMPLES / b["file"])) for b in man["baseline"]]
    base_db = np.mean(base_dbs, axis=0)
    base_q = baseline_q(base_dbs)
    ref_base = np.mean([to_db(seg_power(load(SAMPLES / b["file"]).astype(np.float64) / 32768.0).mean(0))
                        for b in man["baseline"]], axis=0)
    print(f"baseline dB: fixed vs float max |diff| {np.abs(base_db - ref_base).max():.4f} dB")
    tot_diff = 0
    worst = 0
    fixed_q = []
    for s in sorted(man["states"], key=lambda s: s["class_id"]):
        pcm = load(SAMPLES / s["file"])
        db = frame_db(pcm)
        q = features_int8(db, base_q, m)
        ref_db = to_db(seg_power(pcm.astype(np.float64) / 32768.0).mean(0))
        d = np.abs(q.astype(int) - cases[s["class_id"]].astype(int))
        fixed_q.append(q)
        tot_diff += int((d > 0).sum())
        worst = max(worst, int(d.max()))
        print(f"  {s['state']} cls{s['class_id']:2d}: dB max|Δ| {np.abs(db - ref_db).max():.4f}  "
              f"int8 mismatch {int((d > 0).sum()):3d}/334 max {d.max()}", flush=True)
    exp_rows = []
    print(f"total int8 mismatches {tot_diff} / {32 * 334}, worst |Δ| {worst}")
    labels = np.array([s["class_id"] for s in sorted(man["states"], key=lambda s: s["class_id"])])
    out_ref, out_fix = infer(cases, m), infer(np.array(fixed_q), m)
    c_ref, c_fix = out_ref.argmax(1), out_fix.argmax(1)
    print(f"class: header-case {int((c_ref == labels).sum())}/32 correct, fixed-point {int((c_fix == labels).sum())}/32 "
          f"correct, fixed vs header argmax agree {int((c_ref == c_fix).sum())}/32, "
          f"max |Δoutput| {float(np.abs(out_ref - out_fix).max()):.4f}")
    if "--emit" in sys.argv:
        emit_tables(m)
        states = sorted(man["states"], key=lambda s: s["class_id"])
        rows = []
        for i, s in enumerate(states):
            d = np.abs(fixed_q[i].astype(int) - cases[s["class_id"]].astype(int))
            rows.append(dict(state=s["state"], class_id=s["class_id"], file=s["file"],
                             pred_fixed=int(c_fix[i]), pred_header=int(c_ref[i]),
                             mismatch=int((d > 0).sum()), max_diff=int(d.max()),
                             outputs_fixed=[float(v) for v in out_fix[i]]))
        exp = dict(note="sim/board_fixed_feature.py の固定小数点参照 (実機と同じ整数演算)。clip 順 = baseline → class 0..31",
                   baseline=[b["file"] for b in man["baseline"]], states=rows,
                   fixed_correct=int((c_fix == labels).sum()), header_correct=int((c_ref == labels).sum()))
        (GEN / "pipeline_expected.json").write_text(json.dumps(exp, indent=1), encoding="utf-8")
        print("wrote generated/ichi_feature_tables.h and generated/pipeline_expected.json")


if __name__ == "__main__":
    main()
