"""実機 (ML63Q2557) 用の CNN 前段 (int8, CPU) + ELM ヘッド (AxlCORE) 32 クラスモデルを生成する。

構成 (docs/HANDOFF_SOLIST_CNN_FRONTEND_20260925.md の推奨):
  PC:  N333 特徴 (noise_diff_norm 400–3000 Hz) を標準化し int8 量子化して AI_INFER で送る (333 B)
  CPU: Conv1d 8/16/32 (k9/7/5, stride 2, BN 融合, ReLU) → FC 1216→32 (ReLU)。重み int8 (出力 ch 毎スケール)、
       活性化 int8 (層毎スケール)、int32 累積、再量子化は float32 (v = acc·M + B, v≤0→0, 四捨五入, 上限 127)
  AI:  埋め込み 32 を標準化・スケールして bf16 化、167 入力へゼロ埋め → ELM 167→32→32 (Sim seed1 α, hard sigmoid)
学習: UNO Q train session1..8 (1 session を early stop / ハイパラ選択に使用), 周波数シフト aug ±2%。
評価: UNO Q eval gray/evening/survey/crowd (学習に不使用)。

出力 (firmware/IchiPingInference/generated/):
  ichiping_model.h      ELM (β bf16) + 前段 (int8 重み・float32 再量子化係数) + 自己テスト 32 ケース (int8 入力)
  golden_outputs.json   自己テスト golden (PC の実機同等参照計算)
  stream_cases.npz      評価 4 セット全 frame (int8 入力, 期待クラス, PC 参照出力)
  sim_export/solist_ds/board_model_frontend_32cls.npz   全パラメータ

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/emit_frontend_model.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_data import BLO, BHI, FRDM_RUNS, UNOQ_TRAIN, UNOQ_EVAL, build_run, train_net  # noqa: E402
from emit_board_model import to_bf16_bits, from_bf16_bits, q, hard_sigmoid, c_array  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "firmware" / "IchiPingInference" / "generated"
OUT = ROOT / "sim_export" / "solist_ds"
ALPHA = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float32)   # (167,32) 実機と一致確認済み
SCALE_ALPHA_BF16 = 0x3E52
SPEC = [(8, 9, 2), (16, 7, 2), (32, 5, 2)]
EMB = 32
C = 32
ELM_IN, ELM_HIDDEN = 167, 32
VAL_SESSION = "uno_q_train_20260912_session4_wav"
SEED = 0
EVAL_NAMES = {r: r.replace("uno_q_eval_20260912_", "").replace("_wav", "") for r in UNOQ_EVAL}
ACT_PCT = 99.99


# ------------------------------------------------------------------ float model → folded numpy
def fold(net):
    layers, mods = [], list(net.front)
    i = 0
    while i < len(mods):
        m = mods[i]
        if m.__class__.__name__ == "Conv1d":
            bn = mods[i + 1]
            g = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().cpu().numpy()
            w = m.weight.detach().cpu().numpy() * g[:, None, None]
            b = (m.bias.detach().cpu().numpy() - bn.running_mean.detach().cpu().numpy()) * g + bn.bias.detach().cpu().numpy()
            layers.append(dict(w=w.astype(np.float32), b=b.astype(np.float32), k=m.kernel_size[0], s=m.stride[0]))
            i += 3
        elif m.__class__.__name__ == "Linear":
            fc = dict(w=m.weight.detach().cpu().numpy().astype(np.float32), b=m.bias.detach().cpu().numpy().astype(np.float32))
            i += 1
        else:
            i += 1
    return layers, fc


def patches(x, k, s):
    """(N, C, L) → (N, C, K, Lo)。"""
    lo = (x.shape[2] - k) // s + 1
    idx = np.arange(lo)[:, None] * s + np.arange(k)[None, :]          # (Lo, K)
    return x[:, :, idx].transpose(0, 1, 3, 2)


def float_forward(xs, layers, fc):
    acts, h = [], xs[:, None, :].astype(np.float32)
    for L in layers:
        h = np.maximum(np.einsum("nckl,ock->nol", patches(h, L["k"], L["s"]), L["w"]) + L["b"][None, :, None], 0)
        acts.append(h)
    e = np.maximum(h.reshape(len(h), -1) @ fc["w"].T + fc["b"], 0)
    return acts, e


# ------------------------------------------------------------------ int8 model (実機と同じ演算)
def quantize_model(layers, fc, s_in, s_acts):
    qm, s_prev = [], s_in
    for L, s_out in zip(layers, s_acts):
        sw = np.abs(L["w"]).reshape(len(L["w"]), -1).max(1) / 127.0
        wq = np.clip(np.round(L["w"] / sw[:, None, None]), -127, 127).astype(np.int8)
        qm.append(dict(wq=wq, M=(s_prev * sw / s_out).astype(np.float32), B=(L["b"] / s_out).astype(np.float32),
                       k=L["k"], s=L["s"]))
        s_prev = s_out
    sw = np.abs(fc["w"]).max(1) / 127.0
    fq = dict(wq=np.clip(np.round(fc["w"] / sw[:, None]), -127, 127).astype(np.int8),
              M=(s_prev * sw).astype(np.float32), B=fc["b"].astype(np.float32))
    return qm, fq


def requant(acc, M, B):
    v = acc.astype(np.float32) * M + B                                 # float32 (C と同順・同精度)
    r = np.floor(v + np.float32(0.5))
    return np.where(v <= 0, 0, np.where(v >= np.float32(126.5), 127, r)).astype(np.int8)


def int_forward(xq, qm, fq):
    h = xq[:, None, :].astype(np.int8)
    for L in qm:
        acc = np.einsum("nckl,ock->nol", patches(h, L["k"], L["s"]).astype(np.int32), L["wq"].astype(np.int32))
        h = requant(acc, L["M"][None, :, None], L["B"][None, :, None])
    acc = h.reshape(len(h), -1).astype(np.int32) @ fq["wq"].T.astype(np.int32)
    return np.maximum(acc.astype(np.float32) * fq["M"] + fq["B"], np.float32(0))


def elm_input(e, mul, add):
    z = (e.astype(np.float32) * mul + add).astype(np.float32)
    zp = np.zeros((len(z), ELM_IN), np.float32); zp[:, :EMB] = q(z)
    return zp


def elm_fit(Zp, y, lam):
    H = hard_sigmoid(Zp @ ALPHA)
    Y = np.zeros((len(y), C)); Y[np.arange(len(y)), y] = 1
    return np.linalg.solve(H.T @ H + lam * np.eye(ELM_HIDDEN), H.T @ Y).astype(np.float32)


def mcu_reference(Zp, beta):
    h = q(hard_sigmoid(Zp.astype(np.float32) @ q(ALPHA)))
    return q(h @ q(beta))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="unoq", choices=("unoq", "all"),
                    help="unoq: UNO Q session1-8 のみ / all: + FRDM 全 run (前段・ELM とも)")
    args = ap.parse_args()
    train_runs = UNOQ_TRAIN + (FRDM_RUNS if args.train == "all" else [])
    data = {r: build_run(r) for r in train_runs + UNOQ_EVAL}
    cat = lambda rs, k: np.concatenate([data[r][k] for r in rs])
    fit_runs = [r for r in train_runs if r != VAL_SESSION]
    X, y = cat(train_runs, "X").astype(np.float32), cat(train_runs, "y")
    Xf, yf = cat(fit_runs, "X"), cat(fit_runs, "y")
    Xv, yv = data[VAL_SESSION]["X"], data[VAL_SESSION]["y"]
    net, _, logits = train_net(SPEC, EMB, Xf, yf, Xv, yv, SEED)
    mu, sd = net.in_mu.astype(np.float32), net.in_sd.astype(np.float32)
    layers, fc = fold(net)
    std = lambda A: ((A[:, BLO:BHI].astype(np.float32) - mu) / sd).astype(np.float32)

    # 入力・活性化スケール (学習データのみから)
    Xs = std(X)
    s_in = float(np.percentile(np.abs(Xs), 99.9) / 127.0)
    acts, e_float = float_forward(Xs[::4], layers, fc)
    s_acts = [float(np.percentile(a, ACT_PCT) / 127.0) for a in acts]
    qm, fq = quantize_model(layers, fc, s_in, s_acts)
    quant_in = lambda A: np.clip(np.round(std(A) / s_in), -127, 127).astype(np.int8)
    emb = lambda A: np.concatenate([int_forward(quant_in(A[i:i + 2048]), qm, fq) for i in range(0, len(A), 2048)])

    E_fit, E_val = emb(Xf), emb(Xv)
    mu_e, sd_e = E_fit.mean(0), E_fit.std(0) + 1e-6
    best = None
    for s in (0.25, 0.5, 1.0, 2.0):
        for lam in (0.1, 1.0, 10.0):
            mul = (s / sd_e).astype(np.float32); add = (-mu_e * s / sd_e).astype(np.float32)
            beta = elm_fit(elm_input(E_fit, mul, add), yf, lam)
            acc = float(np.mean(mcu_reference(elm_input(E_val, mul, add), beta).argmax(1) == yv))
            if best is None or acc > best[0]:
                best = (acc, s, lam)
    _, s_elm, lam = best
    E_all = emb(X)
    mu_e, sd_e = E_all.mean(0), E_all.std(0) + 1e-6
    mul = (s_elm / sd_e).astype(np.float32); add = (-mu_e * s_elm / sd_e).astype(np.float32)
    beta = elm_fit(elm_input(E_all, mul, add), y, lam)
    print(f"s_in={s_in:.4f} s_acts={[round(v, 4) for v in s_acts]} ELM s={s_elm} λ={lam} (val acc {best[0]:.3f})")

    # 評価 (学習に使っていない UNO Q eval 4 セット)
    print(f"train={args.train}: validation ({VAL_SESSION}) acc = {best[0]:.3f}")
    report, stream = {}, dict(inputs=[], y=[], set=[], ref=[], f32=[], cnn=[])
    for r in UNOQ_EVAL:
        Xe, ye = data[r]["X"].astype(np.float32), data[r]["y"]
        Zp = elm_input(emb(Xe), mul, add)
        P_ref = mcu_reference(Zp, beta)
        P_f32 = hard_sigmoid(Zp @ ALPHA) @ beta
        _, e_fl = float_forward(std(Xe), layers, fc)
        P_float = hard_sigmoid(elm_input(e_fl, mul, add) @ ALPHA) @ beta
        cnn = logits(Xe)
        report[EVAL_NAMES[r]] = dict(frames=len(ye), cnn_head_float=float(np.mean(cnn.argmax(1) == ye)),
                                     elm_float_frontend=float(np.mean(P_float.argmax(1) == ye)),
                                     elm_int8_frontend_bf16_ref=float(np.mean(P_ref.argmax(1) == ye)),
                                     elm_int8_frontend_f32=float(np.mean(P_f32.argmax(1) == ye)))
        stream["inputs"].append(quant_in(Xe)); stream["y"].append(ye); stream["set"] += [EVAL_NAMES[r]] * len(ye)
        stream["ref"].append(P_ref); stream["f32"].append(P_f32)
    for k, v in report.items():
        print(f"  {k:8s} n={v['frames']:4d}  CNN(float)={v['cnn_head_float']:.3f}  ELM(float前段)={v['elm_float_frontend']:.3f}  "
              f"ELM(int8前段, 実機参照)={v['elm_int8_frontend_bf16_ref']:.3f}")

    # 自己テスト: evening の各クラス先頭 frame
    ev = "uno_q_eval_20260912_evening_wav"
    ye = data[ev]["y"]; Xin = quant_in(data[ev]["X"].astype(np.float32))
    case_idx = [int(np.flatnonzero(ye == c)[0]) for c in range(C)]
    P_case = mcu_reference(elm_input(int_forward(Xin[case_idx], qm, fq), mul, add), beta)
    P_case_f32 = hard_sigmoid(elm_input(int_forward(Xin[case_idx], qm, fq), mul, add) @ ALPHA) @ beta

    write_header(qm, fq, mul, add, beta, Xin[case_idx], ye[case_idx])
    npz = OUT / "board_model_frontend_32cls.npz"
    np.savez(npz, in_mu=mu, in_sd=sd, s_in=s_in, s_acts=np.array(s_acts), emb_mul=mul, emb_add=add, beta=beta,
             alpha=ALPHA, s_elm=s_elm, ridge=lam,
             **{f"conv{i}_wq": L["wq"] for i, L in enumerate(qm)}, **{f"conv{i}_M": L["M"] for i, L in enumerate(qm)},
             **{f"conv{i}_B": L["B"] for i, L in enumerate(qm)}, fc_wq=fq["wq"], fc_M=fq["M"], fc_B=fq["B"])
    cases = [dict(board_case_id=i, case_id=f"evening_frame{idx}_class{int(ye[idx])}", eval_index=idx,
                  expected_class=int(ye[idx]), state32=int(ye[idx]), predicted_class=int(P_case[i].argmax()),
                  outputs=P_case[i].astype(float).tolist(), outputs_float32=P_case_f32[i].astype(float).tolist())
             for i, idx in enumerate(case_idx)]
    golden = dict(metadata=dict(
        model="ichiping_frontend_T8-16-32_FC32_elm167x32x32", input_format="int8_frontend",
        input_count=int(BHI - BLO), elm_input_count=ELM_IN, hidden_count=ELM_HIDDEN, output_count=C,
        eval_set="unoq eval gray+evening+survey+crowd", activation="hard_sigmoid", loss="mse",
        scale_alpha_bf16=f"0x{SCALE_ALPHA_BF16:04X}", input_scale=s_in, elm_scale=s_elm, ridge=lam,
        train_data="UNO Q train session1-8" + (" + FRDM all runs" if args.train == "all" else ""), pc_accuracy=report,
        model_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
        alpha_origin="ROHM Solist-AI Simulator seed=1 capture (sim_export/_alpha32_sim.npy), inputSize=167"), cases=cases)
    (GEN / "golden_outputs.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    np.savez(GEN / "stream_cases.npz", inputs_int8=np.concatenate(stream["inputs"]), expected_class=np.concatenate(stream["y"]),
             state32=np.concatenate(stream["y"]), eval_set=np.array(stream["set"]),
             outputs_ref=np.concatenate(stream["ref"]), outputs_f32=np.concatenate(stream["f32"]))
    (OUT / "BOARD_FRONTEND_PC.json").write_text(json.dumps(golden["metadata"], indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {GEN / 'ichiping_model.h'}, golden {len(cases)} cases, stream {sum(len(v) for v in stream['y'])} frames")


def f32_array(name, v):
    flat = np.asarray(v, np.float32).reshape(-1)
    body = ",\n".join("    " + ", ".join(f"{float(x):.9g}f" for x in flat[i:i + 6]) for i in range(0, len(flat), 6))
    return f"static const float {name}[{len(flat)}] = {{\n{body}\n}};\n"


def i8_array(name, v, decl=None):
    flat = np.asarray(v, np.int8).reshape(-1)
    body = "\n".join("    " + ", ".join(str(int(x)) for x in flat[i:i + 24]) + "," for i in range(0, len(flat), 24))
    return f"static const int8_t {decl or f'{name}[{len(flat)}]'} = {{\n{body}\n}};\n"


def write_header(qm, fq, mul, add, beta, cases, case_cls):
    n_in = BHI - BLO
    lens, L = [], n_in
    for (oc, k, s) in SPEC:
        L = (L - k) // s + 1; lens.append(L)
    h = [f"/* Generated by sim/emit_frontend_model.py (CNN front-end T8-16-32 FC32 + ELM 167x32x32, 32 classes). Do not edit. */",
         "#ifndef ICHIPING_MODEL_H", "#define ICHIPING_MODEL_H", "#include <stdint.h>",
         f"#define ICHI_MODEL_INPUT_SIZE {ELM_IN}", f"#define ICHI_MODEL_HIDDEN_SIZE {ELM_HIDDEN}",
         f"#define ICHI_MODEL_OUTPUT_SIZE {C}", f"#define ICHI_MODEL_CASE_COUNT {C}",
         "#define ICHI_MODEL_SEED 1", "#define ICHI_MODEL_ACTIVATION 1", "#define ICHI_MODEL_LOSS 1",
         f"#define ICHI_MODEL_SCALE_ALPHA_BF16 ((int16_t)0x{SCALE_ALPHA_BF16:04X})", "",
         "/* CNN front-end (runs on the Cortex-M0+ CPU) */", "#define ICHI_FRONTEND_ENABLED 1",
         f"#define ICHI_FRONT_INPUT_SIZE {n_in}", f"#define ICHI_FRONT_LAYERS {len(SPEC)}",
         f"#define ICHI_FRONT_EMB {EMB}", f"#define ICHI_FRONT_FLAT {SPEC[-1][0] * lens[-1]}",
         f"#define ICHI_FRONT_MAX_ACT {max(oc * l for (oc, _, _), l in zip(SPEC, lens))}", ""]
    for i, ((oc, k, s), Lo) in enumerate(zip(SPEC, lens)):
        ic = 1 if i == 0 else SPEC[i - 1][0]
        Li = n_in if i == 0 else lens[i - 1]
        h.append(f"#define ICHI_FRONT_L{i}_IN_CH {ic}\n#define ICHI_FRONT_L{i}_OUT_CH {oc}\n#define ICHI_FRONT_L{i}_K {k}\n"
                 f"#define ICHI_FRONT_L{i}_STRIDE {s}\n#define ICHI_FRONT_L{i}_IN_LEN {Li}\n#define ICHI_FRONT_L{i}_OUT_LEN {Lo}")
        h.append(i8_array(f"ichi_front_w{i}", qm[i]["wq"]))
        h.append(f32_array(f"ichi_front_m{i}", qm[i]["M"])); h.append(f32_array(f"ichi_front_b{i}", qm[i]["B"]))
    h.append(i8_array("ichi_front_fc_w", fq["wq"]))
    h.append(f32_array("ichi_front_fc_m", fq["M"])); h.append(f32_array("ichi_front_fc_b", fq["B"]))
    h.append(f32_array("ichi_front_emb_mul", mul)); h.append(f32_array("ichi_front_emb_add", add))
    h.append(c_array("ichi_model_beta", to_bf16_bits(beta)))
    h.append(i8_array("ichi_model_cases", cases, decl="ichi_model_cases[ICHI_MODEL_CASE_COUNT][ICHI_FRONT_INPUT_SIZE]"))
    h.append("static const uint8_t ichi_model_case_class[ICHI_MODEL_CASE_COUNT] = {" + ", ".join(str(int(c)) for c in case_cls) + "};")
    h.append("#endif")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "ichiping_model.h").write_text("\n".join(h) + "\n", encoding="ascii", newline="\n")


if __name__ == "__main__":
    main()
