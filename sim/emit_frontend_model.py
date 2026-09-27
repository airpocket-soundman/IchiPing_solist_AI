"""実機 (ML63Q2557) 用の CNN 前段 (int8, CPU) + ELM ヘッド (AxlCORE) 32 クラスモデルを生成する。

構成 (docs/DEVELOPMENT.md):
  MCU: N333 特徴 (noise_diff_norm 400–3000 Hz, 334 bin) を標準化し int8 量子化 (src/ichi_feature.c)
  CPU: Conv1d 層 (BN 融合, ReLU) → FC → 埋め込み (ReLU)。重み int8 (出力 ch 毎スケール)、活性化 int8 (層毎スケール)、
       int32 累積、再量子化は float32 (v = acc·M + B, v≤0→0, v≥126.5→127, それ以外 floor(v+0.5))
  AI : 埋め込みを標準化・スケールして bf16 化し 167 入力へゼロ埋め → ELM 167→32→32
       (α = Sim seed1 の 167 入力 α, 実機一致確認済み, hard sigmoid)
前段 (--arch):
  b1    Conv 16/32/64/64 (k9/7/5/3, stride 2/2/2/1) → FC 2304→64   ≈174 KB int8, 活性化 2×2.6 KB (既定)
  small Conv 8/16/32 (k9/7/5, stride 2) → FC 1216→32               ≈42 KB int8,  活性化 2×1.3 KB
学習: UNO Q train session1..8 (session4 を early stop / ハイパラ・seed 選択に使用), 周波数シフト aug ±2%。
評価: UNO Q eval gray/evening/survey/crowd (学習に不使用)。

出力 (firmware/IchiPingInference/generated/):
  ichiping_model.h      前段 (int8 重み・層定義表) + ELM (β bf16) + 参照入力 32 ケース (int8)
  ichi_feature_tables.h 実機の特徴計算の入力標準化 (in_mu / in_sd / s_in、このモデルの値)
  sim_export/solist_ds/board_model_frontend_32cls_<tag>.npz   全パラメータ (+ BOARD_FRONTEND_PC.json)

完成版モデル (docs/DEVELOPMENT.md の手順):
  python sim/emit_frontend_model.py --arch small --stamp <Stamp 13 セッション> --val stamp_20260927_s2_wav
         --shift 0.005 --xbase 2 --tag best_s0.005

実行: D:/GitHub/IchiPing/pc/.venv/Scripts/python.exe sim/emit_frontend_model.py [--arch b1|small] [--train unoq|all] [--seeds 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full_data import BLO, BHI, FRDM_RUNS, UNOQ_TRAIN, UNOQ_EVAL, build_run, train_net  # noqa: E402
from emit_board_model import to_bf16_bits, q, hard_sigmoid, c_array  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "firmware" / "IchiPingInference" / "generated"
OUT = ROOT / "sim_export" / "solist_ds"
ALPHA = np.load(ROOT / "sim_export" / "_alpha32_sim.npy").astype(np.float32)   # (167,32) 実機と一致確認済み
SCALE_ALPHA_BF16 = 0x3E52
ARCHS = {
    "b1": ([(16, 9, 2), (32, 7, 2), (64, 5, 2), (64, 3, 1)], 64),
    "small": ([(8, 9, 2), (16, 7, 2), (32, 5, 2)], 32),
}
C = 32
N_IN = BHI - BLO
ELM_IN, ELM_HIDDEN = 167, 32
VAL_SESSION = "uno_q_train_20260912_session4_wav"
EVAL_NAMES = {r: r.replace("uno_q_eval_20260912_", "").replace("_wav", "") for r in UNOQ_EVAL}
ACT_PCT = 99.99


# ------------------------------------------------------------------ float model → BN 融合 numpy
def fold(net):
    layers, fc, mods = [], None, list(net.front)
    for i, m in enumerate(mods):
        kind = m.__class__.__name__
        if kind == "Conv1d":
            bn = mods[i + 1]
            g = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().cpu().numpy()
            w = m.weight.detach().cpu().numpy() * g[:, None, None]
            b = (m.bias.detach().cpu().numpy() - bn.running_mean.detach().cpu().numpy()) * g + bn.bias.detach().cpu().numpy()
            layers.append(dict(w=w.astype(np.float32), b=b.astype(np.float32), k=m.kernel_size[0], s=m.stride[0]))
        elif kind == "Linear":
            fc = dict(w=m.weight.detach().cpu().numpy().astype(np.float32), b=m.bias.detach().cpu().numpy().astype(np.float32))
    return layers, fc


def patches(x, k, s):
    """(N, C, L) → (N, C, K, Lo)。"""
    lo = (x.shape[2] - k) // s + 1
    idx = np.arange(lo)[:, None] * s + np.arange(k)[None, :]
    return x[:, :, idx].transpose(0, 1, 3, 2)


def float_forward(xs, layers, fc):
    acts, h = [], xs[:, None, :].astype(np.float32)
    for L in layers:
        h = np.maximum(np.einsum("nckl,ock->nol", patches(h, L["k"], L["s"]), L["w"]) + L["b"][None, :, None], 0)
        acts.append(h)
    return acts, np.maximum(h.reshape(len(h), -1) @ fc["w"].T + fc["b"], 0)


# ------------------------------------------------------------------ int8 model (実機と同じ演算)
def quantize_model(layers, fc, s_in, s_acts):
    qm, s_prev = [], s_in
    for L, s_out in zip(layers, s_acts):
        sw = np.abs(L["w"]).reshape(len(L["w"]), -1).max(1) / 127.0
        qm.append(dict(wq=np.clip(np.round(L["w"] / sw[:, None, None]), -127, 127).astype(np.int8),
                       M=(s_prev * sw / s_out).astype(np.float32), B=(L["b"] / s_out).astype(np.float32),
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


# ------------------------------------------------------------------ ELM head
def elm_input(e, mul, add):
    z = (e.astype(np.float32) * mul + add).astype(np.float32)
    zp = np.zeros((len(z), ELM_IN), np.float32)
    zp[:, :z.shape[1]] = q(z)
    return zp


def elm_fit(Zp, y, lam):
    H = hard_sigmoid(Zp @ ALPHA)
    Y = np.zeros((len(y), C)); Y[np.arange(len(y)), y] = 1
    return np.linalg.solve(H.T @ H + lam * np.eye(ELM_HIDDEN), H.T @ Y).astype(np.float32)


def mcu_reference(Zp, beta):
    """AxlCORE と同じ bf16 境界の参照計算 (公式 Sim・実機と一致確認済みの方式)。"""
    h = q(hard_sigmoid(Zp.astype(np.float32) @ q(ALPHA)))
    return q(h @ q(beta))


def emb_norm(E, s):
    mu, sd = E.mean(0), E.std(0) + 1e-6
    return (s / sd).astype(np.float32), (-mu * s / sd).astype(np.float32)


def build_candidate(spec, emb_dim, X, Xf, yf, Xv, yv, seed):
    net, _, logits = train_net(spec, emb_dim, Xf, yf, Xv, yv, seed)
    mu, sd = net.in_mu.astype(np.float32), net.in_sd.astype(np.float32)
    layers, fc = fold(net)
    std = lambda A: ((A[:, BLO:BHI].astype(np.float32) - mu) / sd).astype(np.float32)
    # 入力・活性化スケールは学習データだけから決める
    Xs = std(X)
    s_in = float(np.percentile(np.abs(Xs), 99.9) / 127.0)
    acts, _ = float_forward(Xs[::4], layers, fc)
    s_acts = [float(np.percentile(a, ACT_PCT) / 127.0) for a in acts]
    qm, fq = quantize_model(layers, fc, s_in, s_acts)
    quant_in = lambda A: np.clip(np.round(std(A) / s_in), -127, 127).astype(np.int8)
    embed = lambda A: np.concatenate([int_forward(quant_in(A[i:i + 2048]), qm, fq) for i in range(0, len(A), 2048)])
    # ELM のスケール・ridge は検証 session だけで選ぶ
    E_fit, E_val = embed(Xf), embed(Xv)
    best = None
    for s in (0.25, 0.5, 1.0, 2.0):
        mul, add = emb_norm(E_fit, s)
        for lam in (0.1, 1.0, 10.0):
            beta = elm_fit(elm_input(E_fit, mul, add), yf, lam)
            acc = float(np.mean(mcu_reference(elm_input(E_val, mul, add), beta).argmax(1) == yv))
            if best is None or acc > best[0]:
                best = (acc, s, lam)
    return dict(net=net, logits=logits, layers=layers, fc=fc, qm=qm, fq=fq, std=std, quant_in=quant_in, embed=embed,
                mu=mu, sd=sd, s_in=s_in, s_acts=s_acts, val_acc=best[0], s_elm=best[1], lam=best[2], seed=seed)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="b1", choices=tuple(ARCHS))
    ap.add_argument("--train", default="unoq", choices=("unoq", "all"),
                    help="unoq: UNO Q session1-8 のみ (既定, session4 の検証精度で選定済み) / all: + FRDM 全 run")
    ap.add_argument("--seeds", type=int, default=3, help="前段の学習 seed 数 (検証精度最大を採用)")
    ap.add_argument("--stamp", nargs="*", default=[],
                    help="学習に加えるこのハード (Stamp-S3A) のセッション (captures/stamp_*)。先頭を検証 session にする")
    ap.add_argument("--val", default="", help="検証 session (既定: --stamp の先頭、なければ UNO Q session4)")
    ap.add_argument("--shift", type=float, default=0.02, help="比例周波数シフト aug の最大 |ε| (0 = なし)")
    ap.add_argument("--xbase", type=int, default=0,
                    help="クロスベースライン: 学習 frame を同じ世代の別セッション K 本の baseline でも差分して増やす")
    ap.add_argument("--tag", default="", help="npz 名 board_model_frontend_32cls_<tag>.npz (空 = 既定名 board_model_frontend_32cls.npz を上書き)")
    args = ap.parse_args()
    spec, emb_dim = ARCHS[args.arch]
    train_runs = UNOQ_TRAIN + (FRDM_RUNS if args.train == "all" else []) + args.stamp
    val_session = args.val or (args.stamp[0] if args.stamp else VAL_SESSION)
    import eval_full_data
    eval_full_data.SHIFT_MAX = args.shift                  # train_net の周波数シフト aug
    data = {r: build_run(r) for r in train_runs + UNOQ_EVAL}
    cat = lambda rs, k: np.concatenate([data[r][k] for r in rs])
    fit_runs = [r for r in train_runs if r != val_session]
    if args.xbase > 0:
        from eval_best_model import build_train          # クロスベースライン (sim/eval_best_model.py と同じ作り方)
        X, y = build_train(train_runs, args.xbase, np.random.default_rng(0))
        X = X.astype(np.float32)
        Xf, yf = build_train(fit_runs, args.xbase, np.random.default_rng(0))
    else:
        X, y = cat(train_runs, "X").astype(np.float32), cat(train_runs, "y")
        Xf, yf = cat(fit_runs, "X"), cat(fit_runs, "y")
    Xv, yv = data[val_session]["X"], data[val_session]["y"]

    # 前段の学習 (seed 毎) → int8 化 → ELM 選択。検証 session の精度が最も高い seed を採用する
    cands = []
    for seed in range(args.seeds):
        cands.append(build_candidate(spec, emb_dim, X, Xf, yf, Xv, yv, seed))
        print(f"  seed {seed}: validation acc={cands[-1]['val_acc']:.3f}", flush=True)
    m = max(cands, key=lambda c: c["val_acc"])
    net, logits, layers, fc, qm, fq = m["net"], m["logits"], m["layers"], m["fc"], m["qm"], m["fq"]
    std, quant_in, embed = m["std"], m["quant_in"], m["embed"]
    mu, sd, s_in, s_acts = m["mu"], m["sd"], m["s_in"], m["s_acts"]
    val_acc, s_elm, lam, seed = m["val_acc"], m["s_elm"], m["lam"], m["seed"]
    E_all = embed(X)
    mul, add = emb_norm(E_all, s_elm)
    beta = elm_fit(elm_input(E_all, mul, add), y, lam)
    print(f"arch={args.arch} train={args.train}: seed {seed} validation acc={val_acc:.3f}  ELM s={s_elm} ridge={lam}  "
          f"s_in={s_in:.4f} s_acts={[round(v, 4) for v in s_acts]}")

    # 評価 (学習に使っていない UNO Q eval 4 セット)
    report = {}
    for r in UNOQ_EVAL:
        Xe, ye = data[r]["X"].astype(np.float32), data[r]["y"]
        Zp = elm_input(embed(Xe), mul, add)
        P_ref = mcu_reference(Zp, beta)
        P_float = hard_sigmoid(elm_input(float_forward(std(Xe), layers, fc)[1], mul, add) @ ALPHA) @ beta
        report[EVAL_NAMES[r]] = dict(frames=len(ye),
                                     cnn_head_float=float(np.mean(logits(Xe).argmax(1) == ye)),
                                     elm_float_frontend=float(np.mean(P_float.argmax(1) == ye)),
                                     elm_int8_frontend_bf16_ref=float(np.mean(P_ref.argmax(1) == ye)))
    for k, v in report.items():
        print(f"  {k:8s} n={v['frames']:4d}  CNN(float)={v['cnn_head_float']:.3f}  "
              f"ELM(float 前段)={v['elm_float_frontend']:.3f}  ELM(int8 前段, 実機参照)={v['elm_int8_frontend_bf16_ref']:.3f}")

    # 参照入力 (ヘッダに埋め込み、sim/board_fixed_feature.py が固定小数点の特徴計算と照合する): evening の各クラス先頭 frame
    ev = "uno_q_eval_20260912_evening_wav"
    ye = data[ev]["y"]
    case_idx = [int(np.flatnonzero(ye == c)[0]) for c in range(C)]
    cases = quant_in(data[ev]["X"][case_idx].astype(np.float32))

    write_header(spec, emb_dim, qm, fq, mul, add, beta, cases, ye[case_idx], args.arch)
    npz = OUT / (f"board_model_frontend_32cls_{args.tag}.npz" if args.tag else "board_model_frontend_32cls.npz")
    np.savez(npz, arch=args.arch, in_mu=mu, in_sd=sd, s_in=s_in, s_acts=np.array(s_acts), emb_mul=mul, emb_add=add,
             beta=beta, alpha=ALPHA, s_elm=s_elm, ridge=lam, fc_wq=fq["wq"], fc_M=fq["M"], fc_B=fq["B"],
             **{f"conv{i}_{k}": L[k] for i, L in enumerate(qm) for k in ("wq", "M", "B", "k", "s")})
    # 入力の標準化 (in_mu / in_sd / s_in) はモデル毎に違うので、実機の特徴計算用テーブルも同じ npz から作り直す
    from board_fixed_feature import emit_tables
    emit_tables(np.load(npz))
    meta = dict(
        model=f"ichiping_frontend_{args.arch}_elm{ELM_IN}x{ELM_HIDDEN}x{C}", input_format="int8_frontend",
        input_count=int(N_IN), embedding=emb_dim, elm_input_count=ELM_IN, hidden_count=ELM_HIDDEN, output_count=C,
        eval_set="unoq eval gray+evening+survey+crowd", activation="hard_sigmoid", loss="mse",
        scale_alpha_bf16=f"0x{SCALE_ALPHA_BF16:04X}", input_scale=s_in, elm_scale=s_elm, ridge=lam,
        train_data="UNO Q train session1-8" + (" + FRDM all runs" if args.train == "all" else "")
        + (" + Stamp " + ", ".join(args.stamp) if args.stamp else "")
        + f"; shift aug +-{args.shift:g}, cross-baseline {args.xbase}, validation {val_session}",
        validation_accuracy=val_acc, seed=seed, pc_accuracy=report, model_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
        alpha_origin="ROHM Solist-AI Simulator seed=1 capture (sim_export/_alpha32_sim.npy), inputSize=167")
    (OUT / "BOARD_FRONTEND_PC.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {GEN / 'ichiping_model.h'}  {npz.name}")


# ------------------------------------------------------------------ C header
def f32_array(name, v):
    flat = np.asarray(v, np.float32).reshape(-1)
    body = ",\n".join("    " + ", ".join(f"{float(x):.9e}f" for x in flat[i:i + 6]) for i in range(0, len(flat), 6))
    return f"static const float {name}[{len(flat)}] = {{\n{body}\n}};"


def i8_array(name, v, decl=None):
    flat = np.asarray(v, np.int8).reshape(-1)
    body = "\n".join("    " + ", ".join(str(int(x)) for x in flat[i:i + 24]) + "," for i in range(0, len(flat), 24))
    return f"static const int8_t {decl or f'{name}[{len(flat)}]'} = {{\n{body}\n}};"


def write_header(spec, emb_dim, qm, fq, mul, add, beta, cases, case_cls, arch):
    dims, L = [], N_IN
    for i, (oc, k, s) in enumerate(spec):
        Lo = (L - k) // s + 1
        dims.append((1 if i == 0 else spec[i - 1][0], oc, k, s, L, Lo))
        L = Lo
    flat = spec[-1][0] * L
    h = [f"/* Generated by sim/emit_frontend_model.py --arch {arch} "
         f"(int8 CNN front-end + ELM {ELM_IN}x{ELM_HIDDEN}x{C}). Do not edit. */",
         "#ifndef ICHIPING_MODEL_H", "#define ICHIPING_MODEL_H", "#include <stdint.h>", "",
         "/* ELM head (AxlCORE) */",
         f"#define ICHI_MODEL_INPUT_SIZE {ELM_IN}", f"#define ICHI_MODEL_HIDDEN_SIZE {ELM_HIDDEN}",
         f"#define ICHI_MODEL_OUTPUT_SIZE {C}", f"#define ICHI_MODEL_CASE_COUNT {C}",
         "#define ICHI_MODEL_SEED 1", "#define ICHI_MODEL_ACTIVATION 1", "#define ICHI_MODEL_LOSS 1",
         f"#define ICHI_MODEL_SCALE_ALPHA_BF16 ((int16_t)0x{SCALE_ALPHA_BF16:04X})", "",
         "/* CNN front-end (Cortex-M0+) */", "#define ICHI_FRONTEND_ENABLED 1",
         f"#define ICHI_FRONT_INPUT_SIZE {N_IN}", f"#define ICHI_FRONT_LAYERS {len(spec)}",
         f"#define ICHI_FRONT_EMB {emb_dim}", f"#define ICHI_FRONT_FLAT {flat}",
         f"#define ICHI_FRONT_MAX_ACT {max(oc * lo for _, oc, _, _, _, lo in dims)}", "",
         "typedef struct\n{\n    const int8_t *w;      /* [out_ch][in_ch][kernel] */\n"
         "    const float *m;       /* requantization multiplier per out_ch */\n"
         "    const float *b;       /* requantization bias per out_ch */\n"
         "    uint16_t in_ch, out_ch, kernel, stride, in_len, out_len;\n} IchiFrontLayer;", ""]
    for i, L in enumerate(qm):
        h += [i8_array(f"ichi_front_w{i}", L["wq"]), f32_array(f"ichi_front_m{i}", L["M"]),
              f32_array(f"ichi_front_b{i}", L["B"]), ""]
    rows = [f"    {{ ichi_front_w{i}, ichi_front_m{i}, ichi_front_b{i}, {ic}, {oc}, {k}, {s}, {li}, {lo} }},"
            for i, (ic, oc, k, s, li, lo) in enumerate(dims)]
    h += ["static const IchiFrontLayer ichi_front_layers[ICHI_FRONT_LAYERS] = {", *rows, "};", ""]
    h += [i8_array("ichi_front_fc_w", fq["wq"]), f32_array("ichi_front_fc_m", fq["M"]),
          f32_array("ichi_front_fc_b", fq["B"]), f32_array("ichi_front_emb_mul", mul),
          f32_array("ichi_front_emb_add", add), "", c_array("ichi_model_beta", to_bf16_bits(beta)),
          i8_array("ichi_model_cases", cases, decl="ichi_model_cases[ICHI_MODEL_CASE_COUNT][ICHI_FRONT_INPUT_SIZE]"),
          "static const uint8_t ichi_model_case_class[ICHI_MODEL_CASE_COUNT] = {"
          + ", ".join(str(int(c)) for c in case_cls) + "};", "", "#endif"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "ichiping_model.h").write_text("\n".join(h) + "\n", encoding="ascii", newline="\n")


if __name__ == "__main__":
    main()
