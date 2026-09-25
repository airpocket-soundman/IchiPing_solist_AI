"""DT-EBML63Q2557 上の IchiPing 14cls ELM 推論をシリアル経由で検証する。

1. HELLO / STATUS で疎通確認
2. AI_SELFTEST (0x14) case 0..13: 内蔵の評価ベクトルを推論し PC golden (bf16 参照) と比較
3. AI_INFER (0x16): 評価セット全 frame (167 bf16) を PC から送り、クラス一致率とスコア誤差を集計
結果は JSON と Markdown を docs/ 配下へ書く。

usage (pyserial が入っている anaconda python で):
  C:/ProgramData/anaconda3/python.exe firmware/IchiPingInference/tools/board_test.py --port COM3 [--stream-limit 320]
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
GEN = HERE.parent / "generated"
sys.path.insert(0, r"D:/GitHub/acrylic_pan/pc")
from acrylic_pan_monitor import protocol as P  # noqa: E402  (COBS/CRC framing shared with the firmware)

AI_INFER = 0x16
ABS_TOL, REL_TOL = 0.035, 0.05
# 出力数・入力形式は generated/golden_outputs.json の metadata から決める
# (ELM のみ: 14 出力・bf16 入力 / CNN 前段 + ELM: 32 出力・int8 入力)
OUT = 14
RESULT = struct.Struct(f"<BBH{OUT}f")


def configure(meta: dict):
    global OUT, RESULT
    OUT = int(meta.get("output_count", 14))
    RESULT = struct.Struct(f"<BBH{OUT}f")


class Board:
    def __init__(self, port: str, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)
        self.dec = P.FrameStreamDecoder()
        self.seq = 1
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def request(self, mtype: int, payload: bytes = b"", timeout: float = 2.0) -> P.Frame:
        seq = self.seq; self.seq += 1
        self.ser.write(P.encode_frame(P.Frame(P.MessageType(mtype) if mtype in P.MessageType._value2member_map_ else mtype, seq, payload)))
        t0 = time.time()
        while time.time() - t0 < timeout:
            data = self.ser.read(4096)
            if not data:
                continue
            for fr in self.dec.feed(data):
                if fr.sequence == seq:
                    return fr
        raise TimeoutError(f"no reply to 0x{mtype:02X} seq={seq}")

    def close(self):
        self.ser.close()


def parse_result(fr: P.Frame):
    if int(fr.message_type) == P.MessageType.NACK:
        raise RuntimeError(f"NACK: request=0x{fr.payload[0]:02X} reason={fr.payload[1]}")
    if int(fr.message_type) != P.MessageType.AI_RESULT or len(fr.payload) != RESULT.size:
        raise RuntimeError(f"unexpected reply type=0x{int(fr.message_type):02X} len={len(fr.payload)}")
    case_id, cls, infer_us, *scores = RESULT.unpack(fr.payload)
    return case_id, cls, np.array(scores, np.float32), infer_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--stream-limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--out", default="", help="既定: docs/board_inference_test (ELM) / docs/board_inference_test_cnn_frontend (CNN 前段)")
    a = ap.parse_args()
    golden = json.loads((GEN / "golden_outputs.json").read_text(encoding="utf-8"))
    configure(golden["metadata"])
    frontend = golden["metadata"].get("input_format") == "int8_frontend"
    if not a.out:
        a.out = str(ROOT / "docs" / ("board_inference_test_cnn_frontend" if frontend else "board_inference_test"))
    stream = np.load(GEN / "stream_cases.npz")
    b = Board(a.port)
    try:
        hello = b.request(P.MessageType.HELLO)
        status = b.request(P.MessageType.STATUS)
        st = struct.unpack("<BBHHIB", status.payload) if len(status.payload) == 11 else None
        print(f"HELLO -> 0x{int(hello.message_type):02X}  STATUS -> {st}")

        # ---- self-test cases
        selftest = []
        for case in golden["cases"]:
            t0 = time.perf_counter()
            cid, cls, scores, us = parse_result(b.request(P.MessageType.AI_SELFTEST, bytes([case["board_case_id"]]), timeout=10.0))
            rt = time.perf_counter() - t0
            exp = np.array(case["outputs"], np.float32)
            err = np.abs(scores - exp)
            ok = (cls == case["predicted_class"]) and all(
                abs(x - y) <= max(ABS_TOL, REL_TOL * abs(y)) for x, y in zip(scores, exp))
            selftest.append(dict(case=case["board_case_id"], name=case["case_id"], expected_class=case["expected_class"],
                                 pc_class=case["predicted_class"], board_class=int(cls),
                                 max_abs_err=float(err.max()), passed=bool(ok), roundtrip_ms=rt * 1e3, infer_us=us,
                                 board_scores=scores.tolist(), pc_scores=exp.tolist()))
            print(f"  case {cid:2d} {case['case_id']:32s} pc={case['predicted_class']:2d} board={cls:2d} "
                  f"max|Δ|={err.max():.4f} {'PASS' if ok else 'FAIL'} (rt {rt*1e3:.0f} ms, infer {us} us)")

        # ---- streaming inference over the evaluation set
        X = stream["inputs_int8"] if frontend else stream["inputs_bf16"]
        ye = stream["expected_class"]; y32 = stream["state32"]
        sets = stream["eval_set"] if "eval_set" in stream.files else np.array([""] * len(ye))
        P_ref = stream["outputs_ref"]; P_f32 = stream["outputs_f32"]
        n = len(X) if a.stream_limit <= 0 else min(a.stream_limit, len(X))
        board_cls = np.zeros(n, int); board_scores = np.zeros((n, OUT), np.float32); rts = []; infer_us = np.zeros(n, int)
        for i in range(n):
            payload = X[i].astype(np.int8).tobytes() if frontend else X[i].astype("<u2").tobytes()
            t0 = time.perf_counter()
            _, cls, scores, us = parse_result(b.request(AI_INFER, payload, timeout=10.0))
            rts.append(time.perf_counter() - t0)
            board_cls[i] = cls; board_scores[i] = scores; infer_us[i] = us
            if (i + 1) % 40 == 0:
                print(f"  streamed {i+1}/{n}  agree(pc bf16)={np.mean(board_cls[:i+1]==P_ref[:i+1].argmax(1)):.1%}", flush=True)
        ref_cls = P_ref[:n].argmax(1); f32_cls = P_f32[:n].argmax(1)
        err = np.abs(board_scores - P_ref[:n])
        srt = np.sort(P_ref[:n], axis=1); margin = srt[:, -1] - srt[:, -2]
        agree = board_cls == ref_cls
        margin_rows = []
        for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)):
            m = (margin >= lo) & (margin < hi)
            if m.any():
                margin_rows.append(dict(margin_lo=lo, margin_hi=hi, frames=int(m.sum()), agreement=float(agree[m].mean())))
        vote_ok = vote_tot = 0
        for es in np.unique(sets[:n]):
            for s in np.unique(y32[:n]):
                idx = np.where((y32[:n] == s) & (sets[:n] == es))[0]
                if len(idx):
                    vote_ok += int(board_scores[idx].sum(0).argmax() == ye[idx][0]); vote_tot += 1
        per_set = {str(es): dict(frames=int((sets[:n] == es).sum()),
                                 board_accuracy=float(np.mean(board_cls[sets[:n] == es] == ye[:n][sets[:n] == es])),
                                 pc_bf16_accuracy=float(np.mean(ref_cls[sets[:n] == es] == ye[:n][sets[:n] == es])))
                   for es in np.unique(sets[:n])}
        summary = dict(
            generated_at=dt.datetime.now().isoformat(timespec="seconds"), port=a.port,
            model=golden["metadata"], status=st,
            selftest=dict(cases=len(selftest), passed=sum(s["passed"] for s in selftest),
                          class_match=sum(s["board_class"] == s["pc_class"] for s in selftest),
                          max_abs_err=max(s["max_abs_err"] for s in selftest)),
            stream=dict(frames=n, eval_set=golden["metadata"]["eval_set"],
                        board_vs_pc_bf16_argmax_agreement=float(np.mean(board_cls == ref_cls)),
                        board_vs_pc_float32_argmax_agreement=float(np.mean(board_cls == f32_cls)),
                        board_accuracy_frame=float(np.mean(board_cls == ye[:n])),
                        pc_bf16_accuracy_frame=float(np.mean(ref_cls == ye[:n])),
                        pc_float32_accuracy_frame=float(np.mean(f32_cls == ye[:n])),
                        board_accuracy_vote_per_state=vote_ok / max(vote_tot, 1),
                        score_abs_err_max=float(err.max()), score_abs_err_mean=float(err.mean()),
                        score_err_p99=float(np.percentile(err, 99)),
                        within_tolerance_fraction=float(np.mean(err <= np.maximum(ABS_TOL, REL_TOL * np.abs(P_ref[:n])))),
                        roundtrip_ms_mean=float(np.mean(rts) * 1e3), roundtrip_ms_max=float(np.max(rts) * 1e3),
                        infer_us_mean=float(infer_us.mean()), infer_us_max=int(infer_us.max()),
                        agreement_by_pc_margin=margin_rows, per_eval_set=per_set),
            selftest_cases=selftest)
    finally:
        b.close()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_name(out.name + '_stream.npz'), board_cls=board_cls, board_scores=board_scores, infer_us=infer_us,
             pc_ref=P_ref[:n], pc_f32=P_f32[:n], expected=ye[:n], state32=y32[:n], inputs_bf16=X[:n])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    md = summary; s, t = md["selftest"], md["stream"]
    lines = [f"# Solist-AI 実機推論テスト ({md['generated_at']})", "",
             f"- ボード: DT-EBML63Q2557 (ML63Q2557), UART {a.port} 115200 bps, MCU-Link CMSIS-DAP で書込み",
             f"- モデル: {md['model']['model']} (入力 {md['model']['input_count']} {'int8 → CPU CNN 前段 → ELM ' + str(md['model'].get('elm_input_count')) if frontend else 'bf16 → ELM'}, hidden={md['model']['hidden_count']}, {OUT} 出力, hard sigmoid, bfloat16), α=Sim seed1 再生成, scaleAlpha={md['model']['scale_alpha_bf16']}",
             f"- STATUS: {st}", "",
             f"## 1. 内蔵自己テスト (AI_SELFTEST 0x14, クラス別 {s['cases']} ベクトル)", "",
             f"- クラス一致 {s['class_match']}/{s['cases']}, 許容内 {s['passed']}/{s['cases']}, 最大スコア誤差 {s['max_abs_err']:.4f} (許容 abs {ABS_TOL} / rel {REL_TOL})", "",
             "| case | 期待cls | PC cls | Board cls | max abs err | 判定 |", "|---|---|---|---|---|---|"]
    lines += [f"| {c['case']} | {c['expected_class']} | {c['pc_class']} | {c['board_class']} | {c['max_abs_err']:.4f} | {'PASS' if c['passed'] else 'FAIL'} |" for c in selftest]
    lines += ["", f"## 2. PC からの特徴ベクトル送信推論 (AI_INFER 0x16, 評価セット {t['eval_set']} {t['frames']} frame)", "",
              f"- Board vs PC(bf16参照) argmax 一致: {t['board_vs_pc_bf16_argmax_agreement']:.1%}",
              f"- Board vs PC(float32) argmax 一致: {t['board_vs_pc_float32_argmax_agreement']:.1%}",
              f"- 正解率 (frame): Board {t['board_accuracy_frame']:.1%} / PC bf16 {t['pc_bf16_accuracy_frame']:.1%} / PC float32 {t['pc_float32_accuracy_frame']:.1%}",
              f"- 正解率 (状態別投票): Board {t['board_accuracy_vote_per_state']:.1%}",
              "- 評価セット別 (Board / PC bf16 参照): " + ", ".join(f"{k} {v['board_accuracy']:.1%} / {v['pc_bf16_accuracy']:.1%} (n={v['frames']})" for k, v in t['per_eval_set'].items()),
              f"- スコア誤差: max {t['score_abs_err_max']:.4f}, mean {t['score_abs_err_mean']:.4f}, p99 {t['score_err_p99']:.4f}, 許容内 {t['within_tolerance_fraction']:.1%}",
              f"- 往復時間 (UART 送信 + {'CPU 前段 + ' if frontend else ''}推論 + 受信): 平均 {t['roundtrip_ms_mean']:.1f} ms, 最大 {t['roundtrip_ms_max']:.1f} ms",
              f"- アクセラレータ推論時間 (SysTick 実測, ELM 部のみ): 平均 {t['infer_us_mean']:.0f} us, 最大 {t['infer_us_max']} us", "",
              "PC bf16 参照の top-2 マージン別 argmax 一致率 (不一致が僅差ケースに限られるかの確認):", "",
              "| margin | frames | 一致率 |", "|---|---|---|"] + [
              f"| {r['margin_lo']}-{r['margin_hi'] if r['margin_hi'] < 9 else 'inf'} | {r['frames']} | {r['agreement']:.1%} |" for r in t['agreement_by_pc_margin']] + [""]
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
