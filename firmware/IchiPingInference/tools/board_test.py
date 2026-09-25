"""実機 (DT-EBML63Q2557 / ML63Q2557) の IchiPing 推論を PC 参照計算と突き合わせる。

1. HELLO / STATUS で疎通と書き込まれたモデルを確認
2. AI_SELFTEST: 内蔵ケース (クラス毎 1 frame) を推論し golden (PC の実機同等計算) と比較
3. AI_INFER   : generated/stream_cases.npz の全 frame を送り、PC 参照との一致率と正解率を集計
モデル種別 (CNN 前段 int8 入力 / ELM のみ bf16 入力) と出力数は generated/golden_outputs.json から判定する。

usage (pyserial の入った python で):
  python firmware/IchiPingInference/tools/board_test.py --port COM3 [--stream-limit 100]
結果: docs/board_inference_test_cnn_frontend.{md,json} (ELM のみのモデルは docs/board_inference_test.*)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GEN = HERE.parent / "generated"
sys.path.insert(0, str(HERE))
from ichi_serial import Board, HELLO  # noqa: E402

ABS_TOL, REL_TOL = 0.035, 0.05


def within_tol(a, b):
    return np.abs(a - b) <= np.maximum(ABS_TOL, REL_TOL * np.abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--stream-limit", type=int, default=0, help="0 = 全 frame")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    golden = json.loads((GEN / "golden_outputs.json").read_text(encoding="utf-8"))
    meta = golden["metadata"]
    frontend = meta.get("input_format") == "int8_frontend"
    n_out = int(meta.get("output_count", 14))
    stream = np.load(GEN / "stream_cases.npz")
    X = stream["inputs_int8"] if frontend else stream["inputs_bf16"]
    ye = stream["expected_class"]
    sets = stream["eval_set"] if "eval_set" in stream.files else np.array([meta.get("eval_set", "")] * len(ye))
    P_ref = stream["outputs_ref"]
    n = len(X) if a.stream_limit <= 0 else min(a.stream_limit, len(X))
    out = Path(a.out or ROOT / "docs" / ("board_inference_test_cnn_frontend" if frontend else "board_inference_test"))

    b = Board(a.port)
    try:
        hello = b.request(HELLO).payload.decode(errors="replace")
        st = b.status()
        print(f"HELLO {hello}  STATUS {st}")
        if st["frontend"] != frontend or st["outputs"] != n_out:
            raise SystemExit("書き込まれたファームと generated/ のモデルが一致しません (build し直して書き込むこと)")

        selftest = []
        for case in golden["cases"]:
            r = b.selftest(case["board_case_id"], n_out)
            exp = np.array(case["outputs"], np.float32)
            ok = r["cls"] == case["predicted_class"] and bool(within_tol(r["scores"], exp).all())
            selftest.append(dict(case=case["board_case_id"], expected_class=case["expected_class"],
                                 pc_class=case["predicted_class"], board_class=r["cls"],
                                 max_abs_err=float(np.abs(r["scores"] - exp).max()), passed=ok,
                                 total_ms=r["total_ms"], accelerator_us=r["accelerator_us"]))
            print(f"  case {case['board_case_id']:2d} pc={case['predicted_class']:2d} board={r['cls']:2d} "
                  f"max|Δ|={selftest[-1]['max_abs_err']:.4f} {'PASS' if ok else 'FAIL'} ({r['total_ms']} ms)")

        board_cls = np.zeros(n, int); board_scores = np.zeros((n, n_out), np.float32)
        total_ms = np.zeros(n, int); accel_us = np.zeros(n, int); rts = []
        for i in range(n):
            payload = X[i].astype(np.int8).tobytes() if frontend else X[i].astype("<u2").tobytes()
            t0 = time.perf_counter()
            r = b.infer(payload, n_out)
            rts.append(time.perf_counter() - t0)
            board_cls[i], board_scores[i], total_ms[i], accel_us[i] = r["cls"], r["scores"], r["total_ms"], r["accelerator_us"]
            if (i + 1) % 50 == 0:
                print(f"  streamed {i+1}/{n}  agree(PC)={np.mean(board_cls[:i+1] == P_ref[:i+1].argmax(1)):.1%}  "
                      f"acc={np.mean(board_cls[:i+1] == ye[:i+1]):.1%}", flush=True)
    finally:
        b.close()

    ref_cls = P_ref[:n].argmax(1)
    err = np.abs(board_scores - P_ref[:n])
    per_set = {}
    for s in dict.fromkeys(sets[:n]):
        m = sets[:n] == s
        votes = [int(board_scores[m & (ye[:n] == c)].sum(0).argmax() == c) for c in np.unique(ye[:n][m])]
        per_set[str(s)] = dict(frames=int(m.sum()), board_accuracy=float(np.mean(board_cls[m] == ye[:n][m])),
                               pc_accuracy=float(np.mean(ref_cls[m] == ye[:n][m])), board_vote=float(np.mean(votes)))
    summary = dict(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"), port=a.port, hello=hello, status=st, model=meta,
        selftest=dict(cases=len(selftest), passed=sum(s["passed"] for s in selftest),
                      class_match=sum(s["board_class"] == s["pc_class"] for s in selftest),
                      max_abs_err=max(s["max_abs_err"] for s in selftest)),
        stream=dict(frames=n, board_vs_pc_argmax_agreement=float(np.mean(board_cls == ref_cls)),
                    board_accuracy=float(np.mean(board_cls == ye[:n])), pc_accuracy=float(np.mean(ref_cls == ye[:n])),
                    score_abs_err_max=float(err.max()), score_abs_err_mean=float(err.mean()),
                    within_tolerance=float(np.mean(within_tol(board_scores, P_ref[:n]))),
                    total_ms_mean=float(total_ms.mean()), accelerator_us_mean=float(accel_us.mean()),
                    roundtrip_ms_mean=float(np.mean(rts) * 1e3), per_eval_set=per_set),
        selftest_cases=selftest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez(out.with_name(out.name + "_stream.npz"), board_cls=board_cls, board_scores=board_scores,
             pc_ref=P_ref[:n], expected=ye[:n], eval_set=sets[:n], total_ms=total_ms, accelerator_us=accel_us)

    s, t = summary["selftest"], summary["stream"]
    kind = (f"int8 {meta['input_count']} 入力 → CPU CNN 前段 (埋め込み {meta.get('embedding')}) → ELM "
            f"{meta.get('elm_input_count')}→{meta['hidden_count']}→{n_out}") if frontend else \
           f"bf16 {meta['input_count']} 入力 → ELM {meta['hidden_count']}→{n_out}"
    lines = [f"# Solist-AI 実機推論テスト ({summary['generated_at']})", "",
             f"- ボード: DT-EBML63Q2557 (ML63Q2557), UART {a.port} 115200 bps, HELLO `{hello}`",
             f"- モデル: {meta['model']} ({kind}), 学習 {meta.get('train_data', '-')}", "",
             f"## 1. 自己テスト (AI_SELFTEST, {s['cases']} ケース)", "",
             f"- クラス一致 {s['class_match']}/{s['cases']}, スコア許容内 {s['passed']}/{s['cases']}, "
             f"最大スコア誤差 {s['max_abs_err']:.4f} (許容 abs {ABS_TOL} / rel {REL_TOL})", "",
             f"## 2. PC からの入力送信推論 (AI_INFER, {t['frames']} frame)", "",
             f"- Board と PC 参照の argmax 一致: {t['board_vs_pc_argmax_agreement']:.1%}, "
             f"スコア誤差 max {t['score_abs_err_max']:.4f} / mean {t['score_abs_err_mean']:.4f}, 許容内 {t['within_tolerance']:.1%}",
             f"- 正解率 (frame): Board {t['board_accuracy']:.1%} / PC 参照 {t['pc_accuracy']:.1%}",
             f"- 処理時間: 実機全体 平均 {t['total_ms_mean']:.0f} ms (10 ms 分解能), AxlCORE 平均 {t['accelerator_us_mean']:.0f} us, "
             f"UART 往復 平均 {t['roundtrip_ms_mean']:.0f} ms", "",
             "| 評価セット | frame 数 | Board 正解率 | PC 参照 | Board 状態投票 |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['frames']} | {v['board_accuracy']:.1%} | {v['pc_accuracy']:.1%} | {v['board_vote']:.1%} |"
              for k, v in per_set.items()]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(per_set) + 8):]))
    print(f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
