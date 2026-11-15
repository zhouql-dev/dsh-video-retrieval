#!/usr/bin/env python3
"""Optimization closed loop — real labeled data in, tuned config out.

Layer 2 (numerical, Optuna): sweep the reid-sim + hue thresholds (+agree) over
the PRW-exported candidates (real labels), compare macro-F1 before/after the
defaults (reid 0.55 / hue 35°), write the optimized thresholds.

Layer 1 (text, confusable table): optimize the plate confusable table over the
real labeled OCR rows (57 reads from the plate eval, labels from report GT).
Tries gepa first (litellm -> glm-5.1); if gepa's integration misbehaves it
falls back to a direct reflective loop via provider.chat (same mechanism:
LLM reads the (table, score, errors) log and proposes a mutated table).

Outputs (in --out):
  layer2.json          before/after macro-F1 + best thresholds + history
  optimized_thresholds.json
  layer1.json          before/after + best table text
  optimized_confusables.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
for p in (_HERE, _FUSION):
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluator            # noqa: E402
import optimize_text as OX  # noqa: E402
import optimize_thresholds as OT  # noqa: E402


def layer2(candidates_path: str, out: str, n_trials: int = 60):
    cands = json.load(open(candidates_path))
    obj = evaluator.make_threshold_objective(cands, agree=2)
    defaults = {"reid_sim": 0.55, "hue_dhue": 35.0}
    before = obj(defaults)
    before_metrics = evaluator.threshold_metrics(cands, defaults, agree=2)

    t0 = time.time()
    res = OT.run(cands, n_trials=n_trials, seed=0, sampler="tpe",
                 search_space=OT.PRW_SPACE)
    best = res["best"]
    after = obj(best)
    after_metrics = evaluator.threshold_metrics(cands, best, agree=2)
    report = {"layer": "layer2_optuna", "n_candidates": len(cands),
              "trials": n_trials, "search_space": OT.PRW_SPACE,
              "before": {"thresholds": defaults, "macro_f1": before,
                         "precision": before_metrics["precision"],
                         "recall": before_metrics["recall"],
                         "tp": before_metrics["tp"], "fp": before_metrics["fp"]},
              "after": {"thresholds": best, "macro_f1": after,
                        "precision": after_metrics["precision"],
                        "recall": after_metrics["recall"],
                        "tp": after_metrics["tp"], "fp": after_metrics["fp"]},
              "delta_macro_f1": round(after - before, 4),
              "wall_s": round(time.time() - t0, 1),
              "history": res["history"][:20]}
    json.dump(report, open(os.path.join(out, "layer2.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"agree": 2, **best}, open(os.path.join(out, "optimized_thresholds.json"), "w"),
              indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     ensure_ascii=False, indent=2))
    return report


def _gepa_layer1(rows, target, max_metric_calls):
    """gepa.api.optimize over the confusable table; returns (best_table, best_f1)
    or None on any integration failure (caller falls back).

    gepa's DefaultAdapter needs a task LM: the candidate text (the table JSON)
    is injected into each data instance's prompt, the task LM emits the answer
    ("1" if the OCR read confusable-matches the target under that table), and
    our evaluator compares it against the labeled answer. The table is the
    artifact being optimized — the same label-based loop as the fallback, but
    driven by gepa's reflective proposer."""
    import gepa.api as api
    from gepa.optimize_anything import make_litellm_lm
    # litellm 'openai/' prefix honors OPENAI_API_BASE/OPENAI_API_KEY -> GLM.
    # glm-4-flash = free text tier (glm-5.1 hits 429 余额不足 on this account —
    # the same quota wall documented across the skill).
    lm = os.environ.get("GEPA_LM", "openai/glm-4-flash")
    os.environ.setdefault("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("ZHIPUAI_API_KEY", "") or
                          os.environ.get("GLM_API_KEY", ""))

    def ev(data, resp):
        return 1.0 if str(resp).strip() == data["answer"] else 0.0

    # keep the trainset small: each metric call costs one task-LM call per row
    pos = [r for r in rows if r["gt"]][:4]
    neg = [r for r in rows if not r["gt"]][:6]
    trainset = [{"input": (f"OCR文本: {r['ocr']}; 目标车牌: {target}; "
                           f"在该混淆表下是否混淆匹配? 只答1或0"),
                 "answer": "1" if r["gt"] else "0",
                 "additional_context": {}} for r in pos + neg]
    result = api.optimize(
        seed_candidate={"confusables": OX.table_to_text(evaluator.DEFAULT_CONFUSABLES)},
        trainset=trainset, evaluator=ev,
        task_lm=make_litellm_lm(lm),
        reflection_lm=make_litellm_lm(lm),
        max_metric_calls=max_metric_calls, seed=0, raise_on_exception=False,
        display_progress_bar=False)
    best_text = getattr(result, "best_candidate", None)
    if best_text is None:
        try:
            best_text = result.best_candidate
        except AttributeError:
            pass
    if not best_text:
        return None
    if isinstance(best_text, dict):
        best_text = best_text.get("confusables", json.dumps(evaluator.DEFAULT_CONFUSABLES))
    obj = evaluator.make_text_objective(rows, target)
    return evaluator.parse_table(best_text), obj(best_text)


def _reflective_layer1(rows, target, iters=8):
    """Direct reflective loop (gepa fallback): glm-5.1 reads the (table, score,
    error-examples) log and proposes a mutated table. Same mechanism as gepa's
    text gradient, wired through provider.chat so it works with the GLM key."""
    import provider
    table = dict(evaluator.DEFAULT_CONFUSABLES)
    obj = evaluator.make_text_objective(rows, target)
    best = obj(table)
    hist = []
    for it in range(iters):
        m = evaluator.text_metrics(rows, target, table)
        errs = [{"ocr": r["ocr"], "gt": r["gt"]}
                for r in rows
                if evaluator.ocr_matches(target, table, r["ocr"]) != r["gt"]][:8]
        log = json.dumps({"current_table": table, "macro_f1": m["macro_f1"],
                          "wrong_examples": errs}, ensure_ascii=False)
        prompt = ("你是车牌 OCR 字符混淆表优化器。当前表(JSON)与成绩如下,"
                  "请提出一个改进的混淆表 JSON(只输出 JSON,不要解释)。"
                  "原则:每个字符的混淆集只含形近字符;漏检样例要覆盖;避免过宽增加误报。\n" + log)
        out = provider.chat([{"role": "user", "content": prompt}],
                            model="glm-5.1", kind="text", max_tokens=1600)
        if not out:
            hist.append({"iter": it, "note": "no reply"})
            break
        cand = evaluator.parse_table(out)
        score = obj(cand)
        hist.append({"iter": it, "macro_f1": score,
                     "kept": score >= best, "table": cand})
        if score >= best:
            best, table = score, cand
    return table, best, hist


def layer1(rows, target, out, max_metric_calls=40):
    obj = evaluator.make_text_objective(rows, target)
    before = obj(evaluator.DEFAULT_CONFUSABLES)
    t0 = time.time()
    mode = "gepa"
    res = None
    try:
        res = _gepa_layer1(rows, target, max_metric_calls)
    except Exception as e:                     # gepa integration failure -> fallback
        print(f"[layer1] gepa failed ({type(e).__name__}: {e}); falling back to "
              f"direct reflective loop")
    if res is None:
        mode = "reflective_direct"
        table, after, hist = _reflective_layer1(rows, target)
    else:
        table, after = res
        hist = []
    report = {"layer": "layer1_text", "mode": mode, "n_rows": len(rows),
              "target": target, "before_macro_f1": before,
              "after_macro_f1": round(after, 4),
              "delta_macro_f1": round(after - before, 4),
              "wall_s": round(time.time() - t0, 1),
              "best_table": table, "history": hist[:10]}
    json.dump(report, open(os.path.join(out, "layer1.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(table, open(os.path.join(out, "optimized_confusables.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("best_table", "history")}, ensure_ascii=False, indent=2))
    return report


def layer1_ccpd(rows_path: str, out: str):
    """Layer-1 over the CCPD-GT export (run_ccpd_gt.py rows.json): REAL plate
    GT, keyless deterministic confusion-pair reflection (no LLM, no network).
    Writes layer1.json + optimized_confusables.json."""
    from run_ccpd_gt import reflect_confusables, tolerant_metrics
    rows = json.load(open(rows_path))
    t0 = time.time()
    table = reflect_confusables(rows, evaluator.DEFAULT_CONFUSABLES)
    before = tolerant_metrics(rows, evaluator.DEFAULT_CONFUSABLES)
    after = tolerant_metrics(rows, table)
    report = {"layer": "layer1_ccpd_gt", "mode": "ccpd_reflect",
              "n_rows": len(rows),
              "before": before, "after": after,
              "delta_tolerant_acc": round(after["tolerant_acc"]
                                          - before["tolerant_acc"], 4),
              "wall_s": round(time.time() - t0, 1),
              "best_table": table}
    json.dump(report, open(os.path.join(out, "layer1.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(table, open(os.path.join(out, "optimized_confusables.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "best_table"}, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None,
                    help="PRW candidates.json (default bench_out/prw_full/candidates.json)")
    ap.add_argument("--plate-rows", default=None,
                    help="labeled OCR rows (default: fusion dataset.plate_ocr_dataset())")
    ap.add_argument("--ccpd-rows", default=None,
                    help="CCPD-GT rows.json from run_ccpd_gt.py — switches Layer-1 "
                         "to the keyless deterministic reflection on real GT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--layer2-only", action="store_true")
    ap.add_argument("--layer1-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.layer1_only:
        path = args.candidates or os.path.join(os.path.dirname(_HERE),
                                               "bench_out", "prw_full", "candidates.json")
        if not os.path.exists(path):
            raise SystemExit(f"candidates.json not found at {path} — run bench/run_prw.py first")
        layer2(path, args.out, args.n_trials)

    if not args.layer2_only:
        if args.ccpd_rows:
            layer1_ccpd(args.ccpd_rows, args.out)
        else:
            import dataset as DS
            rows = json.load(open(args.plate_rows)) if args.plate_rows else DS.plate_ocr_dataset()
            layer1(rows, DS.plate_target(), args.out)

    print(f"[cycle] done -> {args.out}/")


if __name__ == "__main__":
    main()
