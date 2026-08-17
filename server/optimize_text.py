#!/usr/bin/env python3
"""Layer-1 optimizer — text-artifact optimization (gepa).

CAPABILITY ONLY — does not auto-execute. Call ``run()`` once a REAL labeled
dataset (and ``gepa`` + an LLM key) are in place.

How it optimizes (the mechanism)
--------------------------------
Layer-1 treats a *text* artifact as the parameter and uses an LLM as a
"propose-and-reflect" optimizer (label-free in gepa's general form; here we
have labels so the evaluator is hard macro-F1, which is stronger). The two
optimizable artifacts the guide names:

  1. the **license-plate confusable table** (fast_plate_scan.py:43-49) — a dict
     like {"Q":"O0D9G16", ...}. Serialized to JSON text, it is what gepa
     mutates. evaluator.ocr_matches + make_text_objective score any candidate
     table against the labeled OCR reads. This is field-tested and decisive
     (report-2: it is *why* 838-840s was locked), so it is the primary target.
  2. the **VLM prompts** (verify_target / match_image) — same loop, different
     evaluator (would call provider.verify_target per crop; needs keys).

gepa loop:
    seed = current table (text)
    for N metric calls:
        pick past candidates, show (candidate, score) log to reflection_lm
        LLM proposes a mutated table ("the last 3 missed plates where 1→Q; add 1→Q")
        evaluator scores the new table -> macro_f1
        keep the best.   ← the "text gradient" is the LLM reading oa.log

``max_metric_calls`` is the budget the guide cites (~120 = ~4-5k VLM calls,
free tier). For the table objective each call is a pure string re-match (no
VLM), so the budget goes very far; for prompt objectives each call IS a VLM
call, so the budget is the real cost.
"""
from __future__ import annotations
import json
from typing import Callable

import evaluator
from evaluator import DEFAULT_CONFUSABLES


def table_to_text(table: dict) -> str:
    """Serialize a confusable table to the JSON text gepa optimizes."""
    return json.dumps(table, ensure_ascii=False, sort_keys=True)


def run(rows: list[dict], target: str, *, seed_table: dict = None,
        reflection_lm: str = "glm-5.1", max_metric_calls: int = 120,
        **gepa_kwargs) -> dict:
    """Optimize the confusable table with gepa. REQUIRES ``gepa`` and an LLM
    key. Returns {"best_table": dict, "best_text": str, "best_macro_f1": float}.

    Not auto-executed. Until gepa is installed the import below raises a clear
    ImportError describing exactly what to install; the evaluator it would use
    is already wired and unit-tested via evaluator.make_text_objective.
    """
    try:
        import gepa
    except ImportError as e:
        raise ImportError(
            "Layer-1 optimizer needs gepa — pip install gepa. "
            "(Capability + evaluator are in place; only the search is blocked.)") from e

    seed_text = table_to_text(seed_table or DEFAULT_CONFUSABLES)
    objective: Callable[[str], float] = evaluator.make_text_objective(rows, target)

    # gepa.optimize_anything: label-free, custom evaluator + the (candidate,
    # score) log acts as the "text gradient" for reflection_lm. Argument names
    # follow gepa's documented API; if a future gepa version renames them, the
    # TypeError below pinpoints the one line to adjust.
    try:
        result = gepa.optimize_anything(
            task_evaluator=objective,
            seed_pool=[seed_text],
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            **gepa_kwargs,
        )
    except TypeError as e:
        raise TypeError(
            f"gepa.optimize_anything signature mismatch ({e}). gepa's API may "
            f"have changed — adjust the kwargs in optimize_text.run().") from e

    # gepa returns a rich result; extract the best-scoring candidate text.
    best_text = getattr(result, "best_candidate", None) or (
        result.get("best_candidate") if isinstance(result, dict) else seed_text)
    best_f1 = objective(best_text)
    return {"best_table": evaluator.parse_table(best_text), "best_text": best_text,
            "best_macro_f1": round(best_f1, 4), "raw": result}


def run_prompt_optimization(prompt_seed: str, eval_fn: Callable[[str], float], *,
                             reflection_lm: str = "glm-5.1", max_metric_calls: int = 120,
                             **gepa_kwargs) -> dict:
    """Optimize a VLM *prompt* (verify_target / match_image) instead of the
    table. Same gepa loop; the evaluator is supplied by the caller because it
    must call provider.verify_target per crop (needs keys) — out of scope until
    the labeled per-crop prompt dataset exists."""
    try:
        import gepa
    except ImportError as e:
        raise ImportError("Layer-1 optimizer needs gepa — pip install gepa.") from e
    result = gepa.optimize_anything(
        task_evaluator=eval_fn, seed_pool=[prompt_seed],
        reflection_lm=reflection_lm, max_metric_calls=max_metric_calls, **gepa_kwargs)
    best = getattr(result, "best_candidate", None) or (
        result.get("best_candidate") if isinstance(result, dict) else prompt_seed)
    return {"best_prompt": best, "best_score": round(float(eval_fn(best)), 4), "raw": result}


if __name__ == "__main__":
    # Wiring check only (NOT an optimization run): confirm the table objective
    # behaves on the scaffolding, and that a hand-tweaked table changes the F1.
    import dataset as D
    rows = D.plate_ocr_dataset()
    obj = evaluator.make_text_objective(rows, D.plate_target())
    print("default table  :", obj(DEFAULT_CONFUSABLES))
    # a degraded table that forgets 1->Q should recall worse (misses "11G728").
    degraded = {k: v for k, v in DEFAULT_CONFUSABLES.items() if k != "1"}
    print("table w/o 1->? :", obj(degraded))
    print("(run() is capability-only; call it with a real dataset + gepa + key.)")
