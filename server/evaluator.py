#!/usr/bin/env python3
"""The shared objective function — *the* answer to "how does it optimize?".

Both Phase-2 layers optimize the Phase-1 harness by repeatedly calling ONE
kind of function: *given a candidate configuration, what is its quality on a
labeled set?* This module is that function, in two flavors that share a metric:

  Layer-2 (numbers / Optuna)
      make_threshold_objective(candidates) -> f(thresholds) -> macro_f1
      Re-thresholds the cached per-candidate signal payloads (the ``raw`` dicts
      that signals.py already produced) via scorer.score_candidate, then scores
      the verdicts against each candidate's ``gt`` label. No CV, no network —
      a single evaluation is microseconds, so Optuna can run hundreds of trials.

  Layer-1 (text / gepa)
      make_text_objective(rows, target) -> f(table_text) -> macro_f1
      Parses a candidate *confusable table* (the text artifact gepa mutates),
      re-matches every labeled OCR read against the target, and returns F1.
      The matching is a faithful re-implementation of
      fast_plate_scan.make_variants so the optimized table drops straight back
      into the engine.

Why macro-F1: the reports are heavily class-imbalanced (1 true plate read vs
54 noise reads; 1 true person vs 15 OSNet "exact" FPs). Accuracy is useless
there ("say no to everything" scores 96%). macro-F1 weights the positive class
equally, which is what we actually care about — catching the rare real hit
without drowning in false positives.

The dataset contract (what ``dataset.py`` must eventually supply for real):
  candidates : [{id, t_s, gt: bool, results: {hue/temporal/vlm: {score,raw,evidence}}}]
  rows       : [{ocr: str, gt: bool, ...}]
Until the full labeled set is ready, dataset.py ships scaffolding derived from
the reports' published numbers — enough to wire and *validate* this objective,
NOT to claim an optimized result.
"""
from __future__ import annotations
import itertools
import json
from typing import Callable

import scorer

# The default confusable table — the field-tested fast_plate_scan.py:43-49 set.
# This is the Layer-1 "seed" text artifact gepa starts from.
DEFAULT_CONFUSABLES = {
    "Q": "O0D9G16", "O": "Q0D9", "0": "OQ9", "9": "Q0G16", "D": "OQ",
    "1": "ILT7Q", "I": "1L", "L": "1I", "T": "17",
    "G": "6C0Q19", "6": "GC1", "C": "G6",
    "B": "8R", "8": "B", "R": "B",
    "Z": "2", "2": "Z", "S": "5", "5": "S",
    "7": "T1", "U": "V", "V": "U",
}


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #

def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1 for the positive class."""
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def macro_f1(pos: dict, n_neg: int) -> float:
    """macro-F1 = mean of the positive-class F1 and the negative-class F1.
    The negative class here is "correctly rejecting non-targets": its TP is the
    count of true negatives (``n_neg - pos['fp']``)."""
    neg_tp = n_neg - pos["fp"]
    neg_fn = pos["fp"]                       # a false positive is a missed negative
    neg = _prf(neg_tp, pos["fn"], neg_fn)   # neg "FP" = positives we missed (fn of pos)
    return round((pos["f1"] + neg["f1"]) / 2.0, 4)


# --------------------------------------------------------------------------- #
# Layer-2 objective: thresholds  (numbers / Optuna)
# --------------------------------------------------------------------------- #

def threshold_metrics(candidates: list[dict], thresholds: dict, agree: int) -> dict:
    """Re-score every candidate with ``thresholds``/``agree`` and compare to gt."""
    thr = {**scorer.DEFAULT_THRESHOLDS, **(thresholds or {})}
    tp = fp = fn = 0
    n_neg = 0
    for c in candidates:
        verdict = scorer.score_candidate(c["results"], thr, agree)["verdict"]
        predicted_hit = verdict == "hit"
        is_hit = bool(c.get("gt", False))
        if is_hit:
            (tp := tp + 1) if predicted_hit else (fn := fn + 1)
        else:
            n_neg += 1
            if predicted_hit:
                fp += 1
    pos = _prf(tp, fp, fn)
    return {**pos, "n_pos": tp + fn, "n_neg": n_neg, "macro_f1": macro_f1(pos, n_neg),
            "agree": agree, "thresholds": thr}


def make_threshold_objective(candidates: list[dict],
                              agree: int = 2) -> Callable[[dict], float]:
    """Return f(thresholds_dict) -> macro_f1. Hand this to Optuna's objective."""
    def obj(thresholds: dict) -> float:
        return threshold_metrics(candidates, thresholds, agree)["macro_f1"]
    return obj


# --------------------------------------------------------------------------- #
# Layer-1 objective: text artifact  (confusable table / gepa)
# --------------------------------------------------------------------------- #

def parse_table(table_text: str | dict) -> dict:
    """Accept a table as a dict or a JSON string (gepa mutates text). Tolerant
    parse: on failure returns the default table so a malformed proposal simply
    scores badly rather than crashing the optimizer."""
    if isinstance(table_text, dict):
        return table_text
    try:
        d = json.loads(table_text)
        return d if isinstance(d, dict) else dict(DEFAULT_CONFUSABLES)
    except Exception:
        return dict(DEFAULT_CONFUSABLES)


def _variants(sub: str, table: dict) -> set[str]:
    """Cartesian-product confusable variants of a substring (one per char:
    the char itself ∪ its confusables), exactly as fast_plate_scan does."""
    pools = [sorted(set(table.get(ch, "") + ch)) for ch in sub]
    return {"".join(combo) for combo in itertools.product(*pools)}


def ocr_matches(target: str, table: dict, ocr: str,
                min_len: int = 4, max_len: int = 7) -> bool:
    """True iff ``ocr`` contains any confusable variant of any [min_len,max_len]
    substring of ``target``. This is the matching rule whose *table* Layer-1
    optimizes."""
    target = target.upper().replace(" ", "")
    ocr = (ocr or "").upper().replace(" ", "")
    n = len(target)
    for L in range(min_len, min(max_len, n) + 1):
        for i in range(n - L + 1):
            for v in _variants(target[i:i + L], table):
                if v and v in ocr:
                    return True
    return False


def text_metrics(rows: list[dict], target: str, table: dict | str) -> dict:
    """F1 of confusable matching over labeled OCR rows, for a candidate table."""
    table = parse_table(table)
    tp = fp = fn = 0
    n_neg = 0
    for r in rows:
        pred = ocr_matches(target, table, r["ocr"])
        gt = bool(r.get("gt", False))
        if gt:
            tp += 1 if pred else 0
            fn += 0 if pred else 1
        else:
            n_neg += 1
            fp += 1 if pred else 0
    pos = _prf(tp, fp, fn)
    return {**pos, "n_pos": tp + fn, "n_neg": n_neg, "macro_f1": macro_f1(pos, n_neg)}


def make_text_objective(rows: list[dict], target: str) -> Callable[[str | dict], float]:
    """Return f(table_text) -> macro_f1. Hand this to gepa's evaluator."""
    def obj(table_text) -> float:
        return text_metrics(rows, target, table_text)["macro_f1"]
    return obj


if __name__ == "__main__":
    # Wiring validation only (NOT an optimization run). Shows the objective is
    # sane on the scaffolding dataset: strict thresholds => few FPs; the
    # field-tested table catches the confirmed reads.
    import dataset as D
    cands = D.person_dataset()
    rows = D.plate_ocr_dataset()
    print("[Layer-2 objective] person scaffolding, agree=2:")
    for label, thr in [("defaults (§11.2)", {}),
                       ("loose (over-match)", {"hue_dhue": 120, "temporal_bellness": 0.0,
                                                "temporal_span_s": 0.0, "temporal_peak": 0.4})]:
        m = threshold_metrics(cands, thr, agree=2)
        print(f"  {label:22} macro_f1={m['macro_f1']} "
              f"P={m['precision']} R={m['recall']} tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    print("[Layer-1 objective] plate OCR scaffolding, default table:")
    m = text_metrics(rows, D.plate_target(), DEFAULT_CONFUSABLES)
    print(f"  macro_f1={m['macro_f1']} P={m['precision']} R={m['recall']} "
          f"tp={m['tp']} fp={m['fp']} fn={m['fn']} (n_pos={m['n_pos']}, n_neg={m['n_neg']})")
