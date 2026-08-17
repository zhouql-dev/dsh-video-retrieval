#!/usr/bin/env python3
"""Labeled datasets for the Phase-2 evolution layer.

STATUS: SCAFFOLDING. The production "两份报告 crops + 标注" dataset is NOT yet
in place — what's here is the *schema* plus small sets derived from the
reports' published numbers / the existing eval artifacts, enough to wire and
unit-test ``evaluator.py`` and the two optimizers. Do NOT read an optimized
result off this scaffolding; it exists so the optimization *capability* can be
validated before the real labels arrive. Swap in a real loader (same return
shape) when the dataset is ready.

The two eval reports give us *ground truth* — the moments the target genuinely
appears — which turns the harness into an optimizable objective: given a config
(prompt / confusable table / thresholds), how many real hits does it catch and
how many false positives does it let through? macro-F1 over a labeled candidate
set is that objective, and BOTH Layer-1 (gepa, text) and Layer-2 (Optuna,
numbers) plug into the very same evaluator (``evaluator.py``).

Two datasets, matching the two reports:

person_dataset()    §11.2 person retrieval. Encoded from report-1's concrete
   numbers: 16 OSNet "exact" candidates — ONE true subject (peak sim 0.834 at
   t≈438s, Δhue small, smooth 0.35→0.83→0.35 curve, VLM exact conf 1.0) and 15
   false positives (high sim but wrong hue 60–110°, single-frame spikes, VLM
   "no"). This is the canonical "any single channel fails, intersection wins"
   case, and it is what Layer-2 sweeps thresholds against. Each candidate
   carries the SAME ``raw`` payload shape that ``signals.py`` produces, so the
   scorer re-thresholds them exactly as it would on live output.

plate_ocr_dataset()  report-2 plate retrieval. Loads the real OCR reads from
   ``vtl_fastscan_out/all_ocr.json`` and labels each read positive/negative
   against the report's GT (the target plate physically appears at 838–840s,
   with confirmed reads like ``11G728``/``16728``/``Q1G728``). This is what
   Layer-1 (gepa) optimizes the confusable table against — cheap, no VLM calls.

The person set is generated with a fixed seed so a sweep is reproducible; edit
``PERSON_SEED`` or the constants to stress-test a different regime.
"""
from __future__ import annotations
import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
# eval artifacts live one dir up (the repo root).
_REPO = os.path.dirname(_HERE)
PLATE_OCR_JSON = os.path.join(_REPO, "vtl_fastscan_out", "all_ocr.json")

# report-2 ground truth: target plate "京Q1G728" appears at 838–840s.
PLATE_TARGET = "Q1G728"
PLATE_GT_WINDOW_S = (838.0, 840.0)
# reads the report manually confirmed as genuine target reads (not noise), used
# as labels independent of whatever confusable table is being optimized.
PLATE_POSITIVE_READS = {"11G728", "16728", "916728", "Q1G728", "Q116728728"}

PERSON_SEED = 42


# --------------------------------------------------------------------------- #
# Person §11.2 dataset
# --------------------------------------------------------------------------- #

def _person_candidate(cid, t_s, dhue, peak, bellness, span_s, n, vlm_match,
                      vlm_conf, gt, ref_hue=325.0):
    """Build one candidate carrying the same ``raw`` shape signals.py emits, so
    ``scorer.score_candidate`` can re-threshold it without re-running CV."""
    crop_hue = (ref_hue + dhue) % 360
    return {
        "id": cid, "t_s": t_s, "gt": gt,
        "results": {
            "hue": {"score": round(max(0.0, 1.0 - dhue / 110.0), 3),
                    "raw": {"dhue": dhue, "ref_hue": ref_hue, "crop_hue": round(crop_hue, 1),
                            "colorful": True},
                    "evidence": f"Δhue={dhue:.0f}°"},
            "temporal": {"score": round(0.45 * min(1, peak / 0.8) + 0.55 * bellness, 3),
                         "raw": {"peak": peak, "bellness": bellness, "span_s": span_s, "n": n},
                         "evidence": f"peak={peak} bell={bellness} span={span_s}"},
            "vlm": {"score": {"exact": 1.0, "partial": 0.6, "unclear": 0.3, "no": 0.0}[vlm_match],
                    "raw": {"match": vlm_match, "conf": vlm_conf},
                    "evidence": f"VLM {vlm_match}"},
        },
    }


def person_dataset(n_false_positives: int = 15) -> list[dict]:
    """The §11.2 person-retrieval candidate set (1 TP + N FPs).

    Faithful to report-1: the true subject peaks at sim 0.834 (t≈438s) with a
    smooth rise-fall and small Δhue; the false positives have OSNet sims in the
    0.55–0.72 "exact" band (so a sim-only rule flags all 16) but disagree on hue
    (60–110°) and curve shape (spikes) and are vetoed by the VLM. A few "hard"
    FPs where ONE signal fires demonstrate why ``--agree 2`` is the right bar.
    """
    rng = random.Random(PERSON_SEED)
    cands = [_person_candidate("t438", 438.0, dhue=12.0, peak=0.834, bellness=0.5,
                               span_s=2.0, n=5, vlm_match="exact", vlm_conf=1.0, gt=True)]
    for i in range(n_false_positives):
        # most FPs: high sim, wrong hue, spike, vlm=no
        dhue = rng.uniform(58, 112)
        peak = rng.uniform(0.55, 0.72)
        # ~20% are "hard": a real-looking curve (would fire temporal) but wrong hue+vlm
        hard = (i % 5 == 0)
        bellness = rng.uniform(0.45, 0.6) if hard else rng.uniform(0.0, 0.25)
        span_s = rng.uniform(1.0, 2.0) if hard else 0.0
        vlm = "no"
        cands.append(_person_candidate(f"fp{i}", 120.0 + i * 23.0, dhue, peak, bellness,
                                       span_s, n=rng.choice([3, 5]), vlm_match=vlm,
                                       vlm_conf=rng.uniform(0.0, 0.2), gt=False))
    return cands


# --------------------------------------------------------------------------- #
# Plate OCR dataset (real reads from the eval artifacts)
# --------------------------------------------------------------------------- #

def plate_ocr_dataset(path: str = PLATE_OCR_JSON) -> list[dict]:
    """Load the real OCR reads and label each by the report's GT.

    Label = positive iff the read is one the report confirmed as a genuine
    target read (``PLATE_POSITIVE_READS``) — independent of the confusable
    table under optimization, so optimizing the table against these labels is
    legitimate (not circular). Falls back to a small embedded sample if the
    artifact is absent, so the optimizer is demoable anywhere."""
    rows = []
    if os.path.exists(path):
        for r in json.load(open(path)):
            txt = (r.get("ocr") or "").upper().replace(" ", "")
            rows.append({"ocr": txt, "t_s": r.get("t_s"), "frame": r.get("frame"),
                         "gt": txt in PLATE_POSITIVE_READS})
    if not rows:                       # embedded fallback (report-confirmed reads)
        sample = [("11G728", 838.56, True), ("16728", 838.08, True),
                  ("Q1G728", 838.92, True), ("419071G57", 68.4, False),
                  ("AN17119", 120.0, False), ("WAUHLOTZS", 839.0, False),
                  ("2A5", 200.0, False), ("562", 305.0, False)]
        for txt, t, gt in sample:
            rows.append({"ocr": txt, "t_s": t, "frame": None, "gt": gt})
    return rows


def plate_target() -> str:
    return PLATE_TARGET


if __name__ == "__main__":
    p = person_dataset()
    npos = sum(c["gt"] for c in p)
    print(f"person_dataset: {len(p)} candidates ({npos} TP, {len(p)-npos} FP)")
    print("  true hit:", {k: p[0]["results"][k]["raw"] for k in ("hue", "temporal", "vlm")})
    print(f"  example FP hue dhue:", p[1]["results"]["hue"]["raw"]["dhue"])
    rows = plate_ocr_dataset()
    npos = sum(r["gt"] for r in rows)
    print(f"\nplate_ocr_dataset: {len(rows)} reads ({npos} positive, {len(rows)-npos} negative)")
    print("  positives:", sorted(r["ocr"] for r in rows if r["gt"]))
