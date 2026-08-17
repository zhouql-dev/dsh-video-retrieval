#!/usr/bin/env python3
"""Multi-signal scorer — the §11.2 voting layer.

Takes the per-signal results from ``signals.py`` and applies an **agreement
rule**: a candidate is a hit only if at least ``--agree`` *independent* signals
fire. This is the direct productization of the report-1 finding that any single
channel at low resolution is unreliable (OSNet 16 "exact" → 15 false positives;
VLM all-yes hallucination) but their *intersection* is sharp.

Design choices forced by the field data
---------------------------------------
  * **Thresholds live here, not in signals.py.** The §11.2 defaults (Δhue≈35°,
    REID 0.55→0.65+) and the Optuna Phase-2 sweep all touch *this* dict, so the
    signal math is stable. Every threshold is CLI-overridable.
  * **Abstention.** A signal that genuinely cannot judge (a low-color crop for
    hue; no VLM key for vlm_arbiter) *abstains* — it neither fires nor counts
    toward the denominator. With three signals and ``--agree 2``, two real
    confirmations still hit even if the VLM is offline; a lone signal can never
    hit. ``insufficient`` is reported when too few signals remain to reach the
    bar (e.g. no VLM + a gray crop), rather than silently passing or failing.
  * **Pluggable rules.** ``SIGNAL_RULES`` maps a signal name to a
    ``(fired, abstain)`` predicate over its ``raw`` payload, so a new signal
    (an embedding peak, a CLIP text-score, …) slots in without touching the
    voting core.
"""
from __future__ import annotations
from copy import deepcopy

# §11.2 + report defaults. Phase-2 (Optuna) sweeps these.
DEFAULT_THRESHOLDS = {
    "hue_dhue": 35.0,          # Δhue° at which hue still "fires" (FPs sit at 60–110°)
    "temporal_peak": 0.55,     # minimum OSNet-peak for the curve signal to count
    "temporal_bellness": 0.40, # min sustained-elevation fraction (spike ≈ 0)
    "temporal_span_s": 0.5,    # minimum above-floor span (rejects 1-frame spikes)
    "vlm_match": ("exact", "partial"),  # VLM verdicts that count as a fire
    "reid_sim": 0.55,          # embedding cosine gate (PRW/image-search signal)
    # generic fallback for any ad-hoc signal that only exposes ``score``
    "score_floor": 0.5,
}

# default minimum number of firing (non-abstaining) signals to declare a hit
DEFAULT_AGREE = 2

# E0 runtime-config seam: fusion/config/thresholds.json layers OVER these
# hardcoded defaults at import time (evolution backfills the file; CLI flags
# still win per-call). Missing/malformed config degrades silently to the
# table above — zero behavior change when no config dir exists.
try:
    from config import threshold_overrides  # type: ignore
    _runtime_thr = threshold_overrides()
    if _runtime_thr:
        DEFAULT_THRESHOLDS = {**DEFAULT_THRESHOLDS, **_runtime_thr}
except Exception:                       # noqa: BLE001 — degrade to hardcoded
    pass


def _rule_hue(res, thr):
    raw = res.get("raw", {})
    if not raw.get("colorful", False) or raw.get("dhue") is None:
        return False, True                      # abstain: hue not discriminative
    return float(raw["dhue"]) <= thr["hue_dhue"], False


def _rule_temporal(res, thr):
    raw = res.get("raw", {})
    if raw.get("n", 0) < 3:
        return False, True                      # abstain: too few samples
    fired = (float(raw.get("peak", 0.0)) >= thr["temporal_peak"]
             and float(raw.get("bellness", 0.0)) >= thr["temporal_bellness"]
             and float(raw.get("span_s", 0.0)) >= thr["temporal_span_s"])
    return fired, False


def _rule_vlm(res, thr):
    m = str(res.get("raw", {}).get("match", "unclear"))
    if m in ("unavailable", "skipped"):
        return False, True                      # abstain: no provider / not arbitrated
    return m in thr["vlm_match"], False


def _rule_reid(res, thr):
    """Embedding cosine gate (image search / PRW benchmark). Fires iff the raw
    similarity clears ``reid_sim`` — the "REID 0.55→?" knob from the reports."""
    sim = res.get("raw", {}).get("sim")
    if sim is None:
        return False, True                      # abstain: no embedding computed
    return float(sim) >= thr["reid_sim"], False


def _rule_generic(res, thr):
    return float(res.get("score", 0.0)) >= thr["score_floor"], False


SIGNAL_RULES = {
    "hue": _rule_hue,
    "temporal": _rule_temporal,
    "vlm": _rule_vlm,
    "reid": _rule_reid,
}


def score_candidate(results: dict, thresholds: dict | None = None,
                    agree: int = DEFAULT_AGREE) -> dict:
    """Vote over one candidate's per-signal results.

    ``results`` maps signal name -> ``{score, evidence, raw}`` (as returned by
    the ``signals`` module). Returns:

        {"hit": bool, "verdict": "hit"|"miss"|"insufficient",
         "score": float, "agree": agree, "fired": [names], "abstained": [names],
         "eligible": [names], "signals": [{name, score, fired, abstain, evidence, raw}]}
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows, fired, abstained, eligible = [], [], [], []
    for name, res in results.items():
        raw = res.get("raw", {})
        rule = SIGNAL_RULES.get(name, _rule_generic)
        is_fired, is_abstain = rule(res, thr)
        if not is_abstain:
            eligible.append(name)
            if is_fired:
                fired.append(name)
        else:
            abstained.append(name)
        rows.append({"name": name, "score": float(res.get("score", 0.0)),
                     "fired": bool(is_fired and not is_abstain),
                     "abstain": bool(is_abstain), "evidence": res.get("evidence", ""),
                     "raw": raw})

    # Consensus score = mean score over eligible (non-abstaining) signals, so it
    # is comparable across candidates regardless of how many abstained.
    eligible_scores = [r["score"] for n, r in zip([x["name"] for x in rows], rows)
                       if n in eligible]
    consensus = round(sum(eligible_scores) / len(eligible_scores), 3) if eligible_scores else 0.0

    if len(eligible) < agree:
        verdict = "insufficient"                # can't reach the bar -> don't decide
        hit = False
    else:
        hit = len(fired) >= agree
        verdict = "hit" if hit else "miss"
    return {"hit": hit, "verdict": verdict, "score": consensus, "agree": agree,
            "fired": fired, "abstained": abstained, "eligible": eligible,
            "signals": rows}


def score_all(candidates: list[dict], thresholds: dict | None = None,
              agree: int = DEFAULT_AGREE) -> list[dict]:
    """Score a list of ``{id, t_s, results, ...}`` candidates; attaches the
    verdict in place and returns the list sorted by score desc."""
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = []
    for c in candidates:
        v = score_candidate(c.get("results", {}), thr, agree)
        merged = deepcopy(c)
        merged.update({"verdict": v["verdict"], "hit": v["hit"], "score": v["score"],
                       "fired": v["fired"], "abstained": v["abstained"],
                       "eligible": v["eligible"], "signals": v["signals"]})
        out.append(merged)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


if __name__ == "__main__":
    import signals as S
    import numpy as np
    import cv2
    def swatch(h_deg, s=160, v=180):
        h = int(h_deg / 2)
        return cv2.cvtColor(np.full((40, 40, 3), [h, s, v], np.uint8), cv2.COLOR_HSV2BGR)
    pink = swatch(330)
    # A genuine hit: same hue, smooth bell curve, VLM says exact.
    hit = {"id": "t438", "t_s": 438.0, "results": {
        "hue": S.hue_consistency(pink, swatch(332)),
        "temporal": S.temporal_curve([(t, s) for t, s in
                                      zip(range(5), [.35, .55, .83, .55, .35])]),
        "vlm": {"score": 1.0, "raw": {"match": "exact", "conf": 1.0},
                "evidence": "VLM exact conf=1.00"}}}
    # A false positive: different hue, spike, VLM says no.
    fp = {"id": "t120", "t_s": 120.0, "results": {
        "hue": S.hue_consistency(pink, swatch(120)),
        "temporal": S.temporal_curve([(0, .30), (1, .82), (2, .31)]),
        "vlm": {"score": 0.0, "raw": {"match": "no", "conf": 0.1},
                "evidence": "VLM no"}}}
    for v in score_all([fp, hit]):
        print(f"{v['id']} t={v['t_s']} -> {v['verdict']} score={v['score']} "
              f"fired={v['fired']} abstained={v['abstained']}")
