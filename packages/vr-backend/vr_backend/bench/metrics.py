#!/usr/bin/env python3
"""Benchmark metrics — ReID / OCR / detection / temporal coverage.

Independent, dependency-light implementations so they are unit-testable without
torch. Protocol notes (matching published conventions):

  * mAP / CMC-k   — standard person-ReID: rank gallery per query by similarity,
                    positives = same identity. AP averaged over queries (mAP);
                    CMC-k = fraction of queries with a positive in top-k.
  * ocr_acc       — license-plate recognition: exact match and confusable
                    match (any character confusable substitution allowed) of
                    the OCR read vs the GT plate string.
  * box_iou       — axis-aligned IoU for plate detection.
  * interval_coverage — for the two surveillance repro criteria: fraction of
                    GT presence seconds covered by predicted intervals, plus
                    the raw FP interval count (report-1's "91 intervals" metric).
"""
from __future__ import annotations
import re


# --------------------------------------------------------------------------- #
# Person ReID
# --------------------------------------------------------------------------- #

def ap_one(sims_sorted_by_rank: list[float], is_pos_sorted_by_rank: list[bool]) -> float:
    """Average Precision for one query, given gallery sorted by similarity."""
    hits = 0; acc = 0.0
    n_pos = sum(is_pos_sorted_by_rank)
    if n_pos == 0:
        return 0.0
    for k, pos in enumerate(is_pos_sorted_by_rank, 1):
        if pos:
            hits += 1
            acc += hits / k
    return acc / n_pos


def cmc_k(is_pos_sorted_by_rank: list[bool], ks=(1, 5, 10)) -> dict:
    """CMC-k for one query (whether any positive ranks within top-k)."""
    out = {}
    for k in ks:
        out[f"cmc{k}"] = int(any(is_pos_sorted_by_rank[:k]))
    return out


def reid_metrics(per_query: list[tuple[list[float], list[bool]]]) -> dict:
    """mAP + CMC over queries. Each item = (sims_ranked, is_pos_ranked)."""
    aps, cmc_acc = [], {1: 0, 5: 0, 10: 0}
    n = 0
    for sims, pos in per_query:
        if not sims:
            continue
        n += 1
        aps.append(ap_one(sims, pos))
        c = cmc_k(pos)
        for k, v in c.items():
            cmc_acc[int(k.replace("cmc", ""))] += v
    if not n:
        return {"mAP": 0.0, "CMC1": 0.0, "CMC5": 0.0, "CMC10": 0.0, "n_queries": 0}
    return {"mAP": round(sum(aps) / n, 4), "CMC1": round(cmc_acc[1] / n, 4),
            "CMC5": round(cmc_acc[5] / n, 4), "CMC10": round(cmc_acc[10] / n, 4),
            "n_queries": n}


# --------------------------------------------------------------------------- #
# License plate OCR
# --------------------------------------------------------------------------- #

def normalize_plate(s: str) -> str:
    """Uppercase, strip spaces and the province char if present (京Q1G728 ->
    Q1G728) — the engine's convention."""
    s = (s or "").upper().replace(" ", "")
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _edit_distance_1(a: str, b: str) -> bool:
    """True if a and b are equal-length and differ in at most one char."""
    if a == b:
        return True
    if len(a) != len(b):
        return False
    return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1


def ocr_match(gt: str, pred: str, confusables: dict) -> str:
    """'exact' | 'confusable' | 'miss'. Confusable = every predicted char is
    either the GT char or one of its confusable variants (handles Q->9/O/0...).
    """
    g, p = normalize_plate(gt), normalize_plate(pred)
    if not g or not p:
        return "miss"
    if g == p:
        return "exact"
    if len(g) == len(p) and all(c2 in confusables.get(c1, "") + c1 for c1, c2 in zip(g, p)):
        return "confusable"
    # partial: predicted string is a confusable substring of GT (covers reads
    # that dropped/garbled one char at the ends, like field reports).
    for L in range(max(4, len(p) - 1), len(p) + 1):
        for i in range(0, max(1, len(g) - L + 1)):
            sub = g[i:i + L]
            if len(sub) == len(p) and all(c2 in confusables.get(c1, "") + c1
                                          for c1, c2 in zip(sub, p)):
                return "confusable"
    return "miss"


def ocr_acc(gt: str, pred: str, confusables: dict) -> dict:
    """{exact, confusable_or_better, verdict} for one (gt, ocr) pair."""
    v = ocr_match(gt, pred, confusables)
    return {"exact": int(v == "exact"), "confusable": int(v in ("exact", "confusable")),
            "verdict": v}


# --------------------------------------------------------------------------- #
# Detection / temporal coverage
# --------------------------------------------------------------------------- #

def box_iou(a: list, b: list) -> float:
    """Axis-aligned IoU; boxes as [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def interval_coverage(pred: list[tuple[float, float]],
                      gt: list[tuple[float, float]]) -> dict:
    """Coverage of GT presence seconds by predicted intervals + FP count.

    Mirrors the report-1 acceptance criteria: candidate windows must cover the
    real GT moments (580s, 758-760s) while emitting fewer FP intervals than the
    91 the naive OSNet run produced. Returns:
      {gt_covered_frac, gt_hit (each gt interval covered>=0.5), n_pred, n_gt,
       covered_seconds, gt_seconds}"""
    def merge(segs):
        out = []
        for s, e in sorted(segs):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    p = merge([list(x) for x in pred]) if pred else []
    g = merge([list(x) for x in gt]) if gt else []
    total_gt = sum(e - s for s, e in g)
    covered = 0.0
    for gs, ge in g:
        for ps, pe in p:
            covered += max(0.0, min(ge, pe) - max(gs, ps))
    gt_hit = sum(1 for gs, ge in g
                 if any(min(ge, pe) - max(gs, ps) >= 0.5 * (ge - gs)
                        for ps, pe in p))
    return {"gt_covered_frac": round(covered / total_gt, 4) if total_gt else 0.0,
            "gt_hit": gt_hit, "n_gt": len(g), "n_pred": len(p),
            "covered_seconds": round(covered, 2), "gt_seconds": round(total_gt, 2)}


if __name__ == "__main__":
    conf = {"Q": "O0D9G16", "G": "6C0Q19", "1": "ILT7Q", "9": "Q0G16"}
    print("reid:", reid_metrics([([0.9, 0.5, 0.4], [True, False, False]),
                                 ([0.6, 0.3], [False, True])]))
    print("ocr Q1G728 vs 11G728:", ocr_acc("京Q1G728", "11G728", conf))
    print("ocr miss:", ocr_acc("Q1G728", "AN17119", conf))
    print("coverage:", interval_coverage([(580, 585), (758, 761), (400, 410)],
                                         [(580, 581), (758, 760)]))
