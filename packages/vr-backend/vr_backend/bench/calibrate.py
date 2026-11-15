#!/usr/bin/env python3
"""Scene-adaptive threshold calibration — productizes what the person
acceptance line needed by hand (Δhue 45° / peak 0.45 on the 丰台 scene).

Given a fusion ``scored.json`` (candidates with cached per-signal results) and
the scene's GT moments, grid-sweeps the signal thresholds + interval merge
tolerance and reports every configuration that covers ALL GT moments, ordered
by FP-interval count. Zero cloud cost: re-scoring cached signals is
microseconds, so a full sweep is seconds.

Why this exists: thresholds are scene-dependent (§11.2's 35°/0.55 were
calibrated on a scene whose true subject peaked at sim 0.83; the 丰台 scene's
true subject sits at sim≈0.47/Δhue≈40° — outside the defaults by a hair).
This command turns "re-calibrate for the scene" into a one-liner.

Run:
  "$VENV" bench/calibrate.py --scored bench_out/repro_final/person/scored.json \
      --gt "580,580.5;758,760" --out bench_out/calib_person
"""
from __future__ import annotations
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
for p in (_HERE, _FUSION):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrics as M   # noqa: E402
import scorer          # noqa: E402


def parse_gt(spec: str) -> list[tuple[float, float]]:
    return [tuple(map(float, seg.split(","))) for seg in spec.split(";") if seg]


def restore_results(cands: list[dict]) -> list[dict]:
    """scored.json rows carry `signals` (name/score/raw/evidence) — rebuild the
    {name: result} shape scorer.score_candidate consumes."""
    out = []
    for c in cands:
        if "results" in c and c["results"]:
            out.append(c)
        elif "signals" in c:
            c = dict(c)
            c["results"] = {s["name"]: {"score": s.get("score", 0.0),
                                        "raw": s.get("raw", {}),
                                        "evidence": s.get("evidence", "")}
                            for s in c["signals"]}
            out.append(c)
    return out


def hits_to_intervals(hits, fps, gap_s):
    frames = sorted(h["frame"] for h in hits)
    iv = []
    if not frames:
        return iv
    s = p = frames[0]
    for f in frames[1:]:
        if f - p <= gap_s * fps:
            p = f
        else:
            iv.append((round(s / fps, 2), round(p / fps, 2)))
            s = p = f
    iv.append((round(s / fps, 2), round(p / fps, 2)))
    return iv


def sweep(cands, gt, fps, agree=2):
    best = []
    for dhue in range(30, 66, 5):
        for peak in (0.40, 0.45, 0.50, 0.55, 0.60):
            for bell in (0.30, 0.40, 0.50):
                for gap in (2.0, 3.0, 3.5, 5.0):
                    thr = {"hue_dhue": float(dhue), "temporal_peak": peak,
                           "temporal_bellness": bell}
                    sc = scorer.score_all(cands, thr, agree)
                    hits = [c for c in sc if c["verdict"] == "hit"]
                    iv = hits_to_intervals(hits, fps, gap)
                    cov = M.interval_coverage(iv, gt)
                    if cov["gt_hit"] == len(gt):
                        best.append({"thresholds": thr, "gap_s": gap,
                                     "n_fp_intervals": cov["n_pred"],
                                     "n_hits": len(hits),
                                     "coverage": cov["gt_covered_frac"]})
    best.sort(key=lambda x: x["n_fp_intervals"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="fusion scored.json path")
    ap.add_argument("--gt", required=True, help='"start,end;start,end" seconds')
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--agree", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cands = restore_results(json.load(open(args.scored)))
    gt = parse_gt(args.gt)
    print(f"[calibrate] {len(cands)} candidates, GT={gt}")

    defaults = scorer.DEFAULT_THRESHOLDS
    d_sc = scorer.score_all(cands, {}, args.agree)
    d_hits = [c for c in d_sc if c["verdict"] == "hit"]
    d_iv = hits_to_intervals(d_hits, args.fps, 2.0)
    d_cov = M.interval_coverage(d_iv, gt)

    best = sweep(cands, gt, args.fps, args.agree)
    report = {
        "gt": gt, "n_candidates": len(cands),
        "before_defaults": {"thresholds": {"hue_dhue": defaults["hue_dhue"],
                                           "temporal_peak": defaults["temporal_peak"],
                                           "temporal_bellness": defaults["temporal_bellness"]},
                            "gap_s": 2.0, "gt_hit": d_cov["gt_hit"],
                            "n_fp_intervals": d_cov["n_pred"],
                            "coverage": d_cov["gt_covered_frac"]},
        "configs_covering_all_gt": best[:8],
        "recommended": best[0] if best else None,
    }
    json.dump(report, open(os.path.join(args.out, "calibration.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[calibrate] done -> {args.out}/calibration.json")


if __name__ == "__main__":
    main()
