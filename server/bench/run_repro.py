#!/usr/bin/env python3
"""Acceptance-line reproduction — the two eval reports, full cloud-edge.

Line 1 (person, report-1): 丰台 15-min clip, 80×280 reference crop.
    GT presence: t≈580s and t≈758–760s. Acceptance (EXECUTION-GUIDE):
    candidate windows must cover BOTH GT moments and emit fewer FP intervals
    than the naive OSNet run's 91. Fusion path: `run.py --ref` (cloud
    grounding + local YOLO/OSNet + three-signal vote). The 445MB local file
    exceeds GROUND_VIDEO_MAX_MB, so grounding auto-falls-back to a full scan —
    which itself exercises the "cloud-empty -> full scan" rule.

Line 2 (plate, report-2): same clip, target 京Q1G728. GT presence:
    t≈837.9–840.0s. Acceptance: hits lock that window. Fusion router sends it
    down precise_text -> fast_plate_scan (pure local OCR + confusable table),
    cloud-independent by design.

Requires: DASHSCOPE_API_KEY / ZHIPUAI_API_KEY exported (cloud-edge), the venv
python, and the two Desktop inputs. Outputs: JSON metrics + report for each
line in --out.

Run:
  "$VENV" bench/run_repro.py --out bench_out/repro [--person-only|--plate-only]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
for p in (_HERE, _FUSION):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrics as M  # noqa: E402

VIDEO = "/Users/zhouql1978_1/Desktop/52丰台康宁居小区东侧河边路口_26891F18_1603352385_1.mp4"
REF = "/Users/zhouql1978_1/Desktop/截屏2026-08-02 下午7.04.15.png"
YOLO_MODEL = "/Users/zhouql1978_1/dev/video-retrieval/yolov8s-worldv2.pt"
PERSON_QUERY = "穿粉红色外套、白色长裤、白色运动鞋、黑发扎起的成年女性"
PERSON_GT = [(580.0, 580.5), (758.0, 760.0)]       # report-1 GT moments
PERSON_NAIVE_FP = 91                                # the baseline we must beat
PLATE_QUERY = "车牌号 京Q1G728 的小轿车"
PLATE_TARGET = "Q1G728"
PLATE_GT = [(837.9, 840.0)]                         # report-2 GT window


def run_sub(cmd, out_dir, env=None):
    """Run a command, stream tail to stdout, return (returncode, stdout)."""
    os.makedirs(out_dir, exist_ok=True)
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for ln in (r.stdout or "").splitlines()[-12:]:
        print("    | " + ln)
    if r.returncode != 0:
        print("    ! rc=%d" % r.returncode)
        for ln in (r.stderr or "").splitlines()[-6:]:
            print("    ! " + ln)
    return r.returncode, r.stdout or ""


def person_line(out, vlm_topk=60, hue_dhue=None, temporal_peak=None, gap_s=None,
                temporal_bellness=None, min_track_frames=None):
    """Fusion three-signal run with the reference image; then acceptance.
    Threshold overrides (hue_dhue / temporal_peak) enable the scene-adapted
    settings the optimization loop found (Δhue 45° / peak 0.45 for this clip)."""
    out_dir = os.path.join(out, "person")
    t0 = time.time()
    cmd = [sys.executable, os.path.join(_FUSION, "run.py"),
           "--video", VIDEO, "--query", PERSON_QUERY, "--ref", REF,
           "--out", out_dir, "--step", "10", "--vlm-topk", str(vlm_topk),
           "--agree", "2", "--curve-s", "4", "--model", YOLO_MODEL]
    if hue_dhue is not None:
        cmd += ["--hue-dhue", str(hue_dhue)]
    if temporal_peak is not None:
        cmd += ["--temporal-peak", str(temporal_peak)]
    if gap_s is not None:
        cmd += ["--gap-s", str(gap_s)]
    if temporal_bellness is not None:
        cmd += ["--temporal-bellness", str(temporal_bellness)]
    if min_track_frames is not None:
        cmd += ["--min-track-frames", str(min_track_frames)]
    rc, _ = run_sub(cmd, out_dir)
    intervals_path = os.path.join(out_dir, "intervals.json")
    intervals = json.load(open(intervals_path)) if os.path.exists(intervals_path) else []
    pred = [(i["start_s"], i["end_s"]) for i in intervals]
    cov = M.interval_coverage(pred, PERSON_GT)
    res = {"line": "person", "query": PERSON_QUERY, "gt": PERSON_GT,
           "n_pred_intervals": len(pred), "naive_baseline_fp": PERSON_NAIVE_FP,
           "coverage": cov, "accept": {
               "covers_gt": cov["gt_hit"] == len(PERSON_GT),
               "fp_below_naive": len(pred) < PERSON_NAIVE_FP,
               "pass": cov["gt_hit"] == len(PERSON_GT) and len(pred) < PERSON_NAIVE_FP},
           "wall_s": round(time.time() - t0, 1), "rc": rc}
    json.dump(res, open(os.path.join(out_dir, "acceptance.json"), "w"),
              ensure_ascii=False, indent=2)
    return res


def plate_line(out):
    """Precise-text branch: local OCR + confusable table over the full clip."""
    out_dir = os.path.join(out, "plate")
    t0 = time.time()
    rc, _ = run_sub([sys.executable, os.path.join(_FUSION, "run.py"),
                     "--video", VIDEO, "--query", PLATE_QUERY,
                     "--out", out_dir, "--step", "10"],
                    out_dir)
    hits_path = os.path.join(out_dir, "hits.json")
    hits = json.load(open(hits_path)) if os.path.exists(hits_path) else []
    hit_ts = [h.get("t_s") for h in hits if h.get("t_s") is not None]
    in_win = [t for t in hit_ts if PLATE_GT[0][0] <= t <= PLATE_GT[0][1]]
    res = {"line": "plate", "query": PLATE_QUERY, "target": PLATE_TARGET,
           "gt": PLATE_GT, "n_hits": len(hits), "hit_t_s": hit_ts,
           "hits_in_gt_window": len(in_win),
           "accept": {"locks_window": len(in_win) >= 1,
                      "pass": len(in_win) >= 1},
           "wall_s": round(time.time() - t0, 1), "rc": rc}
    json.dump(res, open(os.path.join(out_dir, "acceptance.json"), "w"),
              ensure_ascii=False, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm-topk", type=int, default=60)
    ap.add_argument("--hue-dhue", type=float, default=None,
                    help="scene-adapted hue threshold (optimization loop: 45.0)")
    ap.add_argument("--temporal-peak", type=float, default=None,
                    help="scene-adapted temporal peak (optimization loop: 0.45)")
    ap.add_argument("--gap-s", type=float, default=None,
                    help="interval merge tolerance (default run.py: 2.0)")
    ap.add_argument("--temporal-bellness", type=float, default=None)
    ap.add_argument("--min-track-frames", type=int, default=None)
    ap.add_argument("--person-only", action="store_true")
    ap.add_argument("--plate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    if not args.plate_only:
        results.append(person_line(args.out, args.vlm_topk,
                                   args.hue_dhue, args.temporal_peak, args.gap_s,
                                   args.temporal_bellness, args.min_track_frames))
    if not args.person_only:
        results.append(plate_line(args.out))
    json.dump(results, open(os.path.join(args.out, "acceptance.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
