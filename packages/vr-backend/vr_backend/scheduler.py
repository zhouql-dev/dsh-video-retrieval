#!/usr/bin/env python3
"""Budget scheduler (W2): video duration × user profile → engine parameters.

The field tests (video.mp4, 15:02) showed every mode blowing the red line:
the search must finish within ~the video's own duration. This module turns
(video duration, profile) into concrete engine knobs — scan step, resolution,
embedding cap, verify top-K, per-stage timeouts — instead of each engine
guessing. It is pure computation + a cheap cv2 probe: no network, no models.

Profiles
--------
fast(默认)   budget = min(duration, 10 min)   — 日常检索
standard     budget = duration                — 重要事件
deep         budget = 3×duration, only offered for ≤5 min videos

The computed step solves  frames/step / fps_proc ≤ scan_budget  so the scan
fits its slice of the budget on THIS machine (fps_proc baseline measured:
YOLO@640+BoT-SORT ≈ 18 fps on M4; 480p ≈ 30 fps).

Env overrides: VTL_PROFILE=fast|standard|deep.
"""
from __future__ import annotations

import math
import os

_FPS_PROC_640 = 18.0     # measured: M4 MPS, YOLO-World@640 + BoT-SORT
_FPS_PROC_480 = 30.0     # measured estimate at 480p
_FPS_VIDEO_DEFAULT = 25.0
# W2 calibration (video.mp4 15:02, two runs): scanned-throughput ~linear in
# step with C ≈ 0.82-1.6 scanned-fps per step unit (grab-decode floor
# dominates); use the conservative C so fast mode lands inside budget.
_SCAN_FPS_PER_STEP = 0.85

_PROFILES = {
    "fast":     {"budget_cap_s": 600, "budget_mult": 1.0,
                 "step_in_win": 3, "step_out_win": 12,
                 "imgsz_win": 640, "imgsz_out": 480,
                 "embed_cap": 2000, "verify_topk": 50},
    "standard": {"budget_cap_s": None, "budget_mult": 1.0,
                 "step_in_win": 2, "step_out_win": 6,
                 "imgsz_win": 640, "imgsz_out": 640,
                 "embed_cap": 5000, "verify_topk": 150},
    "deep":     {"budget_cap_s": None, "budget_mult": 3.0,
                 "step_in_win": 1, "step_out_win": 3,
                 "imgsz_win": 640, "imgsz_out": 640,
                 "embed_cap": None, "verify_topk": None},
}


def probe_duration(video: str) -> float:
    """Video duration in seconds via OpenCV metadata (cheap, no decode)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or _FPS_VIDEO_DEFAULT
        cap.release()
        if n and fps:
            return float(n) / float(fps)
    except Exception:
        pass
    return 0.0


def profile_name(mode: str | None = None) -> str:
    m = (mode or os.environ.get("VTL_PROFILE") or "fast").lower()
    return m if m in _PROFILES else "fast"


def step_for(duration_s: float, fps_video: float, scan_budget_s: float,
             imgsz: int) -> int:
    """Solve the scan step so the pass fits the scan budget, using the
    calibrated scanned-throughput model (grab-decode floor included)."""
    if duration_s <= 0:
        return 3
    frames = duration_s * fps_video
    if frames <= 0 or scan_budget_s <= 0:
        return 3
    step = math.ceil(frames / (_SCAN_FPS_PER_STEP * scan_budget_s))
    # 质量优先回退: 若最紧参数(3@640)就能在预算内扫完 → 用最紧参数
    # (短视频场景; 实测 7.5s 视频 step=3@640 仅需 ~9s)
    if frames / 3.0 / _FPS_PROC_640 <= scan_budget_s:
        step = 3
    return max(3, min(15, step))


def plan(video: str, mode: str | None = None,
         duration_s: float | None = None) -> dict:
    """Compute the engine parameter set for one job. Deterministic & pure."""
    prof = _PROFILES[profile_name(mode)]
    if duration_s is None:
        duration_s = probe_duration(video)
    duration_s = max(0.0, float(duration_s))

    if prof["budget_cap_s"]:
        budget = min(duration_s, prof["budget_cap_s"]) if duration_s else 600.0
    else:
        budget = duration_s * prof["budget_mult"] if duration_s else 600.0
    budget = max(120.0, budget)                     # never below 2 min

    # stage slices of the total budget
    ground_s = min(120.0, budget * 0.15)
    scan_s = budget * 0.40
    verify_s = budget * 0.25

    step = step_for(duration_s, _FPS_VIDEO_DEFAULT, scan_s, prof["imgsz_out"])
    # a scan that fits easily at full resolution gets the tighter in-window
    # step AND keeps 640p — accuracy is free when the budget allows it
    imgsz = prof["imgsz_out"]
    if duration_s and step_for(duration_s, _FPS_VIDEO_DEFAULT, scan_s,
                               prof["imgsz_win"]) <= prof["step_in_win"]:
        step = prof["step_in_win"]
        imgsz = prof["imgsz_win"]

    return {
        "mode": profile_name(mode),
        "duration_s": round(duration_s, 1),
        "budget_s": round(budget, 1),
        "step": int(step),
        "imgsz": int(imgsz),
        "embed_cap": prof["embed_cap"],
        "verify_topk": prof["verify_topk"],
        "stage_budgets": {"ground_s": round(ground_s, 1),
                          "scan_s": round(scan_s, 1),
                          "verify_s": round(verify_s, 1)},
    }


def engine_timeout(sched: dict, kind: str = "scan") -> int:
    """Per-tool watchdog seconds from the stage budgets (kill, don't wait)."""
    b = sched.get("stage_budgets", {})
    mult = {"ground": 1.2, "scan": 2.0, "verify": 1.5}.get(kind, 1.5)
    return int(max(120.0, b.get(f"{kind}_s", 300) * mult))


if __name__ == "__main__":
    import sys, json
    for v in (sys.argv[1:] or []):
        print(v, "->", json.dumps(plan(v), ensure_ascii=False, indent=1))
    if not sys.argv[1:]:
        print(json.dumps(plan("/nonexistent.mp4"), ensure_ascii=False, indent=1))
