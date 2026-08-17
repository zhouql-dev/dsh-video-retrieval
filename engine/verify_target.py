#!/usr/bin/env python3
"""Two-stage retrieve -> verify engine for SPECIFIC-identifier targets
(a particular license plate, a person in distinctive clothes, etc.).

Why this exists alongside locate.py: locate.py grounds a free-form class with
YOLO-World, picks ONE seed crop on the first detected frame (VLM disambiguate),
then tracks blindly with BoT-SORT. In a scene with several similar objects
(multiple white cars at an intersection) the single seed often latches onto the
wrong one and the tracker drifts across vehicles -> bogus high coverage and a
wrong tube. It cannot re-read the identifier on later frames.

This engine instead:
  Stage 1 (local, on-device): detect the object CLASS everywhere (clean noun,
       e.g. "car"), keep close-up boxes (where an identifier is readable), crop.
  Stage 2 (sparse VLM): GLM-5V-Turbo verifies EACH candidate crop against the
       full target description and reads the identifier (e.g. plate). Keep only
       exact/partial matches. Build intervals from the verified frames.

Privacy: only the close-up crops are sent to the API (sparse), never the raw
video. If ZHIPUAI_API_KEY is unset, stage 2 is skipped (manifest still written).

Outputs (in --out): manifest.json, verify.json (per-crop verdicts),
matches.json (hits only), intervals.json, metrics.json, and annotated JPGs for
each matched cluster's best frame.

Usage (venv python; ZHIPUAI_API_KEY for stage 2):
  verify_target.py --video <mp4> --query "白色车牌号 京P3LD03 的小轿车" --out <dir>
                   [--noun car] [--step 3] [--min-area 14400] [--window 1.0]
                   [--device auto|mps|cuda|cpu]
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys
import cv2
from ultralytics import YOLOWorld

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from glmv import refine_query, verify_target  # noqa: E402
from common import (build_intervals, detect_and_crop, dedup_largest_per_window,  # noqa: E402
                    annotate_best_frame, pick_device, resolve_model_path)


def extract_plate_target(query: str):
    """Pull the plate alphanumerics out of a natural-language query.
    '车牌号 京Q1G728 的小轿车' -> 'Q1G728'; 'white plate 京P3LD03' -> 'P3LD03'.
    Returns None if no plausible plate string found."""
    m = re.search(r"京\s*([A-Z][A-Z0-9]{5,6})", query)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z][A-Z0-9]{5,6})\b", query.upper().replace("京", " 京"))
    return m.group(1) if m else None


def vlm_unusable(results) -> bool:
    """VLM stage failed silently (bad key / quota / network), i.e. results are
    almost all 'unclear' with no reads, vs. legitimately 'no' verdicts."""
    if not results:
        return True  # key unset -> stage 2 skipped entirely
    hits = sum(1 for r in results if r.get("match") in ("exact", "partial"))
    unclear = sum(1 for r in results if r.get("match") == "unclear"
                  and not r.get("read") and not r.get("note"))
    return hits == 0 and unclear >= max(3, len(results) // 2)


def run_ocr_degrade(video, query, manifest_path, out, step=10):
    """Spawn fast_plate_scan.py (local tesseract OCR re-scan) as a fallback
    when the VLM stage is unusable. Returns (hits, weak, ocr_out_dir) or None
    if it can't run (no tesseract / no extractable plate target)."""
    target = extract_plate_target(query)
    if not target:
        print("[degrade] SKIPPED — could not extract a plate string from query: "
              f"{query!r}")
        return None
    if not shutil.which("tesseract"):
        print("[degrade] SKIPPED — tesseract not found on PATH "
              "(brew install tesseract); no OCR fallback possible")
        return None
    ocr_out = os.path.join(out, "_degraded_ocr")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fast_plate_scan.py")
    print(f"[degrade] VLM stage unusable -> local OCR re-scan "
          f"(fast_plate_scan.py, target={target}, out={ocr_out})")
    r = subprocess.run([sys.executable, script, "--video", video,
                        "--target", target, "--manifest", manifest_path,
                        "--out", ocr_out, "--step", str(step)],
                       capture_output=True, text=True)
    if r.stdout:
        for ln in r.stdout.splitlines():
            print(f"  [ocr] {ln}", flush=True)
    if r.stderr:
        for ln in r.stderr.splitlines()[-5:]:
            print(f"  [ocr-err] {ln}", flush=True)
    hits_path = os.path.join(ocr_out, "hits.json")
    weak_path = os.path.join(ocr_out, "weak_hits.json")
    if not os.path.exists(hits_path):
        print("[degrade] OCR re-scan produced no hits.json — treating as "
              "target absent")
        return [], [], ocr_out
    hits = json.load(open(hits_path))
    weak = json.load(open(weak_path)) if os.path.exists(weak_path) else []
    # move annotated frames up to the main output dir
    for f in os.listdir(ocr_out):
        if f.startswith("hit_f"):
            src = os.path.join(ocr_out, f)
            dst = os.path.join(out, f)
            if not os.path.exists(dst):
                os.replace(src, dst)
    return hits, weak, ocr_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True, help="full target description incl. identifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--noun", default=None, help="YOLO class noun; if omitted, refine from query")
    ap.add_argument("--step", type=int, default=3, help="detect every Nth frame")
    ap.add_argument("--min-area", type=int, default=14400, help="min box area px^2 to crop (close-up)")
    ap.add_argument("--window", type=float, default=1.0, help="dedup crops per N-second window")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="auto|mps|cuda|cpu (default: auto-detect)")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops"); os.makedirs(crops_dir, exist_ok=True)
    device = pick_device(args.device)

    noun = args.noun or refine_query(args.query) or "object"
    classes = [noun, "vehicle"]
    print(f"[init] device={device} noun={noun} step={args.step} min_area={args.min_area}")
    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        sys.exit(f"模型缺失 / model missing: {args.model} — 请运行 python setup.py (run setup.py to download weights)")
    model = YOLOWorld(model_path)
    model.set_classes(classes)  # call ONCE; repeated set_classes crashes on MPS

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Stage 1: detect + crop close-ups ----
    manifest = []; idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            manifest += detect_and_crop(model, frame, idx, fps, crops_dir, args.min_area,
                                        args.conf, args.imgsz, device)
        idx += 1
        if idx % 750 == 0:
            print(f"  [stage1] scanned {idx}/{total} frames, crops: {len(manifest)}", flush=True)
    cap.release()
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"[stage1] done: {len(manifest)} close-up crops over {idx} frames")

    # dedup: largest-area crop per time window (one VLM call per ~moment)
    dedup = dedup_largest_per_window(manifest, args.window, fps)
    print(f"[stage2] deduped to {len(dedup)} candidate moments (window={args.window}s)")

    # ---- Stage 2: VLM verify each moment ----
    key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
    results = []
    if not key:
        print("[stage2] SKIPPED — ZHIPUAI_API_KEY unset (no VLM verification). "
              "Inspect crops/manifest.json manually.")
    else:
        for i, m in enumerate(dedup):
            crop = cv2.imread(os.path.join(crops_dir, m["crop"]))
            v = verify_target(args.query, crop) or {"match": "unclear"}
            v["frame"] = m["frame"]; v["t_s"] = m["t_s"]; v["box"] = m["box"]; v["crop"] = m["crop"]
            results.append(v)
            print(f"  [{i + 1}/{len(dedup)}] f{m['frame']} t={m['t_s']}s match={v.get('match')} "
                  f"read={v.get('read', '')} | {v.get('note', '')}", flush=True)
    json.dump(results, open(os.path.join(args.out, "verify.json"), "w"), ensure_ascii=False, indent=2)

    matches = [r for r in results if r.get("match") in ("exact", "partial")]

    # ---- Degrade: VLM stage unusable -> local OCR re-scan ----
    degraded = False
    ocr_weak = []
    if vlm_unusable(results):
        print(f"[stage2] VLM stage unusable ({len(results)} results, "
              f"{len(matches)} hits) — attempting local OCR fallback")
        ocr_res = run_ocr_degrade(args.video, args.query,
                                  os.path.join(args.out, "manifest.json"),
                                  args.out)
        if ocr_res is not None:
            degraded = True
            ocr_hits, ocr_weak, ocr_out = ocr_res
            for h in ocr_hits:
                matches.append({
                    "match": "exact" if h.get("strength") == "strong" else "partial",
                    "read": h.get("ocr", ""), "frame": h["frame"], "t_s": h["t_s"],
                    "box": h.get("box_abs", [0, 0, 0, 0]), "crop": "",
                    "verified_by": "ocr", "note": (f"本地OCR降级核验: {h.get('ocr')} "
                                                   f"matched={h.get('matched_sub')} "
                                                   f"({h.get('strength')})"),
                })
            json.dump(ocr_weak, open(os.path.join(args.out, "ocr_weak_hits.json"),
                                     "w"), ensure_ascii=False, indent=2)
            print(f"[degrade] OCR fallback: {len(ocr_hits)} strong hits, "
                  f"{len(ocr_weak)} weak hits (see ocr_weak_hits.json; "
                  f"annotated frames in {args.out}/hit_f*.jpg)")

    match_frames = [r["frame"] for r in matches]
    intervals = build_intervals(match_frames, fps, gap_s=2.0)
    json.dump(matches, open(os.path.join(args.out, "matches.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(intervals, open(os.path.join(args.out, "intervals.json"), "w"), indent=2)

    # annotated best frame per cluster + OSD region crop (top-right) for wall-clock
    cap = cv2.VideoCapture(args.video)
    for ci, itv in enumerate(intervals):
        best = max((r for r in matches if itv["frames"][0] <= r["frame"] <= itv["frames"][1]),
                   key=lambda r: r["box"][2] - r["box"][0], default=None)
        if not best:
            continue
        annotate_best_frame(cap, best["frame"], best["box"], best["t_s"], args.query,
                            os.path.join(args.out, f"hit_cluster{ci}_f{best['frame']}.jpg"))
    cap.release()

    metrics = {"query": args.query, "noun": noun, "n_frames": idx, "n_crops": len(manifest),
               "n_verified": len(results), "n_matches": len(matches), "n_intervals": len(intervals),
               "degraded_to_ocr": degraded,
               "ocr_weak_hits": len(ocr_weak),
               "intervals": intervals}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
