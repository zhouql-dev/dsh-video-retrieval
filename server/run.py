#!/usr/bin/env python3
"""End-to-end cloud-edge fusion CLI (Phase 1).

Wires the four new modules into one command:

    router  ->  ground  ->  (local detect + crop)  ->  signals  ->  scorer

Three branches, picked by ``router.classify_query`` (the loop the two reports
forced — see router.py):

  precise_text   License plate / ID number. Cloud grounding can't read a 70×27
                 plate (returns empty), so it is NOT locked: we run the local
                 ``fast_plate_scan`` (tesseract + the field-tested confusable
                 table) over the whole video. This reproduces report-2's
                 838–840s lock.

  semantic+ref   Person/face search-by-image. The full §11.2 path: cloud
                 grounding narrows the scan window (or empty -> full video),
                 local YOLO+embedding produces candidate moments, then THREE
                 signals (hue / temporal_curve / vlm_arbiter) must agree
                 (``--agree``). This is where report-1's "16 OSNet exact, 15
                 false positives" gets cut down to the one real hit.

  semantic text  Text-only target ("a beige coat"). Cloud ground + local
                 verify_target (VLM per crop). Fewer signals (no ref image for
                 hue/vlm) so this is the verify_target.py path, windowed.

The local engines (detect/crop/embed/annotate) are reused verbatim from the
video-target-localize skill via ``SKILL_SCRIPTS`` — this harness adds routing,
grounding, and multi-signal scoring, not a re-implementation of detection.

Outputs (in --out): route.json, ground.json, matches.json, intervals.json,
metrics.json, and annotated best frames per hit cluster.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

# fusion package is one dir up from this file when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import router        # noqa: E402
import ground as ground_mod  # noqa: E402
import provider       # noqa: E402
import signals as sig  # noqa: E402
import scorer         # noqa: E402
import config         # noqa: E402 — E0 runtime-config layer (observability)

# Where the proven local engines live (common/matcher/glmv). Override with
# SKILL_SCRIPTS if the skill moves. Imported lazily so run.py stays importable
# without torch/ultralytics installed.
SKILL_SCRIPTS = os.environ.get(
    "SKILL_SCRIPTS",
    "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")
FAST_PLATE_SCAN = os.path.join(SKILL_SCRIPTS, "fast_plate_scan.py")


def _skill_import(name: str):
    """Import a module from SKILL_SCRIPTS (adding it to sys.path once)."""
    if SKILL_SCRIPTS not in sys.path:
        sys.path.insert(0, SKILL_SCRIPTS)
    return __import__(name)


def _threshold_args(args) -> dict:
    """Build the scorer threshold override dict from CLI flags (Phase-2 hooks)."""
    thr = {}
    if args.hue_dhue is not None:
        thr["hue_dhue"] = args.hue_dhue
    if args.temporal_peak is not None:
        thr["temporal_peak"] = args.temporal_peak
    if args.temporal_bellness is not None:
        thr["temporal_bellness"] = args.temporal_bellness
    return thr


# --------------------------------------------------------------------------- #
# Branch 1: precise_text — local OCR + confusable table (fast_plate_scan)
# --------------------------------------------------------------------------- #

def run_precise_text(video: str, target: str, out: str, args) -> dict:
    """Delegate to fast_plate_scan.py (the report-2 winner). Grounding is
    intentionally NOT used to lock the scan: a plate is unreadable to the
    cloud, so we scan the whole video (or a manifest's car boxes if given)."""
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(FAST_PLATE_SCAN):
        raise FileNotFoundError(f"fast_plate_scan.py not found at {FAST_PLATE_SCAN} "
                                f"(set SKILL_SCRIPTS)")
    cmd = [sys.executable, FAST_PLATE_SCAN, "--video", video,
           "--target", target, "--out", out, "--step", str(args.step)]
    if args.manifest:
        cmd += ["--manifest", args.manifest]
    else:
        cmd += ["--everywhere"]            # full-video scan — no cloud lock
    print("[precise_text] running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for ln in (r.stdout or "").splitlines():
        print(f"  [ocr] {ln}")
    if r.returncode != 0:
        print(r.stderr[-2000:] if r.stderr else "(no stderr)")
    hits = json.load(open(os.path.join(out, "hits.json"))) if os.path.exists(
        os.path.join(out, "hits.json")) else []
    common = _skill_import("common")
    fps = _video_fps(video)
    intervals = common.build_intervals([h["frame"] for h in hits], fps, gap_s=args.gap_s)
    json.dump(intervals, open(os.path.join(out, "intervals.json"), "w"), indent=2)
    return {"branch": "precise_text", "target": target, "n_hits": len(hits),
            "n_intervals": len(intervals), "intervals": intervals,
            "ocr_returncode": r.returncode}


# --------------------------------------------------------------------------- #
# Branch 2: semantic + reference image — the full §11.2 three-signal path
# --------------------------------------------------------------------------- #

def run_semantic_image(video: str, ref: str, query: str, windows: list[dict],
                       out: str, args) -> dict:
    """Cloud-grounded window + local YOLO/OSNet candidates + hue/temporal/VLM
    multi-signal voting. The §11.2 productization."""
    import cv2
    common = _skill_import("common")
    matcher = _skill_import("matcher")

    os.makedirs(out, exist_ok=True)
    crops_dir = os.path.join(out, "crops"); os.makedirs(crops_dir, exist_ok=True)
    device = common.pick_device(args.device)
    person_model = common.load_yolo_person(device, args.model)

    ref_img = cv2.imread(ref)
    if ref_img is None:
        raise FileNotFoundError(f"cannot read reference image: {ref}")
    subj = matcher.select_ref_subject(ref_img, args.mode, person_model, device)
    ref_crop = subj["crop"]
    kind = subj["kind"]
    qv = subj["query_vec"]
    cv2.imwrite(os.path.join(out, "ref_subject.jpg"), ref_crop)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ranges = ground_mod.windows_to_frames(windows, fps, total, pad_s=args.pad_s)
    full_range = [(0, max(0, total - 1))]
    print(f"[semantic+ref] kind={kind} backend={matcher.backend_label(args.mode, kind)} "
          f"ranges={ranges if len(ranges) <= 8 else f'{len(ranges)} ranges'}")

    def in_ranges(fr, rs):
        return any(lo <= fr <= hi for lo, hi in rs)

    def detect_pass(rs, tag):
        """Track (BoT-SORT) every frame + crop every step-th frame inside ranges."""
        manifest, trajectory, idx = [], [], 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if in_ranges(idx, rs):
                crop_it = idx % args.step == 0
                entries = common.track_and_crop(
                    person_model, frame, idx, fps, crops_dir, args.min_area,
                    args.conf, args.imgsz, device, write_crop=crop_it)
                trajectory += entries
                if crop_it:
                    manifest += entries
            idx += 1
            if idx % 1500 == 0:
                print(f"  [{tag}] {idx}/{total} crops={len(manifest)}", flush=True)
        return manifest, trajectory, idx

    def moments_from(manifest):
        """Embed-match each deduped moment vs the reference."""
        dedup = common.dedup_largest_per_window(manifest, args.window, fps)
        moments = []
        for m in dedup:
            crop = cv2.imread(os.path.join(crops_dir, m["crop"]))
            if crop is None:
                continue
            v = matcher.embed_person(crop) if kind == "person" else matcher.embed_face(crop)
            sim = common.cosine(qv, v) if (qv is not None and v is not None) else 0.0
            moments.append({"id": f"f{m['frame']}", "frame": m["frame"], "t_s": m["t_s"],
                            "box": m["box"], "crop": m["crop"], "sim": round(float(sim), 4),
                            "track_id": m.get("track_id"), "crop_bgr": crop})
        return moments

    def score_moments(moments, with_vlm):
        """§11.2 signals + vote. VLM arbiter is the expensive cloud signal; it
        only runs when with_vlm is set (final pass) and only arbitrates the
        top-K by sim; the rest abstain so agree=2 works on hue+temporal."""
        curve_s = args.curve_s
        topk = args.vlm_topk or len(moments)
        arbiter_idx = ({m["id"] for m in sorted(moments, key=lambda m: -m["sim"])[:topk]}
                       if with_vlm else set())
        candidates = []
        for m in moments:
            nearby = [(x["t_s"], x["sim"]) for x in moments
                      if abs(x["t_s"] - m["t_s"]) <= curve_s]
            if m["id"] in arbiter_idx:
                vlm = sig.vlm_arbiter(ref_crop, m["crop_bgr"])
            else:
                vlm = {"score": 0.0, "raw": {"match": "skipped", "conf": 0.0},
                       "evidence": ("below vlm_topk" if with_vlm else "pass-1 (pre-escalation)") + "; abstains"}
            candidates.append({"id": m["id"], "frame": m["frame"], "t_s": m["t_s"],
                               "box": m["box"], "crop": m["crop"], "sim": m["sim"],
                               "track_id": m.get("track_id"),
                               "results": {"hue": sig.hue_consistency(ref_crop, m["crop_bgr"]),
                                           "temporal": sig.temporal_curve(nearby),
                                           "vlm": vlm}})
        scored = scorer.score_all(candidates, _threshold_args(args), agree=args.agree)
        for c in scored:
            c.pop("results", None)      # keep the JSON dump compact (signals kept)
        hits = [c for c in scored if c["verdict"] == "hit"]
        if args.min_track_frames > 1:
            hits = _filter_by_track(hits, fps, args.min_track_frames, args.gap_s)
        return scored, hits

    # Stage 1: scan the cloud-grounded windows first (fast path)...
    manifest, trajectory, idx = detect_pass(ranges, "stage1")
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    moments = moments_from(manifest)
    print(f"[stage1] {len(manifest)} crops -> {len(moments)} candidate moments")
    scored, hits = score_moments(moments, with_vlm=False)

    # ...and if the windows yield nothing, ESCALATE to the remaining frames —
    # cloud grounding has recall risk (its windows missed GT before), so the
    # search must never be locked to a wrong cloud window.
    if not hits and ranges != full_range:
        covered = set()
        for lo, hi in ranges:
            covered.update(range(lo, hi + 1))
        complement = [(f, f) for f in range(0, total, args.step)
                      if f not in covered and not in_ranges(f, ranges)]
        # merge consecutive complement frames into ranges
        comp_ranges = []
        for f in [c[0] for c in complement]:
            if comp_ranges and f - comp_ranges[-1][1] <= args.step:
                comp_ranges[-1][1] = f
            else:
                comp_ranges.append([f, f])
        print(f"[stage1b] grounding window(s) yielded no hit -> escalating to "
              f"remaining {len(comp_ranges)} ranges")
        manifest2, trajectory2, idx2 = detect_pass(comp_ranges, "stage1b")
        manifest += manifest2
        trajectory += trajectory2
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
        moments += moments_from(manifest2)
        scored, hits = score_moments(moments, with_vlm=True)
    else:
        scored, hits = score_moments(moments, with_vlm=True)
    cap.release()
    json.dump([{k: v for k, v in m.items() if k != "crop_bgr"} for m in moments],
              open(os.path.join(out, "embeddings.json"), "w"), ensure_ascii=False, indent=2)

    intervals = common.build_intervals([c["frame"] for c in hits], fps, gap_s=args.gap_s)
    json.dump(scored, open(os.path.join(out, "scored.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(hits, open(os.path.join(out, "matches.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(intervals, open(os.path.join(out, "intervals.json"), "w"), indent=2)
    json.dump(trajectory, open(os.path.join(out, "boxes.json"), "w"), ensure_ascii=False, indent=2)

    _annotate_hits(video, hits, f"{kind}", out, common)
    return _metrics("semantic_image", query, windows, args, scored, hits, intervals,
                    idx, extra={"ref": ref, "kind": kind,
                                "backend": matcher.backend_label(args.mode, kind)})


# --------------------------------------------------------------------------- #
# Branch 3: semantic text-only — cloud ground + local VLM verify (windowed)
# --------------------------------------------------------------------------- #

def run_semantic_text(video: str, query: str, windows: list[dict],
                      out: str, args) -> dict:
    """verify_target.py-style two-stage engine, but windowed by cloud grounding
    (or full-video if grounding returned nothing). refine_query -> YOLO noun ->
    per-crop provider.verify_target. No ref image, so no hue/VLM-pair signals."""
    import cv2
    common = _skill_import("common")
    from ultralytics import YOLOWorld

    os.makedirs(out, exist_ok=True)
    crops_dir = os.path.join(out, "crops"); os.makedirs(crops_dir, exist_ok=True)
    device = common.pick_device(args.device)
    # cwd 根治:ultralytics weights_dir 按 cwd 解析会触发联网重下 —— 固定到 skill 权重目录
    common.ensure_ultralytics_weights_dir()
    noun = provider.refine_query(query) or "object"
    model = YOLOWorld(common.resolve_model_path(args.model))
    model.set_classes([noun, "vehicle"])          # ONCE (MPS rule)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_ranges = ground_mod.windows_to_frames(windows, fps, total, pad_s=args.pad_s)
    print(f"[semantic+text] noun={noun!r} scan_ranges={len(scan_ranges)}")

    manifest, trajectory, idx = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if any(lo <= idx <= hi for lo, hi in scan_ranges):
            crop_it = idx % args.step == 0
            entries = common.track_and_crop(model, frame, idx, fps, crops_dir,
                                            args.min_area, args.conf, args.imgsz, device,
                                            write_crop=crop_it)
            trajectory += entries
            if crop_it:
                manifest += entries
        idx += 1
    cap.release()
    dedup = common.dedup_largest_per_window(manifest, args.window, fps)
    print(f"[stage1] {len(manifest)} crops -> {len(dedup)} moments")

    results = []
    for m in dedup:
        crop = cv2.imread(os.path.join(crops_dir, m["crop"]))
        v = provider.verify_target(query, crop) or {"match": "unclear"}
        v.update({"frame": m["frame"], "t_s": m["t_s"], "box": m["box"], "crop": m["crop"],
                  "track_id": m.get("track_id")})
        results.append(v)
        print(f"  f{m['frame']} t={m['t_s']}s match={v.get('match')} read={v.get('read','')}")
    hits = [r for r in results if r.get("match") in ("exact", "partial")]
    intervals = common.build_intervals([r["frame"] for r in hits], fps, gap_s=args.gap_s)
    json.dump(results, open(os.path.join(out, "verify.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(hits, open(os.path.join(out, "matches.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(intervals, open(os.path.join(out, "intervals.json"), "w"), indent=2)
    json.dump(trajectory, open(os.path.join(out, "boxes.json"), "w"), ensure_ascii=False, indent=2)
    _annotate_hits(video, hits, noun, out, common)
    return _metrics("semantic_text", query, windows, args, results, hits, intervals, idx,
                    extra={"noun": noun})


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _filter_by_track(hits: list[dict], fps: float, min_frames: int,
                     gap_s: float) -> list[dict]:
    """Track clustering: a real subject persists across several agreeing
    frames; isolated single/dual-frame 'hits' are almost always false
    positives (bench analysis: ~28% of FPs are isolated). Cluster hit frames
    with gaps <= gap_s and keep only clusters with >= min_frames frames."""
    if not hits:
        return []
    rows = sorted(hits, key=lambda h: (h["frame"], h["id"]))
    clusters, cur = [], [rows[0]]
    for r in rows[1:]:
        if r["frame"] - cur[-1]["frame"] <= gap_s * fps:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    kept = [r for cl in clusters if len(cl) >= min_frames for r in cl]
    print(f"[track] {len(hits)} hits -> {len(kept)} in tracks of >= {min_frames} "
          f"frames ({len(clusters)} clusters, {len(clusters)-sum(1 for cl in clusters if len(cl)>=min_frames)} dropped)")
    return kept


def _video_fps(video: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _annotate_hits(video, hits, label, out, common):
    """Draw the best (largest) box per hit cluster."""
    import cv2
    cap = cv2.VideoCapture(video)
    intervals = json.load(open(os.path.join(out, "intervals.json"))) if os.path.exists(
        os.path.join(out, "intervals.json")) else []
    for ci, itv in enumerate(intervals):
        best = max((r for r in hits if itv["frames"][0] <= r["frame"] <= itv["frames"][1]),
                   key=lambda r: r["box"][2] - r["box"][0], default=None)
        if best:
            common.annotate_best_frame(cap, best["frame"], best["box"], best["t_s"],
                                       label, os.path.join(out, f"hit_cluster{ci}_f{best['frame']}.jpg"))
    cap.release()


def _metrics(branch, query, windows, args, results, hits, intervals, n_frames, extra):
    return {"branch": branch, "query": query, "provider": provider.provider_status(),
            "ground_windows": windows, "n_windows": len(windows),
            "agree": args.agree, "step": args.step, "window_s": args.window,
            "n_frames": n_frames, "n_candidates": len(results),
            "n_hits": len(hits), "n_intervals": len(intervals),
            "intervals": intervals, **extra}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_argparser():
    ap = argparse.ArgumentParser(description="cloud-edge fusion retrieval (Phase 1)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--ref", default=None, help="reference image (enables §11.2 multi-signal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default=None, help="glm|qwen (overrides VLM_PROVIDER env)")
    # detection
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--min-area", type=int, default=7200)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--mode", choices=["auto", "face", "person"], default="auto")
    # grounding
    ap.add_argument("--no-ground", action="store_true",
                    help="skip cloud grounding; force full-video local scan")
    ap.add_argument("--ground-fps", type=float, default=ground_mod.DEFAULT_FPS)
    ap.add_argument("--pad-s", type=float, default=1.0, help="pad grounded windows (s)")
    ap.add_argument("--curve-s", type=float, default=4.0,
                    help="±seconds of candidates forming each temporal curve")
    ap.add_argument("--gap-s", type=float, default=2.0,
                    help="interval merge tolerance (s) when building temporal intervals")
    ap.add_argument("--vlm-topk", type=int, default=0,
                    help="VLM-arbiter only the top-K candidates by sim (0=all); "
                         "the rest abstain on that signal (cost control)")
    ap.add_argument("--min-track-frames", type=int, default=1,
                    help="track clustering: keep hit tracks with >=N agreeing "
                         "frames (>=3 cuts isolated single-frame false positives)")
    # scoring
    ap.add_argument("--agree", type=int, default=scorer.DEFAULT_AGREE,
                    help="min firing signals to declare a hit")
    ap.add_argument("--hue-dhue", type=float, default=None)
    ap.add_argument("--temporal-peak", type=float, default=None)
    ap.add_argument("--temporal-bellness", type=float, default=None)
    # precise_text
    ap.add_argument("--manifest", default=None,
                    help="manifest.json (car boxes) for plate scan; omit = full scan")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.provider:
        os.environ["VLM_PROVIDER"] = args.provider
        provider.VLM_PROVIDER = args.provider.lower()
    os.makedirs(args.out, exist_ok=True)

    # E0 runtime config: values are already layered into scorer.DEFAULT_THRESHOLDS
    # at import (priority CLI > config > default — CLI flags flow through
    # _threshold_args); this line only makes the active layer observable.
    print("[config]", config.summary())

    route = router.classify_query(args.query)
    json.dump(route, open(os.path.join(args.out, "route.json"), "w"), ensure_ascii=False, indent=2)
    print("[route]", route)

    # Cloud grounding (semantic only; precise_text scans whole video anyway).
    windows = []
    if route["branch"] == "semantic" and not args.no_ground:
        windows = ground_mod.ground_candidates(args.video, args.query, fps=args.ground_fps)
    json.dump({"windows": windows, "provider": provider.provider_status()},
              open(os.path.join(args.out, "ground.json"), "w"), ensure_ascii=False, indent=2)

    if route["branch"] == "precise_text":
        if not route["target"]:
            sys.exit("[precise_text] no extractable identifier in query; "
                     "add a plate/ID number or rephrase as semantic.")
        metrics = run_precise_text(args.video, route["target"], args.out, args)
    elif args.ref:
        metrics = run_semantic_image(args.video, args.ref, args.query, windows, args.out, args)
    else:
        metrics = run_semantic_text(args.video, args.query, windows, args.out, args)

    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: metrics[k] for k in
                      ("branch", "provider", "n_windows", "n_hits", "n_intervals")
                      if k in metrics}, ensure_ascii=False, indent=2))
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
