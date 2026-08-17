#!/usr/bin/env python3
"""Person / face SEARCH-BY-IMAGE engine.

Given a reference image (a face crop OR a person photo — possibly containing
several people), find that subject across a surveillance video and output the
frames + bounding boxes where they appear. This is the image-input counterpart
to locate.py/verify_target.py (which take natural-language text).

Pipeline:
  0. Auto-select the subject from the reference image (largest face preferred,
     else largest person); save ref_subject.jpg / ref_subject_boxed.jpg /
     ref_subject.json so a wrong pick is correctable by re-cropping.
  1. (on-device) detect faces (insightface) and/or persons (YOLO) every Nth
     frame; crop close-ups; dedup one per time window.
  2. (match) cosine-embedding vs the reference subject (insightface for faces,
     OSNet/CLIP for persons), else GLM-5V image-to-image (vlm) fallback. Keep
     exact/partial matches.
  3. Build intervals + annotated best frames (schema-consistent with verify).

Backends degrade gracefully: insightface missing -> CLIP -> vlm; no embedder and
no ZHIPUAI_API_KEY -> manifest-only, exit 0. Privacy: embeddings are on-device;
only the vlm fallback sends crops to the API.

Usage (venv python; ZHIPUAI_API_KEY only for vlm fallback):
  person_search.py --video <mp4> --ref <ref.jpg> --out <dir>
                   [--mode auto|face|person] [--backend auto|embed|vlm]
                   [--step 3] [--min-area 7200] [--window 1.0]
                   [--device auto|mps|cuda|cpu]
"""
from __future__ import annotations
import argparse, json, os, sys
import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import matcher  # noqa: E402
import common  # noqa: E402
from glmv import match_image  # noqa: E402


def _embed_for(kind, crop, carried=None):
    """Embedding for a crop. For faces, prefer a carried embedding computed at
    detection time (re-detecting on a tight crop fails for small faces)."""
    if kind == "face":
        if carried is not None:
            return carried
        return matcher.embed_face(crop)
    return matcher.embed_person(crop)


def _face_detect_and_crop(img, idx, fps, crops_dir, min_area, embeds):
    """Face boxes (insightface) -> same manifest schema as common.detect_and_crop.
    Carries each face's embedding (already computed by detect_faces) into `embeds`
    keyed by crop filename, so stage 2 need not re-detect on the (small) crop."""
    entries = []
    for k, ((x1, y1, x2, y2), emb) in enumerate(matcher.detect_faces(img)):
        w, h = x2 - x1, y2 - y1; area = w * h
        if area < min_area:
            continue
        H, W = img.shape[:2]
        cx1, cy1 = max(0, int(x1 - 8)), max(0, int(y1 - 8))
        cx2, cy2 = min(W, int(x2 + 8)), min(H, int(y2 + 8))
        crop = img[cy1:cy2, cx1:cx2]
        name = f"crop_f{idx:05d}_k{k}.jpg"
        cv2.imwrite(os.path.join(crops_dir, name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if emb is not None:
            embeds[name] = emb
        entries.append({"frame": idx, "t_s": round(idx / fps, 3),
                        "box": [float(v) for v in (x1, y1, x2, y2)],
                        "area": int(area), "conf": 0.0, "crop": name})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref", required=True, help="reference image (face or person photo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["auto", "face", "person"], default="auto")
    ap.add_argument("--backend", choices=["auto", "embed", "vlm"], default="auto")
    ap.add_argument("--face-thresh", type=float, default=matcher.FACE_THRESH)
    ap.add_argument("--reid-thresh", type=float, default=matcher.REID_THRESH)
    ap.add_argument("--clip-thresh", type=float, default=matcher.CLIP_THRESH)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=7200)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="auto|mps|cuda|cpu (default: auto-detect)")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    crops_dir = os.path.join(args.out, "crops"); os.makedirs(crops_dir, exist_ok=True)
    device = common.pick_device(args.device)

    # ---- 0. reference subject ----
    ref_img = cv2.imread(args.ref)
    if ref_img is None:
        sys.exit(f"cannot read reference image: {args.ref}")
    person_model = common.load_yolo_person(device, args.model) if args.mode != "face" else None
    subj = matcher.select_ref_subject(ref_img, args.mode, person_model, device)
    kind = subj["kind"]
    qv = subj["query_vec"]
    # write ref artifacts
    cv2.imwrite(os.path.join(args.out, "ref_subject.jpg"), subj["crop"],
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    boxed = ref_img.copy()
    x1, y1, x2, y2 = subj["box"]
    cv2.rectangle(boxed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.imwrite(os.path.join(args.out, "ref_subject_boxed.jpg"), boxed,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    json.dump({"chosen_box": list(map(int, subj["box"])), "kind": kind,
               "all_subjects": subj["all"]}, open(os.path.join(args.out, "ref_subject.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- decide backend ----
    embed_ok = qv is not None and (
        (kind == "face" and matcher.FACE_OK) or
        (kind == "person" and (matcher.REID_OK or matcher.CLIP_OK)))
    if args.backend == "embed" and not embed_ok:
        print("[warn] --backend embed requested but no embedder for this kind; using vlm")
    use_embed = embed_ok and args.backend in ("auto", "embed")
    use_vlm = (not use_embed) and bool(os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY"))
    backend = ("insightface" if (kind == "face" and use_embed) else
               "osnet" if (kind == "person" and use_embed and matcher.REID_OK) else
               "clip" if use_embed else
               "vlm" if use_vlm else "none")
    thresh = (args.face_thresh if kind == "face" and backend == "insightface" else
              args.reid_thresh if backend == "osnet" else
              args.clip_thresh if backend == "clip" else 0.0)
    print(f"[init] mode={args.mode} kind={kind} backend={backend} thresh={thresh} "
          f"step={args.step} min_area={args.min_area} ref_subjects={len(subj['all'])}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 1. detect + crop ----
    manifest = []; embeds = {}; idx = 0   # embeds: crop-name -> face vec (carried)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step == 0:
            if kind == "face" and matcher.FACE_OK:
                manifest += _face_detect_and_crop(frame, idx, fps, crops_dir, args.min_area, embeds)
            elif person_model is not None:
                manifest += common.detect_and_crop(person_model, frame, idx, fps, crops_dir,
                                                   args.min_area, args.conf, args.imgsz, device)
        idx += 1
        if idx % 750 == 0:
            print(f"  [stage1] scanned {idx}/{total} frames, crops: {len(manifest)}", flush=True)
    cap.release()
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"[stage1] done: {len(manifest)} close-up crops over {idx} frames")
    dedup = common.dedup_largest_per_window(manifest, args.window, fps)
    print(f"[stage2] deduped to {len(dedup)} candidate moments (window={args.window}s)")

    # ---- 2. match ----
    results = []
    if backend == "none":
        print("[stage2] SKIPPED — no embedder and no ZHIPUAI_API_KEY. "
              "Inspect crops/manifest.json manually.")
    else:
        for i, m in enumerate(dedup):
            crop = cv2.imread(os.path.join(crops_dir, m["crop"]))
            if use_embed:
                v = _embed_for(kind, crop, embeds.get(m["crop"]))
                if v is None:
                    continue
                sim = common.cosine(qv, v)
                verdict = "exact" if sim >= thresh else ("partial" if sim >= thresh - 0.1 else "no")
                rec = {"match": verdict, "sim": round(float(sim), 4),
                       "frame": m["frame"], "t_s": m["t_s"], "box": m["box"], "crop": m["crop"]}
            else:  # vlm
                r = match_image(subj["crop"], crop) or {"match": "unclear", "conf": 0.0}
                rec = {"match": r.get("match", "unclear"), "conf": round(float(r.get("conf", 0.0)), 3),
                       "frame": m["frame"], "t_s": m["t_s"], "box": m["box"], "crop": m["crop"],
                       "note": r.get("note", "")}
            results.append(rec)
            extra = f"sim={rec.get('sim')}" if use_embed else f"conf={rec.get('conf')}"
            print(f"  [{i + 1}/{len(dedup)}] f{m['frame']} t={m['t_s']}s {rec['match']} {extra}", flush=True)
    json.dump(results, open(os.path.join(args.out, "verify.json"), "w"), ensure_ascii=False, indent=2)

    matches = [r for r in results if r.get("match") in ("exact", "partial")]
    intervals = common.build_intervals([r["frame"] for r in matches], fps, gap_s=2.0)
    json.dump(matches, open(os.path.join(args.out, "matches.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(intervals, open(os.path.join(args.out, "intervals.json"), "w"), indent=2)

    # annotated best frame per cluster
    cap = cv2.VideoCapture(args.video)
    for ci, itv in enumerate(intervals):
        best = max((r for r in matches if itv["frames"][0] <= r["frame"] <= itv["frames"][1]),
                   key=lambda r: r["box"][2] - r["box"][0], default=None)
        if not best:
            continue
        common.annotate_best_frame(cap, best["frame"], best["box"], best["t_s"],
                                   f"{kind}/{backend}",
                                   os.path.join(args.out, f"hit_cluster{ci}_f{best['frame']}.jpg"))
    cap.release()

    metrics = {"ref": args.ref, "mode": args.mode, "kind": kind, "backend": backend,
               "threshold": thresh, "ref_subject": {"box": list(map(int, subj["box"])),
               "kind": kind, "n_subjects": len(subj["all"])},
               "n_frames": idx, "n_crops": len(manifest), "n_verified": len(results),
               "n_matches": len(matches), "n_intervals": len(intervals), "intervals": intervals}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
