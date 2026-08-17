#!/usr/bin/env python3
"""video-target-localize engine — NL query -> per-frame bbox tube + temporal intervals.

Edge pipeline (cross-platform: CUDA / MPS / CPU): YOLO-World open-vocab grounding
+ BoT-SORT tracking.
Optional --use-vlm: sparse GLM-5V-Turbo refine (text->noun) + disambiguate (pick candidate).

Self-contained; depends on the project venv (torch, ultralytics, opencv). See SKILL.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings: common/glmv
import cv2
import numpy as np
from ultralytics import YOLOWorld

from common import peak_rss_mb, pick_device, build_intervals, resolve_model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="locate_out")
    ap.add_argument("--world-model", default="yolov8s-worldv2.pt")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-vlm", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)

    classes = [args.query]; vlm_used = False
    if args.use_vlm:
        try:
            from glmv import refine_query, disambiguate
            refined = refine_query(args.query)
            if refined:
                classes = [refined, args.query]; vlm_used = True
                print(f"[vlm] refined classes: {classes}")
        except Exception as e:
            print(f"[vlm] disabled ({e}); using raw query")

    print(f"[init] device={device} classes={classes} tracker={args.tracker}")
    world_path = resolve_model_path(args.world_model)
    if not os.path.exists(world_path):
        sys.exit(f"模型缺失 / model missing: {args.world_model} — 请运行 python setup.py (run setup.py to download weights)")
    model = YOLOWorld(world_path)
    model.set_classes(classes)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    records, present_idx = [], []
    idx, seed_done = 0, False
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        res = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf,
                           imgsz=args.imgsz, device=device, verbose=False)
        boxes_out = []; r0 = res[0]
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy(); conf = r0.boxes.conf.cpu().numpy()
            ids = r0.boxes.id.cpu().numpy() if r0.boxes.id is not None else [None] * len(xyxy)
            if args.use_vlm and not seed_done:
                try:
                    crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in xyxy]
                    pick = disambiguate(args.query, crops)
                    if pick is not None:
                        keep = np.zeros(len(xyxy), dtype=bool); keep[pick] = True
                        xyxy, conf = xyxy[keep], conf[keep]
                        ids = ids[keep] if isinstance(ids, np.ndarray) else np.array([ids[pick]])
                        seed_done = True; print(f"[vlm] seed -> candidate {pick}")
                except Exception as e:
                    print(f"[vlm] disambiguate skipped ({e})")
            for (x1, y1, x2, y2), cf, tid in zip(xyxy, conf, ids):
                boxes_out.append({"xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                                  "conf": round(float(cf), 3), "track_id": None if tid is None else int(tid)})
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[0]} {cf:.2f}", (p1[0], max(0, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if boxes_out:
            present_idx.append(idx)
        cv2.putText(frame, f"f{idx} t={idx/fps:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        writer.write(frame)
        records.append({"frame": idx, "t_s": round(idx / fps, 3), "boxes": boxes_out})
        idx += 1

    cap.release(); writer.release()
    elapsed = time.time() - t0
    intervals = build_intervals(present_idx, fps)
    metrics = {"query": args.query, "classes": classes, "vlm_used": vlm_used, "device": device,
               "world_model": args.world_model, "tracker": args.tracker, "n_frames": idx,
               "n_detected_frames": len(present_idx),
               "coverage": round(len(present_idx) / idx, 3) if idx else 0,
               "fps_video": round(fps, 3), "fps_proc": round(idx / elapsed, 2) if elapsed else 0,
               "wall_s": round(elapsed, 2), "peak_rss_mb": round(peak_rss_mb(), 1), "n_intervals": len(intervals)}
    json.dump(records, open(os.path.join(args.out, "boxes.json"), "w"), indent=2)
    json.dump(intervals, open(os.path.join(args.out, "intervals.json"), "w"), indent=2)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
