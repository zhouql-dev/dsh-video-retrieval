#!/usr/bin/env python3
"""video-target-localize engine — NL query -> per-frame bbox tube + temporal intervals.

Edge pipeline (cross-platform: CUDA / MPS / CPU): YOLO-World open-vocab grounding
+ BoT-SORT tracking.
Optional --use-vlm: sparse GLM-5V-Turbo refine (text->noun) + disambiguate (pick candidate).

Self-contained; depends on the project venv (torch, ultralytics, opencv). See SKILL.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import cv2
import numpy as np
from ultralytics import YOLOWorld

from common import peak_rss_mb, pick_device, build_intervals, resolve_model_path, annotate_best_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="locate_out")
    ap.add_argument("--world-model", default="yolov8s-worldv2.pt")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frame-step", type=int, default=1,
                    help="W2: run detection+tracking every Nth frame only "
                         "(budget scheduler; skipped frames count toward "
                         "timestamps but are not scanned)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-vlm", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)

    classes = [args.query]; vlm_used = False
    # W2: 查询自动英译(refine_query)不再依赖 --use-vlm——YOLO-World 的 CLIP
    # 文本编码器对中文颜色词不敏感(实测「白色轿车」框中黑车), 英文类名才准。
    try:
        from glmv import refine_query
        refined = refine_query(args.query)
        if refined and refined != args.query:
            classes = [refined, args.query]; vlm_used = True
            print(f"[vlm] refined classes: {classes}")
    except Exception as e:
        print(f"[vlm] refine disabled ({e}); using raw query")
    if args.use_vlm:
        try:
            from glmv import disambiguate
        except Exception as e:
            print(f"[vlm] disambiguate disabled ({e})")

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
    # W2 决定性修复: fast 档(frame_step>1) 完全跳过 annotated.mp4——实测任何
    # 形式的全片编码(1080p/半分辨率)都是 wall-time 炸弹(484s/946MB),而命中帧
    # 的标注视频由 fusion_run._annotate_hits 按命中区间单独产出。证据 json
    # (boxes/intervals/metrics)不受影响,立即落盘。
    write_video = args.frame_step <= 1
    writer = None
    if write_video:
        writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    records, present_idx = [], []
    idx, seed_done, scanned = 0, False, 0
    t0 = time.time()
    while True:
        # W2: grab() skips pixel materialization for non-scanned frames —
        # full decode of every frame was the measured wall-time floor
        # (428s vs the 240s scan budget at step=4).
        ok = cap.grab()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        if args.frame_step > 1 and idx % args.frame_step != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            idx += 1
            continue
        scanned += 1
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
        if writer is not None:
            cv2.putText(frame, f"f{idx} t={idx/fps:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            writer.write(frame)
        records.append({"frame": idx, "t_s": round(idx / fps, 3), "boxes": boxes_out})
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    # W2: frame-step leaves holes between scanned frames — merge intervals
    # within a 2×step gap so parked/persistent targets don't fragment.
    gap_s = (args.frame_step / fps) * 2 if args.frame_step > 1 else 0
    intervals = build_intervals(present_idx, fps, gap_s=gap_s)
    # W2: 命中框标注——fast 档跳过了全片 annotated.mp4, 必须在此按区间补画
    # (精确寻址 annotate_best_frame; 每区间取最大框的代表帧, 最多 12 个)
    try:
        cap2 = cv2.VideoCapture(args.video)
        for ci, itv in enumerate(intervals[:12]):
            lo, hi = itv.get("frames", [itv.get("start_s", 0) and 0, 0])
            cands = [r for r in records
                     if lo <= r["frame"] <= hi and r.get("boxes")]
            best = max(cands, key=lambda r: max(
                (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1])
                for b in r["boxes"]), default=None)
            if best and best["boxes"]:
                b = max(best["boxes"], key=lambda b: (b["xyxy"][2] - b["xyxy"][0]) *
                        (b["xyxy"][3] - b["xyxy"][1]))
                annotate_best_frame(cap2, best["frame"], b["xyxy"], best["t_s"],
                                    args.query,
                                    os.path.join(args.out, f"hit_cluster{ci}_f{best['frame']}.jpg"))
        cap2.release()
    except Exception as e:
        print(f"[annotate] skipped ({e})")
    metrics = {"query": args.query, "classes": classes, "vlm_used": vlm_used, "device": device,
               "world_model": args.world_model, "tracker": args.tracker, "n_frames": idx,
               "frame_step": args.frame_step, "n_scanned_frames": scanned,
               "n_detected_frames": len(present_idx),
               "coverage": round(len(present_idx) / scanned, 3) if scanned else 0,
               "fps_video": round(fps, 3), "fps_proc": round(idx / elapsed, 2) if elapsed else 0,
               "wall_s": round(elapsed, 2), "peak_rss_mb": round(peak_rss_mb(), 1), "n_intervals": len(intervals)}
    json.dump(records, open(os.path.join(args.out, "boxes.json"), "w"), indent=2)
    json.dump(intervals, open(os.path.join(args.out, "intervals.json"), "w"), indent=2)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] outputs in {args.out}/")


if __name__ == "__main__":
    main()
