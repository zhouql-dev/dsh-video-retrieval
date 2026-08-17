#!/usr/bin/env python3
"""Shared helpers for the video-target-localize engines (locate/verify/person_search).

Keeps one implementation of the duplicated plumbing: device selection, temporal
intervals, YOLO-World person loading (set_classes-once MPS rule), the sparse
detect+crop+manifest pass, time-window dedup, best-frame annotation, and cosine.
Importing this module is cheap (cv2/numpy); torch/ultralytics load lazily inside
the functions that need them.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import numpy as np
import cv2

# lazy, heavy imports
_torch = None
def _torch_():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def skill_root():
    """Absolute path of the skill directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def models_dir():
    """Cross-platform user models dir (~/.video-analyst/models)."""
    return Path.home() / ".video-analyst" / "models"


FFMPEG_HINT = ("ffmpeg/ffprobe 未找到 / not found on PATH — "
               "安装/install: winget install ffmpeg (Windows) 或/or brew install ffmpeg (Mac)")


def find_ffmpeg(tool="ffmpeg"):
    """Return the tool path or None (callers print FFMPEG_HINT)."""
    return shutil.which(tool)


def resolve_model_path(name):
    """Resolve a model filename against, in order: the literal path (cwd-relative
    or absolute), the skill root (bundled weights), and ~/.video-analyst/models/
    (setup.py downloads). Returns the first existing candidate, else the literal
    name unchanged (lets loaders raise/emit their own missing-weight message)."""
    p = Path(name)
    if p.exists():
        return str(p)
    for cand in (skill_root() / name, models_dir() / name,
                 models_dir() / p.name):
        if cand.exists():
            return str(cand)
    return name


def ensure_ultralytics_weights_dir():
    """Point ultralytics' weights_dir at the skill's bundled weights (absolute).

    Without this, WEIGHTS_DIR resolves relative to CWD; running an engine from any
    other directory makes YOLO-World's set_classes() re-download the CLIP text model
    (and fail on hosts with SSL interception / no network). Called by load_yolo_*.
    """
    try:
        from ultralytics import settings
        from ultralytics.utils import WEIGHTS_DIR
    except Exception:
        return
    target = skill_root() / "weights"
    if not target.is_dir():
        return
    try:
        cur = str(WEIGHTS_DIR)
    except Exception:
        cur = ""
    if os.path.abspath(cur) != str(target):
        try:
            settings.update({"weights_dir": str(target)})
        except Exception:
            pass


def peak_rss_mb():
    """Peak RSS in MB — best-effort metric only; never worth a crash.
    psutil (any OS) -> resource (Unix) -> 0.0 (Windows w/o psutil)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wss", mi.rss) / 1e6
    except Exception:
        pass
    if os.name != "nt":
        try:
            import platform, resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024
        except Exception:
            pass
    return 0.0


def pick_device(req=None):
    """Pick a torch device. Honors an explicit request ('auto'/None => detect):
    CUDA (Windows/Linux NVIDIA) -> MPS (Apple Silicon) -> CPU."""
    if req and req != "auto":
        return req
    try:
        t = _torch_()
        if t.cuda.is_available():
            return "cuda"
        if getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_intervals(frames, fps, gap_s=0):
    """Temporal presence intervals from a list of frame indices.

    gap_s == 0  -> contiguous-run rule (frames must be consecutive; locate.py).
    gap_s  > 0  -> gap-tolerant rule (merge frames within gap_s*fps; verify/search).
    Emits [{"start_s","end_s","frames":[lo,hi]}].
    """
    if not frames:
        return []
    frames = sorted(frames)
    out = []
    s = p = frames[0]
    if gap_s and gap_s > 0:
        gap = gap_s * fps
        for f in frames[1:]:
            if f - p <= gap:
                p = f
            else:
                out.append({"start_s": round(s / fps, 2), "end_s": round(p / fps, 2),
                            "frames": [s, p]}); s = p = f
    else:
        for f in frames[1:]:
            if f == p + 1:
                p = f
            else:
                out.append({"start_s": round(s / fps, 2), "end_s": round(p / fps, 2),
                            "frames": [s, p]}); s = p = f
    out.append({"start_s": round(s / fps, 2), "end_s": round(p / fps, 2), "frames": [s, p]})
    return out


def load_yolo_person(device=None, model_path="yolov8s-worldv2.pt"):
    """Load YOLO-World and set the 'person' class. Call ONCE per process —
    repeated set_classes crashes on MPS ('Placeholder storage ... MPS device!')."""
    ensure_ultralytics_weights_dir()
    from ultralytics import YOLOWorld
    path = resolve_model_path(model_path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"模型缺失 / model missing: {model_path} — 请运行 python setup.py (run setup.py to download weights)")
    m = YOLOWorld(path)
    m.set_classes(["person"])
    return m


def detect_and_crop(model, frame, idx, fps, crops_dir, min_area, conf=0.2,
                    imgsz=640, device="mps"):
    """Run YOLO on one frame, crop boxes with area >= min_area (8px pad), write
    JPGs to crops_dir, return manifest entries.
    Schema (identical to verify_target.py): {frame,t_s,box:[x1,y1,x2,y2],area,conf,crop}.
    Crop filename: crop_f{idx:05d}_k{k}.jpg — do NOT change (regression-sensitive)."""
    r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    entries = []
    if r.boxes is None or not len(r.boxes):
        return entries
    xy = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
    H, W = frame.shape[:2]
    for k, (x1, y1, x2, y2) in enumerate(xy):
        w, h = x2 - x1, y2 - y1; area = w * h
        if area < min_area:
            continue
        cx1, cy1 = max(0, int(x1 - 8)), max(0, int(y1 - 8))
        cx2, cy2 = min(W, int(x2 + 8)), min(H, int(y2 + 8))
        crop = frame[cy1:cy2, cx1:cx2]
        name = f"crop_f{idx:05d}_k{k}.jpg"
        cv2.imwrite(os.path.join(crops_dir, name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        entries.append({"frame": idx, "t_s": round(idx / fps, 3),
                        "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                        "area": int(area), "conf": round(float(c[k]), 3), "crop": name})
    return entries


def track_and_crop(model, frame, idx, fps, crops_dir, min_area, conf=0.2,
                   imgsz=640, device="mps", write_crop=True):
    """BoT-SORT-tracked detection: run model.track (persist across calls) and return
    per-frame entries {frame,t_s,box:[x1,y1,x2,y2],area,conf,track_id[,crop]}.

    With write_crop=False the tracker still advances (callers should call this on
    EVERY in-range frame to keep track continuity), but no crop JPG is written and
    `crop` is omitted — bounding crop/VLM cost to the step cadence while keeping
    dense tracked boxes for boxes.json / tracked-clip."""
    r = model.track(frame, persist=True, tracker="botsort.yaml", conf=conf,
                    imgsz=imgsz, device=device, verbose=False)[0]
    entries = []
    if r.boxes is None or not len(r.boxes):
        return entries
    xy = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(xy)
    H, W = frame.shape[:2]
    for k, (x1, y1, x2, y2) in enumerate(xy):
        w, h = x2 - x1, y2 - y1; area = w * h
        if area < min_area:
            continue
        e = {"frame": idx, "t_s": round(idx / fps, 3),
             "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
             "area": int(area), "conf": round(float(c[k]), 3),
             "track_id": None if (ids is None or ids[k] is None) else int(ids[k])}
        if write_crop:
            cx1, cy1 = max(0, int(x1 - 8)), max(0, int(y1 - 8))
            cx2, cy2 = min(W, int(x2 + 8)), min(H, int(y2 + 8))
            crop = frame[cy1:cy2, cx1:cx2]
            name = f"crop_f{idx:05d}_k{k}.jpg"
            cv2.imwrite(os.path.join(crops_dir, name), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            e["crop"] = name
        entries.append(e)
    return entries



def dedup_largest_per_window(manifest, window_s, fps):
    """Keep the largest-area crop per `window_s`-second bucket, sorted by frame."""
    buckets = {}
    for c in manifest:
        b = c["frame"] // (window_s * fps)
        if b not in buckets or c["area"] > buckets[b]["area"]:
            buckets[b] = c
    return sorted(buckets.values(), key=lambda c: c["frame"])


def annotate_best_frame(cap, frame_idx, box, t_s, label, out_path):
    """Seek cap to frame_idx, draw a green box + label + timestamp, write JPG.
    Mirrors verify_target.py's per-cluster best-frame annotation."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    if not ok:
        return False
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.rectangle(fr, (x1, y1 - 36), (x1 + 280, y1), (0, 255, 0), -1)
    cv2.putText(fr, str(label)[:24], (x1 + 6, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(fr, f"f{frame_idx} t={t_s}s", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(out_path, fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return True


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
