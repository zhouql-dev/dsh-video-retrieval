#!/usr/bin/env python3
"""Standardized speed micro-benchmark — model backend latencies on this host
(M4 / MPS). Measures steady-state (first call warm-up excluded) so the numbers
are comparable across runs and backends.

Measures:
  yolo         YOLO-World person detect per-frame (fps, imgsz 640)
  osnet        OSNet x0.25 / x1.0 person-embed latency (ms/crop)
  clip         CLIP B/32 image / text encode latency (ms)
  clip_l       CLIP ViT-L image / text encode latency (ms, CPU)
  rapidocr     PP-OCRv4 plate OCR latency (ms/plate, 300px-tall crops)
  face         insightface detect+embed latency (ms/face)

Run (venv python):  "$VENV" bench/run_speed.py --out bench_out/speed
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
_SKILL = os.environ.get("SKILL_SCRIPTS",
                        "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")
for p in (_HERE, _FUSION, _SKILL):
    if p not in sys.path:
        sys.path.insert(0, p)

REPO = os.path.join(_FUSION, "..")
PRW_QUERY = os.path.join(REPO, "dataset", "PRW-v16.04.20", "query_box",
                         "479_c1s3_016471.jpg")
CCPD_IMG = os.path.join(REPO, "dataset", "CCPD", "valn",
                        "00311781609195-90_87-194&321_296&359-292&361_192&360_192&326_292&327-10_0_26_7_30_21_33-134-7.jpg")


def bench(fn, n=20):
    fn()                                    # warm-up (model load / MPS compile)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"n": n, "mean_ms": round(1000 * sum(ts) / n, 1),
            "p50_ms": round(1000 * ts[n // 2], 1),
            "p95_ms": round(1000 * ts[int(n * 0.95)], 1)}


def bench_yolo():
    import common
    import cv2
    import numpy as np
    model = common.load_yolo_person(None, os.path.join(REPO, "yolov8s-worldv2.pt"))
    frame = np.zeros((540, 960, 3), np.uint8)
    r = bench(lambda: model.predict(frame, conf=0.2, imgsz=640, verbose=False))
    r["fps"] = round(1000 / r["mean_ms"], 1)
    return r


def bench_osnet(backend):
    import matcher
    import cv2
    img = cv2.imread(PRW_QUERY)
    r = bench(lambda: matcher.embed_person(img), n=30)
    return r


def bench_clip():
    import matcher
    import cv2
    import PIL.Image as Image
    import torch
    model, preprocess, dev = matcher._clip()
    img = cv2.imread(PRW_QUERY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(dev)
    r_img = bench(lambda: model.encode_image(t), n=10)
    import clip
    tok = clip.tokenize(["a person wearing a pink coat"]).to(dev)
    r_txt = bench(lambda: model.encode_text(tok), n=10)
    return {"image": r_img, "text": r_txt}


def bench_clip_l():
    from transformers import CLIPModel, CLIPProcessor
    import cv2
    import PIL.Image as Image
    cache = os.path.join(REPO, "weights", "clip_l_cache")
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir=cache)
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', cache_dir=cache)
    model.eval()
    img = cv2.imread(PRW_QUERY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = proc(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
    r_img = bench(lambda: model.get_image_features(t), n=5)
    tt = proc(text=["a person wearing a pink coat"], return_tensors="pt", padding=True)
    r_txt = bench(lambda: model.get_text_features(**tt), n=5)
    return {"image": r_img, "text": r_txt}


def bench_rapidocr():
    import cv2
    from fast_plate_scan import ocr_text_rapid
    img = cv2.imread(CCPD_IMG)
    bbox = os.path.basename(CCPD_IMG).split("-")[2]
    p1, p2 = bbox.split("_")
    x1, y1 = map(int, p1.split("&")); x2, y2 = map(int, p2.split("&"))
    plate = img[y1:y2, x1:x2]
    return bench(lambda: ocr_text_rapid(plate), n=10)


def bench_face():
    import matcher
    import cv2
    # IJB-A still with a metadata face box
    root = os.path.join(REPO, "dataset", "IJB-A")
    img = cv2.imread(os.path.join(root, "img", "533.JPG"))
    return bench(lambda: matcher.embed_face(img), n=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="comma list: yolo,osnet,clip,clip_l,rapidocr,face")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    skip = set(x for x in args.skip.split(",") if x)
    res = {"host": "M4 (MPS)", "note": "steady-state; first call warm-up excluded"}
    t0 = time.time()

    def guarded(key, fn, label):
        if label in skip:
            return
        try:
            res[key] = fn()
            print(f"[speed] {label}: {res[key]}")
        except Exception as e:                     # one backend failing must not
            print(f"[speed] {label}: FAILED ({type(e).__name__}: {e})")  # drop the rest

    if "osnet" not in skip:
        import os as _os
        for backend, key in (("x0_25", "osnet_x0_25"), ("x1_0", "osnet_x1_0")):
            _os.environ["OSNET_BACKEND"] = backend
            guarded(key, lambda b=backend: bench_osnet(b), key)
    guarded("yolo_detect", bench_yolo, "yolo")
    guarded("clip_b32", bench_clip, "clip")
    guarded("clip_large", bench_clip_l, "clip_l")
    guarded("rapidocr_plate", bench_rapidocr, "rapidocr")
    guarded("insightface_face", bench_face, "face")
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(args.out, "speed.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[speed] done -> {args.out}/speed.json")


if __name__ == "__main__":
    main()
