#!/usr/bin/env python3
"""CCPD plate benchmark — detection IoU + OCR descriptive stats.

⚠️ The hf-mirror ``ccpd2019balance`` subset anonymizes plate text (filename
prefix is a numeric ID, NOT the printed plate), so **recognition accuracy vs
plate-text GT is impossible here**. What IS measured:

  1. detection: ``find_plate_blobs`` (the fast_plate_scan plate-localizer,
     blue/yellow/green blobs + aspect filter) vs the filename bbox GT ->
     per-image IoU, detection rate, best-IoU stats.
  2. OCR coverage: within the GT bbox, tesseract (same pipeline as the engine:
     upscale->gray->equalize, psm 7, alnum whitelist) -> read rate, output
     length distribution (plates are ~6-8 alnum), and confusable-variant
     diversity — descriptive, no accuracy claim.

Recognition-accuracy benchmarking (and Layer-1 confusable-table optimization
against text GT) is BLOCKED on this subset; it needs the official CCPD split
or a mapping file. See bench_out/ccpd*/report.md.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
for p in (_HERE, _FUSION, os.environ.get(
        "SKILL_SCRIPTS", "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import datasets as D   # noqa: E402
import metrics as M    # noqa: E402


def find_plate_blobs(im, min_w=16):
    """Mirror of fast_plate_scan.find_plate_blobs (smaller min_w for 720p)."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    masks = {"blue": cv2.inRange(hsv, (95, 60, 60), (135, 255, 255)),
             "yellow": cv2.inRange(hsv, (18, 80, 80), (38, 255, 255)),
             "green": cv2.inRange(hsv, (40, 40, 40), (90, 255, 255))}
    cands = []
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_w and h >= 10 and 1.5 <= w / h <= 7.0 and cv2.contourArea(c) > 300:
                cands.append((x, y, w, h, color))
    return cands


def ocr_plate(im, ocr_bin, tmpdir, tag):
    """OCR a plate region: PP-OCRv4 (rapid) first — reads the Chinese province
    char — with the tesseract pipeline as fallback (auto)."""
    import cv2
    try:
        from fast_plate_scan import ocr_text_rapid
        txt = ocr_text_rapid(im)
        if txt:
            return txt
    except Exception:
        pass
    scale = max(8.0, 300.0 / max(im.shape[1], 1))
    big = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    png = os.path.join(tmpdir, f"{tag}.png")
    cv2.imwrite(png, gray)
    r = subprocess.run([ocr_bin, os.path.basename(png), "-", "-l", "eng", "--psm", "7",
                        "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
                       capture_output=True, cwd=tmpdir)
    return r.stdout.decode("utf-8", "replace").strip().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="CCPD root (default dataset/CCPD)")
    ap.add_argument("--split", default="valn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max images (0=all)")
    ap.add_argument("--ocr-bin", default="tesseract")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ocr_bin = shutil.which(args.ocr_bin)
    if not ocr_bin:
        raise SystemExit(f"{args.ocr_bin} not on PATH")

    import cv2
    items = D.load_ccpd(args.dataset, args.split, args.limit)
    print(f"[ccpd] {len(items)} images, ocr_bin={ocr_bin}")
    tmpdir = os.path.join(args.out, "_ocr")
    os.makedirs(tmpdir, exist_ok=True)

    rows = []
    len_hist = {}
    t0 = time.time()
    for i, it in enumerate(items):
        img = cv2.imread(it["path"])
        if img is None:
            continue
        gt_box = it["bbox"]
        # detection: blobs over the whole frame vs the GT bbox
        blobs = find_plate_blobs(img)
        best_iou, best_box = 0.0, None
        for (x, y, w, h, color) in blobs:
            iou = M.box_iou([x, y, x + w, y + h], gt_box)
            if iou > best_iou:
                best_iou, best_box = iou, [x, y, x + w, y + h]
        # OCR within the GT bbox (recognition coverage, no text GT to score)
        x1, y1, x2, y2 = gt_box
        plate = img[y1:y2, x1:x2]
        txt = ocr_plate(plate, ocr_bin, tmpdir, f"p{i}") if plate.size else ""
        L = len(txt)
        len_hist[L] = len_hist.get(L, 0) + 1
        rows.append({"path": os.path.basename(it["path"]), "bbox_gt": gt_box,
                     "detected": bool(best_box), "det_iou": round(best_iou, 4),
                     "det_box": best_box, "det_color": (best_box and
                     next((c for (x, y, w, h, c) in blobs
                           if M.box_iou([x, y, x + w, y + h], gt_box) == best_iou), None)),
                     "ocr": txt, "ocr_len": L})
        if i and i % 1000 == 0:
            print(f"  {i}/{len(items)} elapsed={time.time()-t0:.0f}s", flush=True)

    n = len(rows)
    det = sum(1 for r in rows if r["detected"])
    det_rate = det / n if n else 0.0
    iou50 = sum(1 for r in rows if r["det_iou"] >= 0.5) / n if n else 0.0
    iou_mean = sum(r["det_iou"] for r in rows) / n if n else 0.0
    read = sum(1 for r in rows if r["ocr"])
    read_rate = read / n if n else 0.0
    res = {"n_images": n, "detection_rate": round(det_rate, 4),
           "det_iou50_rate": round(iou50, 4), "det_iou_mean": round(iou_mean, 4),
           "ocr_read_rate": round(read_rate, 4),
           "ocr_len_hist": {str(k): v for k, v in sorted(len_hist.items())},
           "wall_s": round(time.time() - t0, 1),
           "note": "text GT anonymized in ccpd2019balance subset -> recognition "
                   "accuracy NOT measurable; detection uses filename bbox GT"}
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[ccpd] done -> {args.out}/")


if __name__ == "__main__":
    main()
