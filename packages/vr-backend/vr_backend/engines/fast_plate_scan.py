#!/usr/bin/env python3
"""Fast full-video re-scan for a specific license plate — the "VLM unavailable"
fallback that takes minutes, not an hour.

WHY THIS EXISTS
---------------
verify_target.py stage 2 (VLM) may be unavailable (unset/expired ZHIPUAI_API_KEY,
or the free-tier model can't be reached), leaving you with 100s of unverified
crops. A naive "OCR every frame everywhere" re-scan is the right idea but is
~50-75 min on a 15-min 1080p clip (tested). This script cuts that to minutes:

  1. Reuse the car boxes from verify_target.py's manifest.json (already on disk)
     instead of re-detecting.
  2. Only search for plate-colored blobs (blue/yellow/green) INSIDE each car
     ROI (plus --margin) on the RAW 1920x1080 frames — far higher resolution
     than the crops verify_target.py saved.
  3. Enlarge + equalize the blob, OCR with tesseract, and fuzzy-match the
     target against known character-confusable variants (Q<->9/O/0, G<->6,
     B<->8, I<->1, T<->7, Z<->2, S<->5 ...).

GOTCHAS LEARNED IN THE FIELD (documented in references/fast-plate-scan.md):
  * tesseract can silently fail to OPEN files depending on cwd — always write
    OCR images into the output dir and run tesseract with cwd=<that dir>.
  * OCR on the verify_target.py *crops* is useless (plate ~20px inside a
    150px crop). The RAW frame is what works.
  * Run with the venv python, like the other engines.

Usage (venv python):
  fast_plate_scan.py --video <mp4> --target Q1G728 \
                     --manifest <verify_out>/manifest.json --out <dir>
                     [--step 15] [--margin 100] [--min-plate-w 40]
                     [--everywhere]   # ignore manifest, scan whole frame
                     [--ocr-bin tesseract]

Outputs (in --out): hits.json, all_ocr.json, hit_<frame>.jpg annotated frames,
metrics.json (incl. wall-clock seconds).
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
import cv2
import numpy as np

# ---- character confusables (typical OCR errors on low-res plates) ----
# Field-tested on surveillance footage: Q->9/1, G->6/1/9, B->8, I<->1, T->7,
# Z<->2, S<->5, plus 京-char noise 'R'/'A' at the front position.
CONFUSABLES = {
    "Q": "O0D9G16", "O": "Q0D9", "0": "OQ9", "9": "Q0G16", "D": "OQ",
    "1": "ILT7Q", "I": "1L", "L": "1I", "T": "17",
    "G": "6C0Q19", "6": "GC1", "C": "G6",
    "B": "8R", "8": "B", "R": "B",
    "Z": "2", "2": "Z", "S": "5", "5": "S",
    "7": "T1", "U": "V", "V": "U",
}


def _load_confusables() -> dict:
    """Runtime-config seam (fusion harness E0): prefer an external
    confusables JSON — ``FUSION_CONFUSABLES`` env, else
    ``<cwd>/fusion/config/confusables.json`` — over the hardcoded field
    table above (self-evolution backfills that file). Any failure degrades
    silently to the hardcoded table."""
    candidates = [os.environ.get("FUSION_CONFUSABLES"),
                  os.path.join(os.getcwd(), "fusion", "config", "confusables.json")]
    for p in candidates:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            table = {str(k): str(v) for k, v in data.items() if k != "_meta"}
            if table:
                return table
        except Exception:                       # noqa: BLE001
            continue
    return CONFUSABLES


CONFUSABLES = _load_confusables()


def make_variants(target: str, min_len: int = 4, max_len: int = 7) -> dict:
    """Variant strings for every contiguous target substring in [min_len,
    max_len]. Returns {substring: {variants}}. Lengths < full plate are
    "weak" evidence (covers lost first char / merged text like Q116728728);
    full-length variants are "strong".
    """
    target = target.upper().replace(" ", "")
    n = len(target)
    out = {}
    for L in range(min_len, min(max_len, n) + 1):
        for i in range(n - L + 1):
            sub = target[i:i + L]
            variants = {""}
            for ch in sub:
                repl = set(CONFUSABLES.get(ch, "")) | {ch}
                variants = {v + r for v in variants for r in repl}
            out[sub] = {v for v in variants if len(v) == L}
    return out


def ocr_text(img_bgr, png_path, ocr_bin, ocr_dir) -> str:
    """OCR an image; runs tesseract with cwd=<ocr_dir> (field-tested gotcha)."""
    scale = max(8.0, 300.0 / max(img_bgr.shape[1], 1))
    big = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cv2.imwrite(png_path, gray)
    r = subprocess.run(
        [ocr_bin, os.path.basename(png_path), "-", "-l", "eng", "--psm", "7",
         "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
        capture_output=True, cwd=ocr_dir)
    return r.stdout.decode("utf-8", errors="replace").strip().replace(" ", "")


# ---- PP-OCRv4 backend (RapidOCR/onnxruntime, reads the province char 京/皖/苏) ----
_RAPID = None


def _rapid_engine():
    """Lazy singleton for RapidOCR (PP-OCRv4 mobile, Chinese dict). Import-guarded:
    if rapidocr_onnxruntime isn't installed the caller falls back to tesseract."""
    global _RAPID
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    return _RAPID


def ocr_text_rapid(img_bgr) -> str:
    """PP-OCRv4 recognition of a plate region. Pads the crop 25% (the det
    model needs context around small plates), upscales to ≥96px height, joins
    all recognized fragments and strips separators (space/·). Reads the Chinese
    province character, which the tesseract whitelist path cannot."""
    H, W = img_bgr.shape[:2]
    y1, y2 = max(0, int(-0.25 * H)), min(H, int(1.25 * H))
    x1, x2 = max(0, int(-0.25 * W)), min(W, int(1.25 * W))
    # pad with a gray border instead of cropping (keeps the det context)
    pad_y, pad_x = int(0.25 * H), int(0.25 * W)
    padded = cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x,
                                cv2.BORDER_CONSTANT, value=(128, 128, 128))
    scale = max(1.0, 96.0 / padded.shape[0])
    if scale != 1.0:
        padded = cv2.resize(padded, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
    res, _ = _rapid_engine()(padded)
    if not res:
        return ""
    txt = "".join(r[1] for r in res)
    return re.sub(r"[·\s\-—]", "", txt)


def find_plate_blobs(im, min_w=40):
    """Blue/yellow/green blobs with plate-ish aspect ratio."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    masks = {
        "blue": cv2.inRange(hsv, (95, 60, 60), (135, 255, 255)),
        "yellow": cv2.inRange(hsv, (18, 80, 80), (38, 255, 255)),
        "green": cv2.inRange(hsv, (40, 40, 40), (90, 255, 255)),
    }
    cands = []
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (w >= min_w and h >= 14 and 1.8 <= w / h <= 5.5
                    and cv2.contourArea(c) > 700):
                cands.append((x, y, w, h, color))
    return cands


def _ocr_backend(plate, png_path, args) -> str:
    """OCR router: PP-OCRv4 (rapid) reads Chinese province chars and is far
    more robust on low-res plates; tesseract is the offline fallback. auto =
    rapid first, tesseract when rapid is unavailable or returns nothing."""
    if args.ocr_backend in ("rapid", "auto"):
        try:
            txt = ocr_text_rapid(plate)
            if txt:
                return txt
            if args.ocr_backend == "rapid":
                return ""
        except Exception:
            if args.ocr_backend == "rapid":
                return ""
    if args.ocr_backend in ("tesseract", "auto"):
        return ocr_text(plate, png_path, args.ocr_bin, args.ocr_dir)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--target", required=True,
                    help="plate alphanumeric part, e.g. Q1G728 (no 京)")
    ap.add_argument("--manifest", default=None,
                    help="manifest.json from verify_target.py (car boxes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=10,
                    help="scan every Nth frame (plate-readable window is ~1s; "
                         "<=10 frames recommended)")
    ap.add_argument("--margin", type=int, default=100,
                    help="pad car box before searching for plates")
    ap.add_argument("--min-plate-w", type=int, default=40)
    ap.add_argument("--everywhere", action="store_true",
                    help="ignore manifest, scan whole frames (slower)")
    ap.add_argument("--ocr-bin", default="tesseract")
    ap.add_argument("--ocr-backend", default="auto", choices=["auto", "rapid", "tesseract"],
                    help="rapid = PP-OCRv4 (中文省字符, 推荐); auto = rapid 优先 tesseract 兜底")
    args = ap.parse_args()

    if not shutil.which(args.ocr_bin):
        sys.exit(f"tesseract not found on PATH — install it (brew install tesseract)")
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ocr_dir = os.path.join(args.out, "_ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    args.ocr_dir = ocr_dir

    variants = make_variants(args.target)
    n_variants = sum(len(v) for v in variants.values())
    print(f"[init] target={args.target} substrings={len(variants)} "
          f"variants={n_variants} step={args.step} everywhere={args.everywhere}")

    boxes = None
    if not args.everywhere:
        if not args.manifest:
            sys.exit("--manifest required unless --everywhere "
                     "(run verify_target.py first to get car boxes)")
        m = json.load(open(args.manifest))
        boxes = {}
        for e in m:
            boxes.setdefault(e["frame"], []).append(
                tuple(int(v) for v in e["box"]))
        print(f"[init] {len(boxes)} frames with car boxes from manifest")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def rois_for(fr, frame, allow_nearest=False):
        """Car ROIs for a frame (or whole frame with --everywhere).
        With allow_nearest (pass 2 only), falls back to the nearest frame's
        boxes (±60 frames) so fine re-scans work even with a sparse manifest
        (e.g. verify_target.py --step 30). Pass 1 stays strict to keep the
        OCR call count low."""
        if boxes and fr in boxes:
            return [(max(0, x0 - args.margin), max(0, y0 - args.margin),
                     min(W, x1 + args.margin), min(H, y1 + args.margin))
                    for (x0, y0, x1, y1) in boxes[fr]]
        if allow_nearest and boxes is not None:
            for d in range(1, 61):
                if fr - d in boxes:
                    return [(max(0, x0 - args.margin), max(0, y0 - args.margin),
                             min(W, x1 + args.margin), min(H, y1 + args.margin))
                            for (x0, y0, x1, y1) in boxes[fr - d]]
                if fr + d in boxes:
                    return [(max(0, x0 - args.margin), max(0, y0 - args.margin),
                             min(W, x1 + args.margin), min(H, y1 + args.margin))
                            for (x0, y0, x1, y1) in boxes[fr + d]]
        if args.everywhere:
            return [(0, 0, W, H)]
        return []

    def process_frame(fr, frame, rescan=False):
        """OCR plate blobs in a frame; append hits/suspicious. Returns
        (n_hits_here, n_susp_here) for progress reporting."""
        nh = ns = 0
        for (rx0, ry0, rx1, ry1) in rois_for(fr, frame, allow_nearest=rescan):
            roi = frame[ry0:ry1, rx0:rx1]
            for (x, y, w, h, color) in find_plate_blobs(roi, args.min_plate_w):
                plate = roi[y:y + h, x:x + w]
                png = os.path.join(ocr_dir, f"p_{fr}_{x}_{y}.png")
                txt = _ocr_backend(plate, png, args)
                n_ocr_holder[0] += 1
                rec = {"frame": fr, "t_s": round(fr / fps, 2), "color": color,
                       "box_abs": [rx0 + x, ry0 + y, rx0 + x + w, ry0 + y + h],
                       "ocr": txt}
                if not txt:
                    if not rescan:  # pass 1 records; pass 2 re-scans, don't re-record
                        suspicious.append(rec)
                    ns += 1
                    continue
                all_ocr.append(rec)
                # match: any substring variant present in OCR text
                best = None
                for sub, vset in variants.items():
                    for v in vset:
                        if v in txt:
                            strength = "strong" if len(sub) >= len(args.target) else "weak"
                            if best is None or (strength == "strong" and best[0] != "strong"):
                                best = (strength, sub, v)
                            break
                if best:
                    strength, sub, v = best
                    rec2 = {**rec, "matched_sub": sub, "matched_variant": v,
                            "strength": strength}
                    (hits if strength == "strong" else weak_hits).append(rec2)
                    nh += 1
                    print(f"HIT f{fr} t={fr/fps:.2f}s {color} ocr='{txt}' "
                          f"match={sub} ({strength})", flush=True)
                    # annotate frame
                    a = frame.copy()
                    cv2.rectangle(a, (rx0 + x, ry0 + y), (rx0 + x + w, ry0 + y + h),
                                  (0, 0, 255), 3)
                    # --everywhere sets boxes=None (no manifest); the car-box
                    # overlay is only available when a manifest was given.
                    for (bx0, by0, bx1, by1) in (boxes or {}).get(fr, []):
                        cv2.rectangle(a, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
                    cv2.putText(a, f"t={fr/fps:.2f}s f{fr} ocr={txt} ({strength})",
                                (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2)
                    cv2.imwrite(os.path.join(args.out, f"hit_f{fr}.jpg"), a)
        return nh, ns

    hits, weak_hits, all_ocr, suspicious = [], [], [], []
    n_ocr_holder = [0]

    # ---- Pass 1: coarse scan ----
    for fr in range(0, total, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok:
            break
        process_frame(fr, frame)
        if fr % 3000 == 0:
            print(f"  scanned {fr}/{total}  ocr_calls={n_ocr_holder[0]} "
                  f"hits={len(hits)} suspicious={len(suspicious)}", flush=True)

    # ---- Pass 2: fine re-scan around unreadable (suspicious) moments ----
    # The readable window for a plate is ~1s and sporadic (motion blur), so a
    # coarse step can land between readable frames. Cluster suspicious frames
    # in time and re-scan each cluster neighborhood at step 2.
    if suspicious and boxes is not None:
        sus_frames = sorted(set(r["frame"] for r in suspicious))
        clusters = []
        for f in sus_frames:
            if clusters and f - clusters[-1][-1] <= 3 * args.step:
                clusters[-1].append(f)
            else:
                clusters.append([f])
        print(f"[pass2] re-scanning {len(clusters)} suspicious clusters "
              f"({len(sus_frames)} frames) at step 2", flush=True)
        for cl in clusters:
            lo = max(0, cl[0] - 40)
            hi = min(total, cl[-1] + 40)
            for fr in range(lo, hi, 2):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, frame = cap.read()
                if not ok:
                    break
                process_frame(fr, frame, rescan=True)
        print(f"[pass2] done: hits={len(hits)} weak={len(weak_hits)}", flush=True)
    cap.release()

    elapsed = round(time.time() - t0)
    json.dump(hits, open(os.path.join(args.out, "hits.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(weak_hits, open(os.path.join(args.out, "weak_hits.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(all_ocr, open(os.path.join(args.out, "all_ocr.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(suspicious, open(os.path.join(args.out, "suspicious.json"), "w"),
              ensure_ascii=False, indent=2)
    metrics = {"target": args.target, "n_frames": total, "step": args.step,
               "n_ocr_calls": n_ocr_holder[0], "n_hits": len(hits),
               "n_weak_hits": len(weak_hits), "n_suspicious": len(suspicious),
               "elapsed_s": elapsed, "hits": hits}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[done] {elapsed}s wall-clock, {n_ocr_holder[0]} OCR calls, "
          f"{len(hits)} strong hits, {len(weak_hits)} weak hits, "
          f"{len(suspicious)} suspicious -> {args.out}/")


if __name__ == "__main__":
    main()
