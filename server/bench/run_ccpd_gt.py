#!/usr/bin/env python3
"""CCPD-GT (98,459) recognition-accuracy benchmark + Layer-1 GT feed.

``dataset/CCPD-GT/ccpd_ocr_data/``: every crop's TRUE plate is the filename
prefix AND the labels.txt column (verified consistent — the recognition GT
the old ccpd2019balance mirror could not provide). Difficulty subsets are
encoded in the filename (``ccpd_<subset>_``): base / challenge / tilt / fn
(雾) / blur / green / rotate.

Measures per subset + overall:
    exact_acc         normalized OCR read == true plate
    tolerant_acc      the ENGINE's own matching rule (evaluator.ocr_matches,
                      confusable table) catches the true plate
    false_equal_rate  reads that confusable-match while differing — the
                      width cost of the tolerance table

Backend: PP-OCRv4 (rapidocr) primary, tesseract fallback — the same engines
as fast_plate_scan. Exports rows.json ({ocr, plate, subset, exact, tolerant})
for the Layer-1 confusion-pair reflection (keyless, no network).

``--limit N`` samples proportionally across subsets (smoke/quick loops);
omit it for the full 98,459-image run (hours — the documented long run).
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import random
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
_REPO = os.path.dirname(_FUSION)
for p in (_HERE, _FUSION):
    if p not in sys.path:
        sys.path.insert(0, p)
_SKILL = os.environ.get(
    "SKILL_SCRIPTS",
    "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")
if _SKILL not in sys.path:
    sys.path.insert(0, _SKILL)

import evaluator as E        # noqa: E402

CCPD_DIR = os.path.join(_REPO, "dataset", "CCPD-GT", "ccpd_ocr_data")
LABELS = os.path.join(CCPD_DIR, "labels.txt")
DATA = os.path.join(CCPD_DIR, "data")
SUBSETS = ["base", "challenge", "tilt", "fn", "blur", "green", "rotate"]
DEFAULT_TABLE = E.DEFAULT_CONFUSABLES
DEFAULT_OUT = os.path.join(_FUSION, "bench_out", "ccpd_gt")


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def subset_of(filename: str) -> str:
    m = re.search(r"ccpd_(\w+)_", filename)
    return m.group(1) if m and m.group(1) in SUBSETS else "base"


def load_labels(path: str = LABELS) -> list[dict]:
    """Parse labels.txt -> [{file, plate, subset}]. The CCPD-GT contract:
    filename prefix (before first '_') == the true plate text."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fn, plate = line.split("\t", 1)
            rows.append({"file": fn, "plate": normalize(plate),
                         "subset": subset_of(fn)})
    return rows


# --------------------------------------------------------------------------- #
# pure logic (unit-testable, no OCR)
# --------------------------------------------------------------------------- #

def classify_row(plate: str, ocr: str, table: dict = None) -> tuple[bool, bool]:
    """(exact, tolerant) for one (true plate, OCR read) pair."""
    ocr_n = normalize(ocr)
    exact = ocr_n == plate
    tolerant = exact or E.ocr_matches(plate, table or DEFAULT_TABLE, ocr_n)
    return exact, tolerant


def accuracy_stats(rows: list[dict], table: dict = None) -> dict:
    """exact/tolerant accuracy per subset + overall over rows carrying
    {plate, ocr, subset}."""
    out = {}
    for key, group in [("overall", rows)] + [
            (s, [r for r in rows if r.get("subset") == s]) for s in SUBSETS]:
        if not group:
            continue
        n_exact = n_tol = 0
        for r in group:
            exact, tolerant = classify_row(r["plate"], r["ocr"], table)
            n_exact += exact
            n_tol += tolerant
        n = len(group)
        out[key] = {"n": n,
                    "exact_acc": round(n_exact / n, 4),
                    "tolerant_acc": round(n_tol / n, 4),
                    "false_equal_rate": round((n_tol - n_exact) / n, 4)}
    return out


def tolerant_metrics(rows: list[dict], table: dict) -> dict:
    """The evolve-layer gate metric over rows {plate, ocr}."""
    st = accuracy_stats(rows, table)["overall"]
    return st


def reflect_confusables(rows: list[dict], base_table: dict = None,
                        min_support: int = 2) -> dict:
    """Deterministic Layer-1 reflection from REAL GT: for every (ocr, plate)
    pair, align with difflib and count char-level confusion pairs; pairs with
    >= min_support are unioned onto the base table (both directions). No LLM,
    no network — this is the CCPD-GT data source for cycle.layer1."""
    table = dict(base_table or DEFAULT_TABLE)
    pairs: Counter = Counter()
    for r in rows:
        o, p = normalize(r.get("ocr") or ""), normalize(r.get("plate") or "")
        if not o or not p or o == p:
            continue
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, o, p, autojunk=False).get_opcodes():
            if op != "replace":
                continue
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                a, b = o[i1 + k], p[j1 + k]
                if a != b:
                    pairs[(a, b)] += 1
                    pairs[(b, a)] += 1
    added = 0
    for (a, b), n in pairs.items():
        if n >= min_support and b not in table.get(a, ""):
            table[a] = table.get(a, "") + b
            added += 1
    return table


# --------------------------------------------------------------------------- #
# OCR backends (lazy — heavy imports only when OCR actually runs)
# --------------------------------------------------------------------------- #

def _ocr_rapid(img_bgr) -> str:
    """The ENGINE's own field-tested PP-OCRv4 path (fast_plate_scan):
    25% gray pad + upscale to ≥96px height + separator strip."""
    import fast_plate_scan as FPS
    return FPS.ocr_text_rapid(img_bgr)


def _ocr_tesseract(img_bgr) -> str:
    import cv2
    import subprocess
    import tempfile
    import numpy as np
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    tmp = os.path.join(tempfile.mkdtemp(prefix="ccpd-ocr-"), "p.png")
    cv2.imwrite(tmp, gray)
    r = subprocess.run(["tesseract", tmp, "stdout", "--psm", "7",
                        "-c", "tessedit_char_whitelist="
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
                       capture_output=True, text=True, timeout=30)
    return re.sub(r"\s+", "", r.stdout or "").upper()


def ocr_image(path: str) -> str:
    """OCR one crop: PP-OCRv4 primary, tesseract fallback."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return ""
    try:
        return _ocr_rapid(img)
    except Exception:                   # noqa: BLE001 — tesseract fallback
        try:
            return _ocr_tesseract(img)
        except Exception:               # noqa: BLE001
            return ""


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #

def sample_rows(rows: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """Proportional sampling across subsets (fair per-subset stats)."""
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_subset: dict[str, list] = {}
    for r in rows:
        by_subset.setdefault(r["subset"], []).append(r)
    per = max(1, limit // max(1, len(by_subset)))
    picked = []
    for sub in SUBSETS:
        rng.shuffle(by_subset.get(sub, []))
        picked += by_subset.get(sub, [])[:per]
    if len(picked) < limit:             # top up from the remainder
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        picked += rest[: limit - len(picked)]
    return picked[:limit]


def main(argv=None):
    ap = argparse.ArgumentParser(description="CCPD-GT recognition benchmark")
    ap.add_argument("--limit", type=int, default=None,
                    help="sample N images across subsets (omit = full 98,459)")
    ap.add_argument("--subset", default=None, help="only this subset")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows = load_labels()
    if args.subset:
        rows = [r for r in rows if r["subset"] == args.subset]
    rows = sample_rows(rows, args.limit, args.seed)
    print(f"[ccpd_gt] {len(rows)} images "
          f"({'full GT' if args.limit is None else f'sampled limit={args.limit}'})")

    t0 = time.time()
    n_fail = 0
    for i, r in enumerate(rows):
        path = os.path.join(CCPD_DIR, r["file"])
        r["ocr"] = ocr_image(path) if os.path.exists(path) else ""
        r["exact"], r["tolerant"] = classify_row(r["plate"], r["ocr"])
        n_fail += (not r["ocr"])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    stats = accuracy_stats(rows)
    metrics = {"dataset": "CCPD-GT", "n_images": len(rows),
               "n_ocr_empty": n_fail, "wall_s": round(time.time() - t0, 1),
               "subsets": {s: stats[s] for s in SUBSETS if s in stats},
               "overall": stats["overall"],
               "sota_reference": {"exact_acc": ">0.99 (SOTA 车牌识别, 清晰裁剪)",
                                  "note": "对照用; 本机 PP-OCRv4 + 引擎匹配规则"}}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump([{k: r[k] for k in ("file", "plate", "subset", "ocr", "exact",
                                   "tolerant")} for r in rows],
              open(os.path.join(args.out, "rows.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[ccpd_gt] done -> {args.out}/")
    return metrics


if __name__ == "__main__":
    main()
