#!/usr/bin/env python3
"""IJB-A face benchmark — 1:1 verification (TAR@FAR) + 1:N identification (Rank).

Protocol (official 10-split):
  1:1  — template pairs from verify_comparisons_N.csv; GT = same SUBJECT_ID.
         Face crop per file from the metadata box when present (else insightface
         detects in the full image); template embedding = mean of its file
         embeddings (ArcFace via insightface). Scores = cosine; TAR@FAR swept
         over all pair scores (FAR measured on different-subject pairs).
  1:N  — gallery/probe templates from search_gallery/search_probe_N.csv;
         Rank-1/5/10 (any same-subject gallery template in top-k).

SOTA reference (2019): TAR@FAR=0.001 ≈ 0.92, Rank-1 ≈ 0.98.

Run (venv python; needs insightface + onnxruntime):
  "$VENV" bench/run_ijba.py --out bench_out/ijba [--split 1] [--all-splits]
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

import datasets as D   # noqa: E402


def embed_template(matcher, cv2, numpy, files):
    """Mean ArcFace embedding over a template's files. Crop by metadata face
    box when given (cheap, no re-detection); else detect in the full image."""
    vecs = []
    for f in files:
        img = cv2.imread(f["file"])
        if img is None:
            continue
        if f.get("bbox"):
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            H, W = img.shape[:2]
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(W, x2 + 10), min(H, y2 + 10)
            crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img
        else:
            crop = img
        v = matcher.embed_face(crop)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    m = vecs[0]
    for v in vecs[1:]:
        m = m + v
    return m / len(vecs)


def tars_at_far(scores: list[tuple[float, bool]]) -> dict:
    """TAR@FAR=0.001/0.01/0.1 by sweeping a threshold over pair scores."""
    import numpy as np
    diff = np.array([s for s, same in scores if not same], dtype=np.float64)
    same = np.array([s for s, same_ in scores if same_], dtype=np.float64)
    if diff.size == 0 or same.size == 0:
        return {"TAR@FAR=0.001": None, "TAR@FAR=0.01": None, "TAR@FAR=0.1": None}
    out = {}
    for far in (0.001, 0.01, 0.1):
        k = max(1, int(np.ceil(far * diff.size)))
        thresh = float(np.sort(diff)[-k])          # threshold with FAR <= target
        out[f"TAR@FAR={far}"] = round(float((same > thresh).mean()), 4)
    return out


def run_11(matcher, cv2, numpy, root, split):
    data = D.load_ijba_11(root, split)
    print(f"[ijba] 1:1 split{split}: {len(data['templates'])} templates, "
          f"{len(data['pairs'])} pairs")
    t0 = time.time()
    emb = {}
    for tid, t in data["templates"].items():
        v = embed_template(matcher, cv2, numpy, t["files"])
        if v is not None:
            emb[tid] = v
    print(f"[ijba] embedded {len(emb)}/{len(data['templates'])} templates "
          f"({time.time()-t0:.0f}s)")
    scores = []
    for p in data["pairs"]:
        if p["t1"] in emb and p["t2"] in emb:
            s = float(numpy.dot(emb[p["t1"]], emb[p["t2"]]) /
                      (numpy.linalg.norm(emb[p["t1"]]) * numpy.linalg.norm(emb[p["t2"]])))
            scores.append((s, p["same"]))
    res = {"protocol": "1:1", "split": split, "n_pairs_scored": len(scores),
           "n_same": sum(1 for _, s in scores if s), "n_diff": sum(1 for _, s in scores if not s),
           **tars_at_far(scores)}
    return res


def run_1n(matcher, cv2, numpy, root, split):
    data = D.load_ijba_1n(root, split)
    print(f"[ijba] 1:N split{split}: {len(data['gallery'])} gallery, "
          f"{len(data['probe'])} probe templates")
    t0 = time.time()
    emb = {}
    for tid, t in data["templates"].items():
        v = embed_template(matcher, cv2, numpy, t["files"])
        if v is not None:
            emb[tid] = v
    print(f"[ijba] embedded {len(emb)} templates ({time.time()-t0:.0f}s)")
    G = [(tid, emb[tid]) for tid in data["gallery"] if tid in emb]
    ranks, scored = [], 0
    for tid in data["probe"]:
        if tid not in emb:
            continue
        subj = data["templates"][tid]["subject_id"]
        sims = sorted(((float(numpy.dot(emb[tid], gv) /
                               (numpy.linalg.norm(emb[tid]) * numpy.linalg.norm(gv))), gt)
                       for gt, gv in G), key=lambda x: -x[0])
        pos = [data["templates"][gt]["subject_id"] == subj for _, gt in sims]
        scored += 1
        ranks.append(next((k for k, p in enumerate(pos, 1) if p), None))
    n = scored or 1
    res = {"protocol": "1:N", "split": split, "n_probe_scored": scored,
           "Rank-1": round(sum(1 for r in ranks if r == 1) / n, 4),
           "Rank-5": round(sum(1 for r in ranks if r is not None and r <= 5) / n, 4),
           "Rank-10": round(sum(1 for r in ranks if r is not None and r <= 10) / n, 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--protocol", choices=["11", "1n", "both"], default="both")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matcher
    import cv2
    import numpy
    if not matcher.FACE_OK:
        raise SystemExit("insightface unavailable (FACE_OK=False) — pip install "
                         "insightface onnxruntime, then re-run")
    splits = range(1, 11) if args.all_splits else [args.split]
    all_res = []
    for sp in splits:
        if args.protocol in ("11", "both"):
            r = run_11(matcher, cv2, numpy, args.dataset, sp)
            all_res.append(r)
            print(json.dumps(r, indent=2))
        if args.protocol in ("1n", "both"):
            r = run_1n(matcher, cv2, numpy, args.dataset, sp)
            all_res.append(r)
            print(json.dumps(r, indent=2))
    json.dump(all_res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"[ijba] done -> {args.out}/metrics.json")


if __name__ == "__main__":
    main()
