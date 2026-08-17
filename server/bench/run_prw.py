#!/usr/bin/env python3
"""PRW person-search benchmark — OSNet matching under the official protocol.

Protocol: gallery = ALL GT boxes of the official test frames (we reuse the
provided detections so the number isolates MATCHING quality — the engine's
value-add; end-to-end detection quality is reported separately via
fast_plate_scan-style runs). Queries = the 2,057 official crops.

Metrics (standard person-ReID): mAP, CMC1/5/10. Plus the report-1 lens:
thresholded P/R/F1 at the current ``reid_sim`` gate (0.55) to quantify the
over-matching the reports documented.

Evolution export: writes ``candidates.json`` in the dataset.py contract — each
query × top-k gallery match carries a two-signal result set (reid similarity +
clothing-hue Δ) with GT, so Layer-2 (Optuna) can sweep reid_sim/hue thresholds
and ``--agree`` against REAL labels. (No temporal signal here: PRW is a
still-image benchmark; the scorer treats missing signals as absent.)

Run (venv python — needs torch/ultralytics/scipy):
  "$VENV" bench/run_prw.py --out bench_out/prw [--limit 100] [--topk 20]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
if _FUSION not in sys.path:
    sys.path.insert(0, _FUSION)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import metrics as M            # noqa: E402
import datasets as D           # noqa: E402
import signals as S            # noqa: E402

SKILL_SCRIPTS = os.environ.get(
    "SKILL_SCRIPTS",
    "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)


def load_engine():
    import common
    import matcher
    if not matcher.REID_OK:
        raise SystemExit(f"OSNet weights not found (matcher.OSNET_WEIGHTS="
                         f"{matcher.OSNET_WEIGHTS}); set OSNET_WEIGHTS or run setup.py")
    dev = common.pick_device(None)
    return common, matcher, dev


def embed_gallery(matcher, cv2, gallery, frames_dir, device, limit=0):
    """Group gallery by frame -> one imread per frame -> embed each crop.
    Returns parallel lists (entries, embs, hues)."""
    import numpy as np
    by_frame = {}
    for i, g in enumerate(gallery):
        if limit and i >= limit:
            break
        by_frame.setdefault(g["frame"], []).append(g)
    entries, embs, hues = [], [], []
    t0 = time.time()
    for fi, (fn, gs) in enumerate(sorted(by_frame.items())):
        img = cv2.imread(os.path.join(frames_dir, fn + ".jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        for g in gs:
            x1 = max(0, int(g["x"])); y1 = max(0, int(g["y"]))
            x2 = min(W, int(g["x"] + g["w"])); y2 = min(H, int(g["y"] + g["h"]))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = img[y1:y2, x1:x2]
            emb = matcher.embed_person(crop)
            if emb is None:
                continue
            entries.append(g)
            embs.append(emb.astype(np.float32))
            hue = S._dominant_hue(crop)[0]
            hues.append(hue)
        if fi % 1000 == 0:
            print(f"  [gallery] {fi}/{len(by_frame)} frames, "
                  f"{len(embs)} embeddings, {time.time()-t0:.0f}s", flush=True)
    return entries, np.array(embs), hues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="PRW root (default dataset/PRW-v16.04.20)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max queries (0=all 2057)")
    ap.add_argument("--max-gallery", type=int, default=0, help="cap gallery size for smoke")
    ap.add_argument("--topk", type=int, default=20, help="candidates exported per query")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    common, matcher, dev = load_engine()
    import cv2
    import numpy as np
    root = args.dataset or os.path.join(D.DATA_ROOT, "PRW-v16.04.20")
    prw = D.load_prw(root)
    frames_dir = os.path.join(root, "frames")
    print(f"[prw] gallery={len(prw['gallery'])} queries={len(prw['queries'])} device={dev}")

    t0 = time.time()
    entries, G, g_hues = embed_gallery(matcher, cv2, prw["gallery"], frames_dir, dev,
                                       limit=args.max_gallery)
    print(f"[prw] gallery embedded: {G.shape} in {time.time()-t0:.0f}s")

    # queries
    Q, q_entries, q_hues = [], [], []
    queries = prw["queries"][:args.limit] if args.limit else prw["queries"]
    for q in queries:
        crop = cv2.imread(q["crop_path"])
        if crop is None:
            continue
        emb = matcher.embed_person(crop)
        if emb is None:
            continue
        Q.append(emb.astype(np.float32))
        q_entries.append(q)
        q_hues.append(S._dominant_hue(crop)[0])
    Q = np.array(Q)
    print(f"[prw] queries embedded: {Q.shape}")

    # similarity per query (chunked to bound memory). L2-normalize both sides
    # first so ``sim`` is a COSINE in [0,1] — dot products otherwise have
    # magnitude ~norm² (≈900 for OSNet) which makes thresholds meaningless and
    # biases rankings slightly (norms vary across crops).
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-8)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True).clip(min=1e-8)
    per_query, candidates, thr_stats = [], [], {"tp": 0, "fp": 0, "fn": 0}
    for qi in range(len(Q)):
        sims = (Qn[qi] @ Gn.T).astype(np.float32)
        order = np.argsort(-sims)
        is_pos = np.array([entries[i]["id"] == q_entries[qi]["id"] for i in order], dtype=bool)
        per_query.append((sims[order].tolist(), is_pos.tolist()))

        # thresholded stats at the current reid gate
        for i in order:
            same = entries[i]["id"] == q_entries[qi]["id"]
            if sims[i] >= 0.55:
                if same:
                    thr_stats["tp"] += 1
                else:
                    thr_stats["fp"] += 1
            elif same:
                thr_stats["fn"] += 1

        # evolution export: top-k by sim + all positives (is_pos is indexed by
        # rank position, so map back through `order`)
        positive_idx = {int(order[j]) for j in range(len(order)) if is_pos[j]}
        export_idx = set(order[:args.topk].tolist()) | positive_idx
        for i in sorted(export_idx):
            dhue = S._circ_delta(q_hues[qi], g_hues[i]) if (q_hues[qi] is not None and g_hues[i] is not None) else None
            same = entries[i]["id"] == q_entries[qi]["id"]
            candidates.append({
                "id": f"q{qi}:g{i}", "query_id": q_entries[qi]["id"],
                "gallery_id": entries[i]["id"], "sim": round(float(sims[i]), 4),
                "gt": bool(same),
                "results": {
                    "reid": {"score": round(float(sims[i]), 3), "raw": {"sim": round(float(sims[i]), 4)},
                             "evidence": f"sim={sims[i]:.3f}"},
                    "hue": {"score": 1.0 if (dhue is not None and dhue <= 35) else
                                     max(0.0, 1.0 - (dhue or 999) / 110.0),
                            "raw": {"dhue": dhue, "colorful": dhue is not None},
                            "evidence": f"Δhue={dhue:.0f}°" if dhue is not None else "low-color"},
                }})
        if qi % 250 == 0:
            print(f"  [match] {qi}/{len(Q)}", flush=True)

    res = M.reid_metrics(per_query)
    prec = thr_stats["tp"] / (thr_stats["tp"] + thr_stats["fp"]) if (thr_stats["tp"] + thr_stats["fp"]) else 0.0
    rec = thr_stats["tp"] / (thr_stats["tp"] + thr_stats["fn"]) if (thr_stats["tp"] + thr_stats["fn"]) else 0.0
    res["threshold_0.55"] = {"precision": round(prec, 4), "recall": round(rec, 4),
                             "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                             **thr_stats}
    res["n_gallery"] = int(G.shape[0])
    res["wall_s"] = round(time.time() - t0, 1)

    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(candidates, open(os.path.join(args.out, "candidates.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[prw] done -> {args.out}/ (metrics.json, candidates.json)")

    # quick baseline: two-signal voting at defaults (reid 0.55 + hue 35, agree 2)
    import scorer
    sc = scorer.score_all(candidates, {"reid_sim": 0.55, "hue_dhue": 35.0}, agree=2)
    hits = [c for c in sc if c["verdict"] == "hit"]
    tp = sum(1 for c in hits if c["gt"]); fp = len(hits) - tp
    pos = sum(1 for c in candidates if c["gt"]); fn = pos - tp
    print(f"[prw] two-signal defaults: {len(hits)} hits (tp={tp} fp={fp} fn={fn}); "
          f"vs threshold-only 0.55: tp={thr_stats['tp']} fp={thr_stats['fp']} fn={thr_stats['fn']}")


if __name__ == "__main__":
    main()
