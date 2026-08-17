#!/usr/bin/env python3
"""RSTPReid text-to-person benchmark — CLIP cross-modal retrieval.

Protocol: test split (1,000 images, 2 captions each). Gallery = CLIP image
embeddings of all test images; query = CLIP text embedding of each caption
(optionally refined through provider.refine_query first — the engine's NL→
phrase channel). Rank gallery per query; R@1/5/10 (hit = the caption's paired
image) + mAP (positives = all images of the same identity, standard
text-person-ReID mAP).

This validates the "语言→人" channel the semantic branch relies on (CLIP is
also the engine's always-available body-embedder fallback). SOTA reference
(LAIP 2024): R@1 0.62, mAP 0.47 — trained ReID models; ours is zero-shot, so
the honest comparison is against the zero-shot baseline, noted in the report.

Run (venv python; needs clip + weights):
  "$VENV" bench/run_rstpreid.py --out bench_out/rstpreid [--limit 50] [--use-refine]
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
import metrics as M    # noqa: E402


def clip_models(clip_backend):
    import matcher
    if clip_backend == "large":
        # transformers CLIPModel (ViT-L/14, cached under weights/clip_l_cache)
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14',
                                          cache_dir=os.path.join(
                                              os.path.dirname(_HERE), "..", "..", "weights",
                                              "clip_l_cache"))
        proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14',
                                             cache_dir=os.path.join(
                                                 os.path.dirname(_HERE), "..", "..", "weights",
                                                 "clip_l_cache"))
        model.eval()
        dev = "cpu"
        import torch
        return model, proc, dev, torch
    if not matcher.CLIP_OK:
        raise SystemExit("clip not importable in this env; install clip + weights")
    model, preprocess, dev = matcher._clip()
    import torch
    return model, preprocess, dev, torch


# prompt ensemble for the text side (zero cloud cost; CLIP is prompt-sensitive)
PROMPTS = [
    "a photo of a person: {q}",
    "{q}",
    "a surveillance photo of a person, {q}",
    "a person described as: {q}",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clip-backend", choices=["b32", "large"], default="b32")
    ap.add_argument("--use-refine", action="store_true",
                    help="refine captions via provider.refine_query first (needs a key)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = D.load_rstpreid(args.dataset, "test")
    if args.limit:
        items = items[:args.limit]
    print(f"[rstpreid] {len(items)} test items (clip={args.clip_backend})")

    model, prep, dev, torch = clip_models(args.clip_backend)
    import cv2
    import numpy as np
    import PIL.Image as Image

    # gallery: one CLIP image embedding per item
    G, gids = [], []
    t0 = time.time()
    for it in items:
        img = cv2.imread(it["img_path"])
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args.clip_backend == "large":
            t = prep(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                v = model.get_image_features(t)
            if hasattr(v, "pooler_output"):        # transformers 5.x returns pooling output
                v = v.pooler_output
        else:
            t = prep(Image.fromarray(rgb)).unsqueeze(0).to(dev)
            with torch.no_grad():
                v = model.encode_image(t)
        G.append(v.cpu().numpy().ravel().astype(np.float32))
        gids.append(it["id"])
    G = np.array(G)
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True).clip(min=1e-8)
    print(f"[rstpreid] gallery CLIP embedded: {G.shape} ({time.time()-t0:.0f}s)")

    # queries: each caption (2 per item), optionally refined
    import provider
    queries = []
    for it in items:
        for cap in it["captions"]:
            qtext = cap
            if args.use_refine:
                r = provider.refine_query(cap)
                if r:
                    qtext = r
            queries.append({"qtext": qtext, "gt_id": it["id"],
                            "gt_idx": len(gids) and gids.index(it["id"])})
    print(f"[rstpreid] {len(queries)} queries (refine={args.use_refine})")

    per_query = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        # prompt ensemble: best text prompt per query (max over prompts)
        best_sims = None
        for tmpl in PROMPTS:
            text = tmpl.format(q=q["qtext"])
            if args.clip_backend == "large":
                t = prep(text=[text], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    te = model.get_text_features(**t)
                if hasattr(te, "pooler_output"):   # transformers 5.x returns pooling output
                    te = te.pooler_output
            else:
                import clip
                te = model.encode_text(clip.tokenize([text]).to(dev))
            te = te / te.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                sims = (te @ Gn.T).cpu().numpy().ravel()
            if best_sims is None:
                best_sims = sims
            else:
                best_sims = np.maximum(best_sims, sims)
        order = np.argsort(-best_sims)
        is_pos = np.array([gids[i] == q["gt_id"] for i in order], dtype=bool)
        per_query.append((best_sims[order].tolist(), is_pos.tolist()))
        if qi % 500 == 0:
            print(f"  {qi}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)

    res = M.reid_metrics(per_query)
    res["use_refine"] = args.use_refine
    res["clip_backend"] = args.clip_backend
    res["prompt_ensemble"] = len(PROMPTS)
    res["wall_s"] = round(time.time() - t0, 1)
    res["n_gallery"] = int(G.shape[0])
    res["note"] = ("zero-shot CLIP " + ("ViT-L/14" if args.clip_backend == "large" else "ViT-B/32")
                   + "; SOTA(LAIP2024 trained) R@1=0.62 mAP=0.47")
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"[rstpreid] done -> {args.out}/")


if __name__ == "__main__":
    main()
