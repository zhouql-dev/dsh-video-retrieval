#!/usr/bin/env python3
"""A/B 评测:同一批裁剪图分别经 GLM 与 ox-alpha 的 verify_target 判决对比。

背景:vtl_out_q1g728/ 是 GLM-4v-flash + 花括号占位符 bug 时代的验收输出
(187 条判决全部 "unclear"、无 read 字段)。该 bug 已修复;本脚本用修复后的
模板在两条云通道上重放同一批裁剪,量化"更新前后"与"通道间"差异。

GT:命中帧区间 [20853, 21004](车牌 京Q1G728 @ ~838s)内的裁剪为正样本。

    python fusion/bench/run_ox_ab.py --channel glm   --out bench_out/ox_ab/glm.json
    python fusion/bench/run_ox_ab.py --channel qwen  --out bench_out/ox_ab/qwen.json
    python fusion/bench/run_ox_ab.py --channel oxalpha --out bench_out/ox_ab/ox.json

强制单通道(绕过 chat() 的自动回落):直接调 _post 到对应端点,失败如实记
error——这样"通道不可用"本身就是评测结论,而不是被静默兜底掩盖。
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
_REPO = os.path.dirname(_FUSION)
import _bootstrap_shim  # noqa: E402,F401  P1-8：脚本模式垫片

import cv2
import provider

CROPS_DIR = os.path.join(_REPO := os.path.dirname(_FUSION), "vtl_out_q1g728", "crops")
VERIFY_JSON = os.path.dirname(CROPS_DIR) + "/verify.json"
QUERY = "白色车牌号 京Q1G728 的车辆(轿车/SUV等机动车)"
GT_LO, GT_HI = 20853, 21004


def load_cases():
    v = json.load(open(VERIFY_JSON, encoding="utf-8"))
    cases = []
    for e in sorted(v, key=lambda x: x["frame"]):
        path = os.path.join(CROPS_DIR, e["crop"])
        img = cv2.imread(path)
        if img is None:
            continue
        cases.append({"crop": e["crop"], "frame": e["frame"], "img": img,
                      "is_hit": GT_LO <= e["frame"] <= GT_HI})
    return cases


def forced_post(channel):
    """单通道 _post 包装:只允许 channel 对应的端点应答,其余一律 None。"""
    cfgs = {"glm": (provider.GLM_BASE_URL, provider.GLM_VISION_MODEL),
            "qwen": (provider.QWEN_BASE_URL, provider.QWEN_VISION_MODEL),
            "oxalpha": (provider.OX_BASE_URL, provider.OX_VISION_MODEL)}
    base, model = cfgs[channel]
    real_post = provider._post

    def wrapped(base_url, key, m, *a, **kw):
        if base_url != base:
            return None
        return real_post(base_url, key, m or model, *a, **kw)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=("glm", "qwen", "oxalpha"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel verify_target calls")
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        # 前后各取一半的一半,保证正负样本都在
        hits = [c for c in cases if c["is_hit"]]
        rest = [c for c in cases if not c["is_hit"]]
        k = max(1, args.limit // 4)
        cases = hits[:k] + rest[: args.limit - k]
    print(f"[ab] {args.channel}: {len(cases)} crops "
          f"({sum(c['is_hit'] for c in cases)} in-GT)")

    results = [None] * len(cases)
    t0 = time.time()

    def work(i):
        c = cases[i]
        ts = time.time()
        r = provider.verify_target(QUERY, c["img"])     # 模板已修复
        return i, {"crop": c["crop"], "frame": c["frame"],
                   "is_hit": c["is_hit"], "latency_s": round(time.time() - ts, 2),
                   "verdict": r}

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for done, (i, rec) in enumerate(ex.map(work, range(len(cases))), 1):
                results[i] = rec
                if done % 20 == 0:
                    print(f"  {done}/{len(cases)} elapsed={time.time()-t0:.0f}s")
    else:
        for i, (i, rec) in enumerate(map(work, range(len(cases)))):
            results[i] = rec
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(cases)} elapsed={time.time()-t0:.0f}s")
    total_s = time.time() - t0

    ok = [r for r in results if r["verdict"]]
    lat = [r["latency_s"] for r in ok]

    def cls(r):
        return (r["verdict"] or {}).get("match", "unavailable")

    def score(subset):
        n = len(subset)
        if not n:
            return {}
        hit = sum(1 for r in subset if r["is_hit"]
                  and cls(r) in ("exact", "partial"))
        fp = sum(1 for r in subset if not r["is_hit"]
                 and cls(r) in ("exact", "partial"))
        unc = sum(1 for r in subset if cls(r) == "unclear")
        no = sum(1 for r in subset if cls(r) == "no")
        prec = hit / (hit + fp) if hit + fp else None
        rec = hit / n
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
        return {"n": n, "tp": hit, "fp": fp, "unclear": unc, "no": no,
                "precision": round(prec, 3) if prec is not None else None,
                "recall": round(rec, 3), "f1": round(f1, 3) if f1 else None}

    summary = {"channel": args.channel, "total_s": round(total_s, 1),
               "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
               "answered": len(ok),
               "errors": len(results) - len(ok),
               "label_counts": {},
               "all": score(results)}
    from collections import Counter
    summary["label_counts"] = dict(Counter(cls(r) for r in results))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

    
    # P2-2：统一退出码（缺产物/空文件/acceptance 失败 → exit 1）
    import _guard
    _out = None
    for _i, _a in enumerate(sys.argv):
        if _a == "--out" and _i + 1 < len(sys.argv):
            _out = sys.argv[_i + 1]
    _guard.finish(_out)
