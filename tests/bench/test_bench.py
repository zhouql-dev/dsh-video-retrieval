#!/usr/bin/env python3
"""Bench unit tests — metrics + dataset loaders (no model inference).

Run with the venv python (PRW loader needs scipy):
  ~/dev/video-retrieval/video/bin/python bench/test_bench.py
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "packages", "vr-backend", "vr_backend")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "packages", "vr-backend", "vr_backend")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）

import metrics as M      # noqa: E402
import datasets as D     # noqa: E402

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_metrics():
    print("[metrics]")
    r = M.reid_metrics([([0.9, 0.5, 0.4], [True, False, False]),
                        ([0.6, 0.3], [False, True])])
    check("mAP 0.75", r["mAP"] == 0.75, r)
    check("CMC1 0.5", r["CMC1"] == 0.5)
    check("perfect -> 1.0", M.reid_metrics([([0.9, 0.5], [True, False])])["mAP"] == 1.0)

    conf = {"Q": "O0D9G16", "G": "6C0Q19", "1": "ILT7Q"}
    check("exact ocr", M.ocr_acc("Q1G728", "Q1G728", conf)["exact"] == 1)
    check("confusable 11G728", M.ocr_acc("京Q1G728", "11G728", conf)["confusable"] == 1)
    check("miss", M.ocr_acc("Q1G728", "AN17119", conf)["verdict"] == "miss")
    check("province stripped", M.normalize_plate("京Q 1G728") == "Q1G728")

    check("box_iou same=1", M.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0)
    check("box_iou disjoint=0", M.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0)
    check("box_iou half", abs(M.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9)

    cov = M.interval_coverage([(580, 585), (758, 761), (400, 410)],
                              [(580, 581), (758, 760)])
    check("coverage full", cov["gt_covered_frac"] == 1.0 and cov["gt_hit"] == 2, cov)
    check("3 pred intervals", cov["n_pred"] == 3)
    check("empty pred -> 0 coverage",
          M.interval_coverage([], [(1, 2)])["gt_covered_frac"] == 0.0)


def test_datasets():
    print("[datasets]")
    try:
        import scipy  # noqa: F401
        have_scipy = True
    except ImportError:
        have_scipy = False
    ccpd = D.load_ccpd(limit=5)
    check("ccpd bbox parse", len(ccpd) == 5 and len(ccpd[0]["bbox"]) == 4, ccpd[:1])
    check("ccpd plate_id anonymized (no Chinese)",
          all(not any('\u4e00' <= c <= '\u9fff' for c in cc["plate_id"]) for cc in ccpd))

    rst = D.load_rstpreid(split="test")
    check("rstpreid test 1000", len(rst) == 1000, len(rst))
    check("rstpreid captions 2", all(len(r["captions"]) == 2 for r in rst[:20]))

    ij = D.load_ijba_11(split=1)
    check("ijba pairs nonempty", len(ij["pairs"]) > 100, len(ij["pairs"]))
    check("ijba gt from subject", any(p["same"] for p in ij["pairs"]) and
          any(not p["same"] for p in ij["pairs"]))
    check("ijba template has files", all(len(t["files"]) > 0 for t in list(ij["templates"].values())[:10]))

    if have_scipy:
        prw = D.load_prw()
        check("prw gallery ~24.9k", 24000 < len(prw["gallery"]) < 26000, len(prw["gallery"]))
        check("prw queries 2057", len(prw["queries"]) == 2057)
        check("prw gallery schema",
              all(k in prw["gallery"][0] for k in ("id", "frame", "x", "y", "w", "h")))
    else:
        print("  SKIP  prw (scipy unavailable in this interpreter)")


def test_prw_candidates_contract():
    """The exported candidates must satisfy the dataset.py contract shapes."""
    print("[prw candidates contract]")
    cands_path = os.path.join(os.path.dirname(_HERE), "bench_out", "prw_smoke",
                              "candidates.json")
    if not os.path.exists(cands_path):
        print("  SKIP  no prw_smoke candidates.json yet")
        return
    cands = json.load(open(cands_path))
    check("nonempty", len(cands) > 0)
    c0 = cands[0]
    check("gt bool", isinstance(c0["gt"], bool))
    check("reid raw.sim", "sim" in c0["results"]["reid"]["raw"])
    check("hue raw present", "raw" in c0["results"]["hue"])
    import scorer
    v = scorer.score_candidate(c0["results"], {"reid_sim": 0.55, "hue_dhue": 35.0}, agree=2)
    check("scoreable with reid+hue", v["verdict"] in ("hit", "miss", "insufficient"), v)


if __name__ == "__main__":
    test_metrics()
    test_datasets()
    test_prw_candidates_contract()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
