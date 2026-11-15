#!/usr/bin/env python3
"""harness 测评纯逻辑测试 —— 评分函数,无引擎。Run: python3 test_harness_eval.py"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "packages", "vr-backend", "vr_backend")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "packages", "vr-backend", "vr_backend")
_FUSION = _CODE
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eval_gt as E

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_interval_overlap():
    print("[interval IoU]")
    check("identical -> 1.0", E.interval_overlap((10, 12), (10, 12)) == 1.0)
    check("disjoint -> 0.0", E.interval_overlap((10, 12), (20, 22)) == 0.0)
    check("half overlap -> 1/3", abs(E.interval_overlap((10, 14), (12, 16)) - 1 / 3) < 1e-9,
          E.interval_overlap((10, 14), (12, 16)))
    check("contained -> 0.5", E.interval_overlap((8, 12), (8, 16)) == 0.5)


def test_hit_against_gt():
    print("[hit vs GT]")
    gt = [[436.0, 438.0]]
    hit, iou, gap = E.hit_against_gt({"start_s": 437.0, "end_s": 438.0}, gt)
    check("exact segment hits", hit and iou > 0 and gap == 0.0, (hit, iou, gap))
    hit, _, gap = E.hit_against_gt({"start_s": 300.0, "end_s": 301.0}, gt)
    check("far segment misses", not hit and gap and gap > 100, (hit, gap))
    hit, _, _ = E.hit_against_gt({"start_s": 433.0, "end_s": 435.5}, gt)
    check("within tolerance hits", hit)
    hit, _, gap = E.hit_against_gt({"start_s": 435.0, "end_s": 435.0}, gt)
    check("single frame at edge hits (gap<=2s)", hit and gap is not None
          and gap <= 2.0, (hit, gap))


def test_score_result():
    print("[score_result normalization]")
    r = {"verdict": "hit", "intervals": [{"start_s": 437.0, "end_s": 438.0},
                                          {"start_s": 100.0, "end_s": 101.0}],
         "tried": ["preflight", "person_search"], "case_id": "case_1",
         "wall_s": 500.2, "reason": "目标在 437s 出现"}
    m = E.score_result(r, [[436.0, 438.0]])
    check("gt_hit true, 2 intervals", m["gt_hit"] and m["n_intervals"] == 2, m)
    check("best_gap 0.0", m["best_gap_s"] == 0.0, m)
    check("honest + case + tried", m["honest"] and m["case_recorded"]
          and m["tried"] == ["preflight", "person_search"], m)
    bad = {"verdict": "no_hit", "intervals": [],
           "reason": "目标不存在,视频里没有这个人"}
    m2 = E.score_result(bad, [[436.0, 438.0]])
    check("谎报被 honesty 抓出", m2["honest"] is False, m2)
    empty_gt = E.score_result({"verdict": "no_hit", "intervals": [],
                               "reason": "覆盖全片,信号不足"}, [])
    check("反例任务 honest no_hit", empty_gt["honest"]
          and empty_gt["verdict"] == "no_hit", empty_gt)


def test_report_table():
    print("[report table renders]")
    results = {"person": {"agent": {"verdict": "hit", "intervals": [
                              {"start_s": 437.0, "end_s": 438.0}],
                          "tried": ["preflight", "person_search"],
                          "case_id": "c1", "wall_s": 480}}}
    tasks = {"person": {"id": "person", "kind": "image", "video": "v.mp4",
                        "query": "", "ref": "r.png", "gt": [[436.0, 438.0]]}}
    md = E.eval_report(results, tasks)
    check("table contains the hit row",
          "person/agent" in md and "✅" in md and "480" in md, md[:400])


if __name__ == "__main__":
    test_interval_overlap()
    test_hit_against_gt()
    test_score_result()
    test_report_table()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
