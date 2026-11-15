#!/usr/bin/env python3
"""Agent-layer wiring tests — toolbox contracts + case-log closed loop.

No engines are launched here (subprocess tools are only checked for envelope
shape against a missing-script stub); only the in-process judges (router /
rescoring) execute, and the case recording protocol is exercised end-to-end
with a fabricated confirmed case. Run: ``python3 test_agent.py``.
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

import toolbox as TB           # noqa: E402
import case_log as CL          # noqa: E402

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_manifest():
    print("[toolbox manifest]")
    names = [t["name"] for t in TB.TOOL_MANIFEST]
    for required in ("preflight", "locate", "verify_target", "person_search",
                     "fusion_run", "router", "ground", "rescoring"):
        check(f"tool present: {required}", required in names, names)
    for t in TB.TOOL_MANIFEST:
        for f in ("name", "kind", "impl", "desc", "inputs", "outputs", "cost", "fallback"):
            check(f"{t['name']} has field {f}", f in t, t)
        check(f"{t['name']} cost tags", "time" in t["cost"] and "cloud" in t["cost"],
              t["cost"])
    check("get_tool finds rescoring", TB.get_tool("rescoring")["impl"] == "inproc")
    check("get_tool unknown -> None", TB.get_tool("nope") is None)
    check("check_requirements unknown -> missing",
          TB.check_requirements("nope")["ok"] is False)
    # router/ground requirements check runs without heavy imports
    check("router has no requirements", TB.check_requirements("router")["ok"] is True)
    ctx = TB.agent_context()
    check("agent_context lists tools", "fusion_run" in ctx and "成本" in ctx, len(ctx))


def test_dispatch_inproc():
    print("[toolbox dispatch (inproc)]")
    r = TB.dispatch("router", {"query": "车牌号 京Q1G728 的小轿车"})
    # 产品决策（W1 预算红线）：车牌/证件号已改判 unsupported，不再走 precise_text
    check("router dispatch ok",
          r["status"] == "ok" and r["summary"]["branch"] == "unsupported"
          and r["summary"].get("kind") == "plate", r)
    tb_names = set(TB.list_tools()) if hasattr(TB, "list_tools") else set()
    if tb_names:
        check("fast_plate_scan 已从工具表移除", "fast_plate_scan" not in tb_names, sorted(tb_names))
    check("router target extracted", r["summary"]["target"] == "Q1G728", r)

    cands = [
        {"id": "t438", "t_s": 438.0, "results": {
            "hue": {"score": 0.9, "raw": {"dhue": 12.0, "colorful": True}, "evidence": ""},
            "temporal": {"score": 0.7, "raw": {"peak": 0.83, "bellness": 0.5, "span_s": 2.0, "n": 5}, "evidence": ""},
            "vlm": {"score": 0.0, "raw": {"match": "unavailable"}, "evidence": ""}}},
        {"id": "f120", "t_s": 120.0, "results": {
            "hue": {"score": 0.1, "raw": {"dhue": 95.0, "colorful": True}, "evidence": ""},
            "temporal": {"score": 0.4, "raw": {"peak": 0.62, "bellness": 0.1, "span_s": 0.0, "n": 3}, "evidence": ""},
            "vlm": {"score": 0.0, "raw": {"match": "no"}, "evidence": ""}}},
    ]
    r = TB.dispatch("rescoring", {"candidates": cands, "agree": 2})
    check("rescoring ok", r["status"] == "ok", r)
    check("1 hit (vlm abstains, two fire)",
          r["summary"]["verdicts"].get("hit", 0) == 1, r["summary"])
    check("hit is t438", r["summary"]["hits"][0]["id"] == "t438", r["summary"])

    r = TB.dispatch("rescoring", {"candidates": [], "agree": 2})
    check("rescoring empty -> failed envelope (not crash)", r["status"] == "failed", r)
    # an LLM can hand prose as thresholds — must degrade, never raise
    cands_bad = cands[:1]
    r = TB.dispatch("rescoring", {"candidates": cands_bad, "thresholds": "放宽一点"})
    check("string thresholds coerced (no crash)", r["status"] == "ok", r)
    r = TB.dispatch("rescoring", {"candidates": cands_bad, "agree": "不是数字"})
    check("string agree -> default bar (no crash)", r["status"] == "ok"
          and r["summary"]["verdicts"].get("hit", 0) == 1, r)
    r = TB.dispatch("nope", {})
    check("unknown tool -> failed envelope", r["status"] == "failed", r)


def test_dispatch_missing_script():
    print("[toolbox dispatch (missing script) -> skipped, no run]")
    old = TB.SKILL_SCRIPTS
    TB.SKILL_SCRIPTS = "/nonexistent/scripts"
    r = TB.dispatch("locate", {"video": "x.mp4", "query": "a car", "out": "/tmp/x"})
    TB.SKILL_SCRIPTS = old
    check("missing script -> skipped envelope", r["status"] == "skipped" and "script" in r["error"], r)


def test_build_argv():
    print("[toolbox argv mapping (regression: positional video)]")
    # preflight takes VIDEO as a positional — no empty-string flags allowed.
    argv = TB._build_argv("preflight", {"video": "v.mp4", "sample": "s.jpg"})
    check("preflight: script then positional video",
          argv[1].endswith("preflight.py") and argv[2] == "v.mp4"
          and "" not in argv, argv)
    check("preflight: --sample flag", "--sample" in argv and "s.jpg" in argv, argv)
    argv = TB._build_argv("preflight",
                          {"video": "v.mp4", "sample": "s.jpg", "recover": "r.mp4"})
    check("preflight: --recover flag", "--recover" in argv and "r.mp4" in argv, argv)
    argv = TB._build_argv("fusion_run", {"video": "v.mp4", "query": "白车",
                                         "out": "/tmp/o", "agree": 2})
    check("fusion_run: flags mapped", "--video" in argv and "--query" in argv
          and "--agree" in argv and "--out" in argv, argv)
    model = argv[argv.index("--model") + 1] if "--model" in argv else None
    check("fusion_run: model absolutized (no cwd re-download)",
          model and os.path.isabs(model) and os.path.exists(model), model)
    argv = TB._build_argv("person_search", {"video": "v.mp4", "ref": "r.png",
                                            "out": "/tmp/o", "reid_thresh": 0.75})
    check("person_search: --reid-thresh mapped",
          "--reid-thresh" in argv and "0.75" in argv, argv)
    argv = TB._build_argv("fast_plate_scan", {"video": "v.mp4", "target": "Q1G728",
                                              "out": "/tmp/o"})
    check("fast_plate_scan: --everywhere when no manifest",
          "--everywhere" in argv and "--target" in argv, argv)


def test_dispatch_streaming():
    print("[dispatch on_line: live engine stdout + timeout]")
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix="srv-stream-")
    with open(os.path.join(tmp, "locate.py"), "w") as f:
        f.write("import time\n"
                "print('[stage1] scanned 5/100 frames', flush=True)\n"
                "print('mid line', flush=True)\n"
                "time.sleep(0.2)\n"
                "print('[done] outputs in out/', flush=True)\n")
    old = TB.SKILL_SCRIPTS
    TB.SKILL_SCRIPTS = tmp
    try:
        lines = []
        r = TB.dispatch("locate", {"video": "v.mp4", "query": "q", "out": tmp},
                        on_line=lines.append)
        check("streamed run ok", r["status"] == "ok", r)
        check("lines streamed in order",
              lines and "[stage1]" in lines[0] and "[done]" in lines[-1]
              and lines.index("[stage1] scanned 5/100 frames")
                  < lines.index("[done] outputs in out/"), lines)
        # timeout must still hold while streaming
        with open(os.path.join(tmp, "locate.py"), "w") as f:
            f.write("import time\nprint('start', flush=True)\ntime.sleep(30)\n")
        r2 = TB.dispatch("locate", {"video": "v.mp4", "query": "q", "out": tmp,
                                    "timeout": 1}, on_line=lambda ln: None)
        check("streamed timeout -> failed envelope", r2["status"] == "failed"
              and "timeout" in r2["error"], r2)
    finally:
        TB.SKILL_SCRIPTS = old


def test_case_log():
    print("[case log protocol]")
    bad = {"id": "x"}
    check("validate catches missing fields", len(CL.validate(bad)) > 0, CL.validate(bad))
    check("validate catches bad query_type",
          "query_type must be" in "".join(CL.validate(
              {"id": "x", "ts": "t", "query": "q", "query_type": "wat",
               "inputs": {}, "plan": [], "attempts": [], "outcome": "resolved"})).lower()
          or any("query_type" in p for p in CL.validate(
              {"id": "x", "ts": "t", "query": "q", "query_type": "wat",
               "inputs": {}, "plan": [], "attempts": [], "outcome": "resolved"})))

    with tempfile.TemporaryDirectory() as tmp:
        # 1. fabricate a confirmed composite case with a rescoring envelope
        path = os.path.join(tmp, "cases.jsonl")
        case = {
            "id": "case_test_001", "ts": "2026-08-13T00:00:00",
            "query": "找 3 点蓝门附近穿红衣的人", "query_type": "composite",
            "inputs": {"video": "/tmp/clip.mp4", "ref": None, "provider_status": {"glm": False}},
            "plan": ["定位蓝门", "时间窗", "窗内红衣", "核验"],
            "attempts": [{"tool": "rescoring", "params": {"agree": 2}, "note": "双信号一致",
                          "envelope": {"outputs": {"scored": [
                              {"id": "f438", "t_s": 438.0, "results": {
                                  "hue": {"score": 0.9, "raw": {"dhue": 12.0}, "evidence": ""},
                                  "temporal": {"score": 0.7, "raw": {"peak": 0.83, "bellness": 0.5, "span_s": 2.0, "n": 5}, "evidence": ""}}},
                              {"id": "f120", "t_s": 120.0, "results": {
                                  "hue": {"score": 0.1, "raw": {"dhue": 95.0}, "evidence": ""},
                                  "temporal": {"score": 0.4, "raw": {"peak": 0.62, "bellness": 0.1, "span_s": 0.0, "n": 3}, "evidence": ""}}}]}}}],
            "outcome": "resolved", "agent_reasoning": "拆步锁定 t=438",
            "human_confirmed": True, "gt": {"hit_intervals": [[437.0, 439.0]]},
        }
        # 2. an unconfirmed case must NOT enter the dataset
        unconfirmed = dict(case, id="case_test_002", human_confirmed=None, gt=None)
        ok, _ = CL.record_case(unconfirmed, path)
        ok2, _ = CL.record_case(case, path)
        check("record_case accepts valid", ok and ok2)
        check("iter_cases reads 2", len(list(CL.iter_cases(path))) == 2)
        check("confirmed filters to 1", len(CL.confirmed_cases(path)) == 1)

        # 3. closed loop: case -> dataset contract
        ds = CL.cases_to_dataset(path)
        check("dataset candidates extracted", len(ds["candidates"]) == 2, ds)
        check("gt from hit_intervals",
              sum(c["gt"] for c in ds["candidates"]) == 1 and
              next(c for c in ds["candidates"] if c["gt"])["t_s"] == 438.0)
        check("rows empty (no OCR attempt)", ds["rows"] == [])

        # 4. OCR extraction path
        all_ocr = os.path.join(tmp, "all_ocr.json")
        json.dump([{"ocr": "11G728", "t_s": 838.5, "frame": 20964},
                   {"ocr": "AN17119", "t_s": 120.0, "frame": 3000}], open(all_ocr, "w"))
        plate_case = {
            "id": "case_test_003", "ts": "2026-08-13T00:00:00",
            "query": "新能源绿牌 京A12345", "query_type": "novel_charset",
            "inputs": {"video": "/tmp/clip.mp4", "ref": None, "provider_status": {}},
            "plan": ["OCR 全片", "绿牌色块", "补规则"],
            "attempts": [{"tool": "fast_plate_scan", "params": {}, "note": "补了绿牌规则",
                          "envelope": {"outputs": {"all_ocr.json": all_ocr}}}],
            "outcome": "resolved", "agent_reasoning": "", "human_confirmed": True,
            "gt": {"positive_reads": ["11G728"]},
        }
        CL.record_case(plate_case, path)
        ds = CL.cases_to_dataset(path)
        rows = ds["rows"]
        check("ocr rows extracted", len(rows) == 2, rows)
        check("rows labeled by positive_reads",
              sum(r["gt"] for r in rows) == 1 and
              next(r for r in rows if r["gt"])["ocr"] == "11G728", rows)


if __name__ == "__main__":
    test_manifest()
    test_dispatch_inproc()
    test_dispatch_missing_script()
    test_build_argv()
    test_dispatch_streaming()
    test_case_log()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
