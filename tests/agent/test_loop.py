#!/usr/bin/env python3
"""E1 agent-loop tests — state machine, guardrails, degradation, case log.

No engines launch here: dispatch is mocked with a scripted fake, the LLM is a
fake function (or absent to force the deterministic playbook), and the glue
runner executes a trivial stdlib script. Verifies:

  * deterministic playbook (no key): preflight -> router -> engine, honest
    hit/no-hit verdicts, "尝试/覆盖/原因/建议" evidence, NO "不存在" claim;
  * insufficient verdict triggers rescoring -> permissive -> engine swap,
    ends with a recorded hard case (case_id + JSONL persistence);
  * environment failure (bad video) reports honestly;
  * LLM loop: tool-call -> envelope feedback -> final; max_steps stops;
  * text ReAct line parser (tool-calling fallback contract);
  * glue subprocess: whitelisted imports run, blacklisted rejected;
  * mode="deterministic" never consults the LLM even when available.

Run: ``python3 test_loop.py`` (exit code 0 = all pass).
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

import loop as L
import case_log as CL

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def _ok_env(tool, n_hits=1, intervals=None, verdict="OK", extra=None):
    return {"tool": tool, "status": "ok", "summary": {
        "verdict": verdict,
        "metrics.json": {"n_hits": n_hits, "n_intervals": len(intervals or []),
                         "intervals": intervals or []},
        **(extra or {})},
        "outputs": {"scored.json": "/tmp/fake/scored.json"},
        "error": None, "cost": {}}


def _insufficient_env(tool):
    return {"tool": tool, "status": "ok", "summary": {
        "verdict": "insufficient",
        "metrics.json": {"n_hits": 0, "n_intervals": 0, "intervals": []},
        "verdicts": {"hit": 0, "miss": 3, "insufficient": 1}},
        "outputs": {"scored.json": "/tmp/fake/scored.json"},
        "error": None, "cost": {}}


def _failed_env(tool, error):
    return {"tool": tool, "status": "failed", "summary": {},
            "outputs": {}, "error": error, "cost": {}}


def _fake_dispatch(script):
    """script: list of (tool, params) -> envelope. Repeats the last entry."""
    calls = []

    def fn(tool, params, workdir=None):
        calls.append((tool, params))
        for t, env in script:
            if t == tool:
                return env
        return script[-1][1]
    fn.calls = calls
    return fn


class FakeLLM:
    """Programmable LLM: each call pops one reply; last reply repeats."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append(messages)
        if self.replies:
            return self.replies.pop(0)
        return self.replies[-1]


def _tmp_cases():
    return os.path.join(tempfile.mkdtemp(prefix="loop-cases-"), "cases.jsonl")


def test_playbook_hit_and_events():
    print("[deterministic playbook: hit path]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None, "query": "白车"},
                    "outputs": {}, "error": None, "cost": {}}),
        ("fusion_run", _ok_env("fusion_run", n_hits=2,
                               intervals=[{"start_s": 10.0, "end_s": 12.0}]))])
    events = []
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd,
                on_event=events.append, cases_path=_tmp_cases())
    res = a.run("白车", "/tmp/a.mp4", mode="deterministic")
    check("verdict hit", res["verdict"] == "hit", res)
    check("intervals extracted", res["intervals"] == [{"start_s": 10.0, "end_s": 12.0}],
          res["intervals"])
    check("evidence populated", res["evidence"].get("n_hits") == 2, res["evidence"])
    check("routine case recorded on hit (confirm/deny feed)",
          bool(res["case_id"]), res["case_id"])
    check("events emitted", any(e["type"] == "start" for e in events)
          and any(e.get("tool") == "fusion_run" for e in events), events[:5])
    check("preflight ran first", fd.calls[0][0] == "preflight", fd.calls)
    check("no 不存在 anywhere", "不存在" not in json.dumps(res, ensure_ascii=False))


def test_insufficient_records_case_and_is_honest():
    print("[insufficient -> rescoring -> engine swap -> recorded case]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None, "query": "推婴儿车的男子"},
                    "outputs": {}, "error": None, "cost": {}}),
        ("fusion_run", _insufficient_env("fusion_run")),
        ("rescoring", _insufficient_env("rescoring")),
        ("verify_target", _insufficient_env("verify_target"))])
    cases_path = _tmp_cases()
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, cases_path=cases_path)
    res = a.run("推婴儿车的男子", "/tmp/a.mp4", mode="deterministic")
    check("verdict no_hit (honest)", res["verdict"] == "no_hit", res["verdict"])
    check("case recorded", res["case_id"] and res["case_id"].startswith("case_"),
          res["case_id"])
    check("case persisted to JSONL",
          os.path.exists(cases_path) and sum(1 for _ in CL.iter_cases(cases_path)) == 1)
    check("attempts logged",
          any(c["tool"] == "rescoring" for c in CL.iter_cases(cases_path).__iter__().__next__()["attempts"]))
    check("coverage/reason/suggestion present",
          all(k in res for k in ("tried", "coverage", "reason", "suggestion")), res)
    check("never claims 不存在", "不存在" not in json.dumps(res, ensure_ascii=False))
    check("rescoring tried before swap",
          [t for t, _ in fd.calls].index("rescoring")
          < [t for t, _ in fd.calls].index("verify_target"), fd.calls)


def test_environment_failure_honest():
    print("[unreadable video -> honest environment report]")
    fd = _fake_dispatch([
        ("preflight", {"tool": "preflight", "status": "ok", "summary": {
            "verdict": "UNREADABLE", "metrics.json": {}},
            "outputs": {}, "error": None, "cost": {}})])
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, cases_path=_tmp_cases())
    res = a.run("白车", "/tmp/bad.mp4", mode="deterministic")
    check("verdict inconclusive", res["verdict"] == "inconclusive", res)
    check("reason mentions video", "视频" in (res.get("reason") or "")
          or "UNREADABLE" in (res.get("reason") or ""), res.get("reason"))
    check("case recorded for environment trouble", bool(res["case_id"]), res["case_id"])
    check("no fake hit", res["intervals"] == [], res["intervals"])


def test_llm_loop_and_max_steps():
    print("[LLM loop: tool call -> envelope -> final; max_steps stops]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None, "query": "白车"},
                    "outputs": {}, "error": None, "cost": {}}),
        ("fusion_run", _ok_env("fusion_run", n_hits=1,
                               intervals=[{"start_s": 5.0, "end_s": 6.0}]))])
    llm = FakeLLM([
        {"tool_calls": [{"id": "c1", "name": "fusion_run",
                         "arguments": json.dumps({"video": "/tmp/a.mp4",
                                                  "query": "白车",
                                                  "out": "/tmp/out1"})}]},
        {"text": json.dumps({"verdict": "hit", "note": "三信号一致"})},
    ])
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, llm_fn=llm,
                cases_path=_tmp_cases())
    res = a.run("白车", "/tmp/a.mp4")
    check("llm-driven hit", res["verdict"] == "hit", res)
    check("hit intervals from envelope",
          res["intervals"] == [{"start_s": 5.0, "end_s": 6.0}], res["intervals"])
    check("llm consulted twice", len(llm.calls) == 2, len(llm.calls))
    check("system prompt built",
          any("dispatch" in str(m.get("content", "")) for m in llm.calls[0]
              if m["role"] == "system"), llm.calls[0][0]["content"][:200])

    # endless tool-calling LLM -> max_steps halts honestly
    llm2 = FakeLLM([{"tool_calls": [{"id": f"c{i}", "name": "router",
                                     "arguments": json.dumps({"query": "白车"})}]}
                    for i in range(100)])
    a2 = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, llm_fn=llm2,
                 max_steps=3, cases_path=_tmp_cases())
    res2 = a2.run("白车", "/tmp/a.mp4")
    check("max_steps enforced", len(a2._steps) <= 3, len(a2._steps))
    check("step overflow honest", res2["verdict"] in ("no_hit", "inconclusive")
          and "不存在" not in json.dumps(res2, ensure_ascii=False), res2)


def test_react_parser():
    print("[text ReAct line parser]")
    d1 = L.parse_react_line('dispatch(fusion_run, {"video": "/tmp/a.mp4", "query": "白车"})')
    check("dispatch parsed", d1 == ("dispatch", "fusion_run",
                                    {"video": "/tmp/a.mp4", "query": "白车"}), d1)
    d2 = L.parse_react_line('final({"verdict": "hit"})')
    check("final parsed", d2 == ("final", None, {"verdict": "hit"}), d2)
    check("garbage -> None", L.parse_react_line("随便说点什么") is None)
    check("invalid json -> None", L.parse_react_line("dispatch(router, {bad})") is None)


def test_glue_runner():
    print("[controlled glue subprocess]")
    wd = tempfile.mkdtemp(prefix="glue-")
    ok = L.run_glue("import json\nprint(json.dumps({'sum': 1+2}))", wd, timeout=30)
    check("whitelisted script runs", ok["status"] == "ok"
          and '"sum": 3' in ok.get("stdout", ""), ok)
    bad = L.run_glue("import requests\nprint(requests.get('http://x'))", wd, timeout=30)
    check("non-whitelisted import rejected", bad["status"] == "rejected", bad)
    slow = L.run_glue("import time\ntime.sleep(5)", wd, timeout=1)
    check("timeout enforced", slow["status"] == "timeout", slow)


def test_ref_only_search():
    print("[ref-only search: 以图搜人 without text query]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None},
                    "outputs": {}, "error": None, "cost": {}}),
        ("person_search", _ok_env("person_search", n_hits=1,
                                  intervals=[{"start_s": 42.0, "end_s": 44.0}]))])
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd,
                cases_path=_tmp_cases())
    res = a.run("", "/tmp/a.mp4", ref="/tmp/ref.png", mode="deterministic")
    check("ref-only runs and hits", res["verdict"] == "hit"
          and res["intervals"] == [{"start_s": 42.0, "end_s": 44.0}], res)
    check("router skipped on empty query",
          "router" not in [t for t, _ in fd.calls], fd.calls)
    check("default query is 以图搜人",
          res["log"][0].get("query", "").startswith("参考图中的人物"),
          res["log"][0])
    check("ref-only hit records a case", bool(res["case_id"]), res["case_id"])
    llm = FakeLLM([{"tool_calls": [{"id": "c1", "name": "person_search",
                                    "arguments": json.dumps({
                                        "video": "/tmp/a.mp4",
                                        "out": "/tmp/o"})}]},
                   {"text": json.dumps({"verdict": "hit"})}])
    fd2 = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("person_search", _ok_env("person_search", n_hits=1,
                                  intervals=[{"start_s": 1.0, "end_s": 2.0}]))])
    a2 = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd2, llm_fn=llm,
                 cases_path=_tmp_cases())
    res2 = a2.run("", "/tmp/a.mp4", ref="/tmp/ref.png")
    check("ref-only LLM loop works", res2["verdict"] == "hit", res2)
    check("step events carry duration_s",
          all("duration_s" in e for e in res2["log"] if e["type"] == "step"),
          [e for e in res2["log"] if e["type"] == "step"][:1])
    # the model's own out/video must NOT leak through — the task owns them
    ps_params = [p for t, p in fd2.calls if t == "person_search"][0]
    check("model out redirected into workdir",
          ps_params["out"].startswith(a2.workdir), ps_params)
    check("video forced to task video", ps_params["video"] == "/tmp/a.mp4",
          ps_params)
    # W-reid 改版：无 VLM key 时注入 0.65（并接 WeVisOne 边缘复核）
    check("reid_thresh 0.65 injected without VLM key",
          ps_params.get("reid_thresh") == 0.65, ps_params)


def test_final_as_tool_call():
    print("[model emits final as a pseudo tool call]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None, "query": "白车"},
                    "outputs": {}, "error": None, "cost": {}}),
        ("fusion_run", _ok_env("fusion_run", n_hits=1,
                               intervals=[{"start_s": 3.0, "end_s": 4.0}]))])
    llm = FakeLLM([
        {"tool_calls": [{"id": "c1", "name": "fusion_run",
                         "arguments": json.dumps({"video": "/tmp/a.mp4",
                                                  "query": "白车",
                                                  "out": "/tmp/o"})}]},
        {"tool_calls": [{"id": "c2", "name": "final",
                         "arguments": json.dumps({"verdict": "hit",
                                                  "note": "三信号一致"})}]},
    ])
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, llm_fn=llm,
                cases_path=_tmp_cases())
    res = a.run("白车", "/tmp/a.mp4")
    check("final pseudo-call resolves to hit", res["verdict"] == "hit"
          and res["intervals"] == [{"start_s": 3.0, "end_s": 4.0}], res)
    check("final never dispatched as a tool",
          all(t != "final" for t, _ in fd.calls), fd.calls)


def test_mode_deterministic_ignores_llm():
    print("[mode=deterministic never consults the LLM]")
    fd = _fake_dispatch([
        ("preflight", _ok_env("preflight")),
        ("router", {"tool": "router", "status": "ok",
                    "summary": {"branch": "semantic", "target": None, "query": "白车"},
                    "outputs": {}, "error": None, "cost": {}}),
        ("fusion_run", _ok_env("fusion_run", n_hits=0, intervals=[]))])
    llm = FakeLLM([{"text": json.dumps({"verdict": "hit"})}])
    a = L.Agent(workdir=tempfile.mkdtemp(), dispatch_fn=fd, llm_fn=llm,
                cases_path=_tmp_cases())
    res = a.run("白车", "/tmp/a.mp4", mode="deterministic")
    check("llm never called", len(llm.calls) == 0, len(llm.calls))
    check("still honest no-hit", res["verdict"] == "no_hit", res)


if __name__ == "__main__":
    test_playbook_hit_and_events()
    test_insufficient_records_case_and_is_honest()
    test_environment_failure_honest()
    test_llm_loop_and_max_steps()
    test_react_parser()
    test_glue_runner()
    test_ref_only_search()
    test_final_as_tool_call()
    test_mode_deterministic_ignores_llm()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
