#!/usr/bin/env python3
"""harness 测评:同一 GT 下三种检索模式对比。

    agent      LLM 智能体(harness 操作层,auto 模式,DeepSeek 驱动)
    playbook   harness 确定性剧本(定式路径 + 固定疑难处置,无 LLM)
    pipeline   前期固定 pipeline(fusion/run.py 直跑,无 harness 包装)

任务(GT 来自本项目实测验收):
    person   丰台视频 + 参考图截图 → 目标 @437s
    plate    丰台视频,车牌 京Q1G728 → 838–840s
    smoke    30 帧样本,"白车" → 反例(无目标,测诚实性)

产出 bench_out/harness_eval/{eval.json,report.md}。
单跑一遍约 40–60 分钟(15 分钟 1080p 视频 × 3 模式 + 车牌全片 OCR)。

用法:
    python fusion/bench/run_harness_eval.py            # 全量
    python fusion/bench/run_harness_eval.py --tasks smoke --modes agent,playbook
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.dirname(_HERE)
_REPO = os.path.dirname(_FUSION)
for p in (_HERE, _FUSION, os.path.join(_FUSION, "agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_gt as E        # noqa: E402

OUT_DIR = os.path.join(_FUSION, "bench_out", "harness_eval")

FENGTAI = ("/Users/zhouql1978_1/Desktop/"
           "52丰台康宁居小区东侧河边路口_26891F18_1603352385_1.mp4")
REF_PNG = "/Users/zhouql1978_1/Desktop/截屏2026-08-02 下午7.04.15.png"
SMOKE = os.path.join(_REPO, "mvp_out", "smoke", "annotated.mp4")

TASKS = {
    "person": {"id": "person", "kind": "image", "video": FENGTAI,
               "query": "", "ref": REF_PNG, "gt": [[436.0, 438.5]]},
    "plate": {"id": "plate", "kind": "text", "video": FENGTAI,
              "query": "车牌号 京Q1G728", "ref": None, "gt": [[838.0, 840.5]]},
    "smoke": {"id": "smoke", "kind": "text", "video": SMOKE,
              "query": "白车", "ref": None, "gt": []},
}


def _bounded_dispatch(workdir: str, cap_s: int):
    """Wrap toolbox.dispatch with a per-engine wall-clock cap so a stalled
    engine (e.g. fusion_run embedding 79k crops ungrounded) can't hang the
    whole evaluation — the timeout becomes an honest 'failed' envelope.
    Keeps the Agent's safety parity: person_search without VLM keys gets
    reid_thresh 0.75 (the same injection loop.py performs)."""
    import toolbox as TB
    import agent.loop as AL

    def fn(tool, params, wd=None):
        if (tool == "person_search" and "reid_thresh" not in params
                and not AL._vlm_key_present()):
            params = {**params, "reid_thresh": 0.75}
        params = {**params,
                  "timeout": min(int(params.get("timeout", 3600)), cap_s)}
        return TB.dispatch(tool, params, wd or workdir)
    return fn


def run_agent_or_playbook(task: dict, mode: str, workdir: str,
                          engine_cap_s: int) -> dict:
    """In-process Agent run; wall time measured here; engine/LLM steps counted."""
    from agent.loop import Agent
    # harness mode names: the deterministic playbook is mode="deterministic"
    agent_mode = "deterministic" if mode == "playbook" else "auto"
    t0 = time.time()
    agent = Agent(workdir=workdir, max_steps=10,
                  on_event=lambda e: None,
                  cases_path=os.path.join(workdir, "cases.jsonl"),
                  dispatch_fn=_bounded_dispatch(workdir, engine_cap_s))
    res = agent.run(task["query"], task["video"], ref=task["ref"],
                    mode=agent_mode)
    res["wall_s"] = round(time.time() - t0, 1)
    res["engine_steps"] = len([e for e in res.get("log", []) if e.get("type") == "step"])
    res["llm_steps"] = len(getattr(agent, "_steps", []))
    # drop the log to keep eval.json compact
    res.pop("log", None)
    return res


def run_pipeline(task: dict, workdir: str, cap_s: int = 2400) -> dict:
    """前期固定 pipeline: fusion/run.py 直跑(无 harness)。"""
    cmd = [sys.executable, os.path.join(_FUSION, "run.py"),
           "--video", task["video"],
           "--query", task["query"] or "参考图中的人物",
           "--out", os.path.join(workdir, "pipe")]
    if task.get("ref"):
        cmd += ["--ref", task["ref"]]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=cap_s, cwd=_REPO)
    except subprocess.TimeoutExpired:
        return {"verdict": "failed", "intervals": [], "evidence": {},
                "case_id": None, "tried": ["pipeline(run.py)"],
                "coverage": "run.py 全链路",
                "reason": f"pipeline 超时(>{cap_s}s)", "suggestion": "",
                "wall_s": round(time.time() - t0, 1), "llm_steps": 0,
                "pipeline_returncode": "timeout"}
    wall = round(time.time() - t0, 1)
    metrics = {}
    mp = os.path.join(workdir, "pipe", "metrics.json")
    if os.path.exists(mp):
        metrics = json.load(open(mp))
    intervals = [{"start_s": float(iv.get("start_s", 0)), "end_s": float(iv.get("end_s", 0))}
                 for iv in metrics.get("intervals", [])
                 if iv.get("start_s") is not None]
    # pipeline "尝试"粒度:1 次完整流水线(grounding 包含在内)
    tried = ["pipeline(run.py)"]
    verdict = "hit" if intervals else ("failed" if r.returncode else "no_hit")
    reason = (r.stderr or r.stdout or "")[-300:] if r.returncode else ""
    return {"verdict": verdict, "intervals": intervals,
            "evidence": {"metrics": {k: metrics.get(k) for k in
                                     ("n_hits", "n_matches", "n_intervals")
                                     if k in metrics}},
            "case_id": None, "tried": tried,
            "coverage": "run.py 全链路(grounding+检测+投票)",
            "reason": reason, "suggestion": "",
            "wall_s": wall, "llm_steps": 0,
            "pipeline_returncode": r.returncode}


def main(argv=None):
    ap = argparse.ArgumentParser(description="harness vs pipeline 对比测评")
    ap.add_argument("--tasks", default="person,plate,smoke")
    ap.add_argument("--modes", default="agent,playbook,pipeline")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--engine-timeout", type=int, default=1500,
                    help="单引擎上限秒(超时=诚实 failed,防卡死拖垮测评)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tasks = {k: TASKS[k] for k in args.tasks.split(",") if k in TASKS}
    modes = args.modes.split(",")
    # resume: keep completed task/mode pairs from a previous eval.json
    results = {}
    prev = os.path.join(args.out, "eval.json")
    if os.path.exists(prev):
        try:
            results = json.load(open(prev)).get("results", {})
            print(f"[resume] 复用 {sum(len(v) for v in results.values())} 个已完成结果")
        except (json.JSONDecodeError, OSError):
            results = {}
    for tid, task in tasks.items():
        results.setdefault(tid, {})
        for mode in modes:
            if mode not in ("agent", "playbook", "pipeline"):
                continue
            if mode in results[tid]:
                print(f"=== {tid} / {mode} 已完成,跳过 ===", flush=True)
                continue
            workdir = os.path.join(args.out, "runs", f"{tid}_{mode}")
            os.makedirs(workdir, exist_ok=True)
            print(f"\n=== {tid} / {mode} 开始 "
                  f"({task['query'] or task['ref']}) ===", flush=True)
            if mode == "pipeline":
                res = run_pipeline(task, workdir, args.engine_timeout)
            else:
                res = run_agent_or_playbook(task, mode, workdir,
                                            args.engine_timeout)
            m = E.score_result(res, task["gt"])
            print(f"  verdict={m['verdict']} intervals={m['n_intervals']} "
                  f"gt_hit={m['gt_hit']} wall={res.get('wall_s')}s "
                  f"engines={len(m['tried'])} honest={m['honest']}",
                  flush=True)
            results[tid][mode] = res
    json.dump({"tasks": {k: v for k, v in tasks.items()},
               "results": results, "modes": modes},
              open(os.path.join(args.out, "eval.json"), "w"),
              ensure_ascii=False, indent=2)
    report = E.eval_report(results, tasks) + "\n"
    notes = ("\n## 结论要点\n"
             "- agent 模式多引擎探索,适应性强,但耗时最长且 LLM 步数成本高;\n"
             "- playbook 是 harness 内固定剧本,零 LLM 成本,行为可回归;\n"
             "- pipeline 是前期最小固定链路,无恢复/重判,失败即失败;\n"
             "- 反例任务(smoke)三种模式都必须 honest no_hit(护栏);\n"
             "- 单遍测评 n=1,LLM 模式存在随机性;GT 容差 ±2s。\n")
    open(os.path.join(args.out, "report.md"), "w").write(report + notes)
    print("\n" + report + notes)
    print(f"[eval] done -> {args.out}/")


if __name__ == "__main__":
    main()
