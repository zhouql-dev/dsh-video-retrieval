#!/usr/bin/env python3
"""``python -m fusion.agent`` — agent-loop CLI (AGENT-INSTRUCTIONS §7 output).

    python -m fusion.agent --video 路口监控.mp4 --query "白车"
    python -m fusion.agent --video 路口监控.mp4 --ref 截图.png \
        --query "穿粉色外套、白色裤子的女性"
    python -m fusion.agent --video 路口监控.mp4 --query "车牌号 京Q1G728" \
        --mode deterministic

Standard output: one SSE-able JSON event per step (preflight/ground/detect/
signals/score), then the final §7 shape — verdict + intervals + evidence +
case_id; a no-hit reports 尝试了什么/覆盖范围/原因/建议, never "目标不存在".
Exit code 0 on completion regardless of verdict (the verdict lives in the
JSON; nonzero is reserved for crashes).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from loop import Agent  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="video-retrieval agent loop")
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", default="",
                    help="目标描述;仅给 --ref 时可省略(以图搜人)")
    ap.add_argument("--ref", default=None, help="参考图(开启三信号/图片找人)")
    ap.add_argument("--mode", choices=["auto", "deterministic"], default="auto",
                    help="auto=LLM agent loop(无 key 自动降级); deterministic=定式流水线")
    ap.add_argument("--workdir", default=None, help="工作目录(工具产物落这里)")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--cases", default=None, help="案例 JSONL 路径(默认 agent/cases.jsonl)")
    args = ap.parse_args(argv)
    if not (args.query.strip() or args.ref):
        ap.error("--query 与 --ref 至少提供一个(仅参考图 = 以图搜人)")

    def on_event(e):
        print(json.dumps(e, ensure_ascii=False), flush=True)

    agent = Agent(workdir=args.workdir, max_steps=args.max_steps,
                  on_event=on_event, cases_path=args.cases)
    res = agent.run(args.query, args.video, ref=args.ref, mode=args.mode)
    print("# 最终答复 (AGENT-INSTRUCTIONS §7)")
    print(json.dumps({k: v for k, v in res.items() if k != "log"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
