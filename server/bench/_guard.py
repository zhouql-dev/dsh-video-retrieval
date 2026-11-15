#!/usr/bin/env python3
"""P2-2：bench 脚本的统一退出码判定。

bench 过去一律 exit 0 —— 失败也是绿，进 CI 等于没有门禁。
现在每个 run_*.py 在 __main__ 末尾调用 ``finish(out_dir)``：
  - out 目录里没有 json 产物        → exit 1
  - 产物是空文件                    → exit 1
  - 有 acceptance.json 且存在失败项 → exit 1
  - 其余                            → exit 0
"""
from __future__ import annotations
import glob
import json
import os
import sys


def collect_problems(out_dir: str | None) -> list[str]:
    problems: list[str] = []
    if not out_dir:
        return ["no --out dir"]
    if not os.path.isdir(out_dir):
        return [f"out dir missing: {out_dir}"]
    files = glob.glob(os.path.join(out_dir, "*.json"))
    if not files:
        problems.append("no json artifacts in out dir")
    for f in files:
        if os.path.getsize(f) <= 2:
            problems.append(f"empty artifact: {os.path.basename(f)}")
    acc_path = os.path.join(out_dir, "acceptance.json")
    if os.path.exists(acc_path):
        try:
            acc = json.load(open(acc_path, encoding="utf-8"))
            if isinstance(acc, list) and acc:
                bad = [r.get("name") or r.get("line") or "?"
                       for r in acc if isinstance(r, dict) and not r.get("pass")]
                if bad:
                    problems.append(f"acceptance failed: {bad}")
        except Exception as e:                      # noqa: BLE001
            problems.append(f"acceptance.json unreadable: {e}")
    return problems


def finish(out_dir: str | None) -> None:
    problems = collect_problems(out_dir)
    if problems:
        print(f"[bench] FAIL ({out_dir})")
        for p in problems[:6]:
            print(f"        - {p}")
        sys.exit(1)
    print(f"[bench] PASS ({out_dir})")
    sys.exit(0)
