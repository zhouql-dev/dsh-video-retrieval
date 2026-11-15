#!/usr/bin/env python3
"""P1-16：路径的单一来源（Python 是唯一实现，JS 通过 ``--print`` 复用同一份结果）。

四个入口迟早会漂：独立启动（scripts/serve.sh）、DSH 插件（src/index.js）、
引擎子进程（engines/*.py）、bench 脚本。这里把"东西在哪"一次性算清楚，
其它地方只许问本模块，不许各自拼路径。

优先级（与 ``src/paths.js`` 逐条对齐，``tests/test_paths_contract.py`` 守着）：

  state_root   VTL_STATE_DIR > <DSH_HOME>/video-retrieval
  weights_root VTL_WEIGHTS_DIR > <repo>/models > ~/.cache/vr/weights
  bench_root   BENCH_DATA > <repo>/dataset > <repo>/../dataset

``state_root`` 的默认值刻意保留 ``~/.dsh/video-retrieval``：81 条案例库、
jobs/ 与 config/ 已经在那个根下，插件侧的默认值也是它。换根没有功能收益，
只有一次数据迁移的风险；要换请显式设 ``VTL_STATE_DIR``。
"""
from __future__ import annotations

import json
import os
import sys


def repo_root() -> str:
    """仓库根：从本文件上溯，第一个含 package.json 或 .git 的目录。

    装进 node_modules 时包内没有 .git，但有 package.json（仓库根清单）。
    都没有则回退到 packages/vr-backend 的上一级。
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(cur, "package.json")) or \
                os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def backend_dir() -> str:
    """Python 包目录（server.py / run.py / engines/ 所在处）。"""
    return os.path.dirname(os.path.abspath(__file__))


def engines_dir() -> str:
    """引擎脚本目录。SKILL_SCRIPTS 可覆盖（独立部署时指向外部 skill 源）。"""
    return os.environ.get("SKILL_SCRIPTS") or os.path.join(backend_dir(), "engines")


def dsh_home() -> str:
    return os.environ.get("DSH_HOME") or os.path.join(os.path.expanduser("~"), ".dsh")


def state_root() -> str:
    """唯一可写根。两个入口、四个进程都必须拿到同一个值。"""
    env = os.environ.get("VTL_STATE_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(dsh_home(), "video-retrieval")


def weights_root() -> str:
    env = os.environ.get("VTL_WEIGHTS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    cand = os.path.join(repo_root(), "models")
    if os.path.isdir(cand):
        return cand
    return os.path.join(os.path.expanduser("~"), ".cache", "vr", "weights")


def bench_root() -> str:
    env = os.environ.get("BENCH_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    for cand in (os.path.join(repo_root(), "dataset"),
                 os.path.join(os.path.dirname(repo_root()), "dataset")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(repo_root(), "dataset")


def client_root() -> str:
    """控制台前端（插件资产）。FUSION_WEB 可覆盖。"""
    env = os.environ.get("FUSION_WEB")
    if env:
        return env
    return os.path.join(repo_root(), "packages", "vr-plugin", "client")


def model_path(name: str) -> str:
    """模型权重：权重根 → 仓库根 → 引擎的 ../models。找不到就原样返回（交 ultralytics）。"""
    for cand in (os.path.join(weights_root(), name),
                 os.path.join(repo_root(), name),
                 os.path.join(engines_dir(), "..", "models", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return name


def as_dict() -> dict:
    return {
        "repo_root": repo_root(),
        "backend_dir": backend_dir(),
        "engines_dir": engines_dir(),
        "state_root": state_root(),
        "weights_root": weights_root(),
        "bench_root": bench_root(),
        "client_root": client_root(),
        "dsh_home": dsh_home(),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print" in argv or "--json" in argv or not argv:
        print(json.dumps(as_dict(), indent=2, ensure_ascii=False))
        return 0
    key = argv[0]
    if key not in as_dict():
        print(f"unknown key: {key} (one of: {', '.join(as_dict())})", file=sys.stderr)
        return 2
    print(as_dict()[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
