#!/usr/bin/env python3
"""P1-16：路径契约 —— Python 与 JS 两份解析器必须给出同一个答案。

两个入口（独立启动 / DSH 插件）各自拼路径是历史上所有"foreign server"、
"改了 data/ 却在读另一处"事故的根因。现在 Python 是权威实现
(`vr_backend/paths.py`)，JS 逐条镜像 (`src/paths.js`)。本用例在三种环境下
跑两边的 `--print`，逐键比对；任何一个键漂了都红。

环境：
  1. 默认（清掉 VTL_STATE_DIR / DSH_HOME / SKILL_SCRIPTS / FUSION_WEB / BENCH_DATA）
  2. VTL_STATE_DIR=/tmp/vtl-state-contract
  3. DSH_HOME=/tmp/dsh-home-contract（不给 VTL_STATE_DIR）

跑法：``python3 tests/test_paths_contract.py``（需要 node ≥18，不需要模型）。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BACKEND = os.path.join(ROOT, "packages", "vr-backend")
PLUGIN = os.path.join(ROOT, "packages", "vr-plugin")

sys.path.insert(0, os.path.join(BACKEND, "vr_backend"))

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


PY_KEYS = ("repo_root", "backend_dir", "engines_dir", "state_root",
           "weights_root", "bench_root", "client_root", "dsh_home")

CLEAR = ("VTL_STATE_DIR", "DSH_HOME", "SKILL_SCRIPTS", "FUSION_WEB",
         "BENCH_DATA", "VTL_WEIGHTS_DIR", "VTL_ENV_FILE")


def base_env():
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_py(env):
    e = dict(env, PYTHONPATH=BACKEND)
    r = subprocess.run([sys.executable, "-m", "vr_backend.paths", "--print"],
                       capture_output=True, text=True, cwd=ROOT, env=e, timeout=60)
    if r.returncode != 0:
        return None, r.stderr
    return json.loads(r.stdout), ""


def run_node(env):
    node = shutil.which("node")
    if not node:
        return None, "node not found"
    r = subprocess.run([node, os.path.join(PLUGIN, "src", "paths.js"), "--print"],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=60)
    if r.returncode != 0:
        return None, r.stderr
    return json.loads(r.stdout), ""


def compare(name, env):
    py, py_err = run_py(env)
    js, js_err = run_node(env)
    check(f"{name} · 两端都可解析", py is not None and js is not None,
          f"py={py_err[:120]} js={js_err[:120]}")
    if py is None or js is None:
        return {}
    for k in PY_KEYS:
        check(f"{name} · {k} 一致", py.get(k) == js.get(k), f"py={py.get(k)} js={js.get(k)}")
    return py


def main():
    print("[P1-16] Python ↔ JS 路径契约")

    default = compare("默认", base_env())
    # 默认状态根必须是那个已经在用的根（81 条案例库在这里，换根=静默丢数据）
    expected = os.path.join(os.path.expanduser("~"), ".dsh", "video-retrieval")
    check("默认状态根 = ~/.dsh/video-retrieval", default.get("state_root") == expected,
          str(default.get("state_root")))
    check("默认状态根存在", os.path.isdir(default.get("state_root", "")), default.get("state_root", ""))
    check("后端目录含 server.py",
          os.path.isfile(os.path.join(default.get("backend_dir", ""), "server.py")),
          default.get("backend_dir", ""))
    check("引擎目录含 preflight.py",
          os.path.isfile(os.path.join(default.get("engines_dir", ""), "preflight.py")),
          default.get("engines_dir", ""))
    check("控制台目录含 index.html",
          os.path.isfile(os.path.join(default.get("client_root", ""), "index.html")),
          default.get("client_root", ""))

    custom = compare("VTL_STATE_DIR", dict(base_env(), VTL_STATE_DIR="/tmp/vtl-state-contract"))
    check("VTL_STATE_DIR 生效", custom.get("state_root") == "/tmp/vtl-state-contract",
          str(custom.get("state_root")))

    dsh = compare("DSH_HOME", dict(base_env(), DSH_HOME="/tmp/dsh-home-contract"))
    check("DSH_HOME 生效",
          dsh.get("state_root") == os.path.join("/tmp/dsh-home-contract", "video-retrieval"),
          str(dsh.get("state_root")))

    print("[P1-16] 单点化：后端不再自己拼状态根")
    import config as CFG
    import paths

    old = os.environ.pop("VTL_STATE_DIR", None)
    try:
        check("config.state_root() 委托 paths.state_root()",
              CFG.state_root() == paths.state_root(), CFG.state_root())
        check("路径里没有第二份默认值",
              paths.state_root().endswith(os.path.join(".dsh", "video-retrieval")),
              paths.state_root())
    finally:
        if old is not None:
            os.environ["VTL_STATE_DIR"] = old

    # 插件源码里不许再出现硬编码的状态根（历史事故：两份默认值）
    src = open(os.path.join(PLUGIN, "src", "index.js"), encoding="utf-8").read()
    check("index.js 不再自己拼 .dsh/video-retrieval",
          "'.dsh', 'video-retrieval'" not in src and '".dsh", "video-retrieval"' not in src,
          "found a local state-root default")
    check("index.js 走 paths.js 的 stateRoot()", "stateRoot()" in src, "stateRoot() not used")

    print()
    if FAILED:
        print(f"FAIL {len(FAILED)} 项：{FAILED[:6]}")
        return 1
    print("PASS 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
