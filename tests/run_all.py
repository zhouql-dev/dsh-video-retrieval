#!/usr/bin/env python3
"""跑全部测试（含 fusion 迁入的 10 套 + P0 回归套件）。

用法：python3 tests/run_all.py [--quiet]
任一子进程非 0 退出 → 整体 exit 1（这是 CI 的判定口径）。
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_BIN = os.environ.get("VTL_PY") or sys.executable
QUIET = "--quiet" in sys.argv

files = []
for d in (HERE, os.path.join(HERE, "agent"), os.path.join(HERE, "bench")):
    if os.path.isdir(d):
        files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.startswith("test_") and f.endswith(".py")]
files.sort()

total_pass, total_fail, failed = 0, 0, []
for f in files:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY_BIN, f], capture_output=True, text=True, env=env, cwd=ROOT)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+) passed(?:, (\d+) failed)?", out)
    p = int(m.group(1)) if m else 0
    fl = int(m.group(2) or 0) if m else (0 if r.returncode == 0 else 1)
    total_pass += p; total_fail += fl
    ok = r.returncode == 0
    if not ok: failed.append(os.path.basename(f))
    print(f"  {'PASS' if ok else 'FAIL'}  {os.path.basename(f):30s} {p:3d} 通过 / {fl} 失败")
    if not ok and not QUIET:
        tail = [l for l in out.splitlines() if "FAIL" in l or "Error" in l][:5]
        for l in tail: print("        " + l[:140])

print()
print(f"合计：{total_pass} 通过 / {total_fail} 失败 / {len(files)} 个套件")
if failed:
    print("失败套件：" + ", ".join(failed)); sys.exit(1)
print("PASS 全部通过"); sys.exit(0)
