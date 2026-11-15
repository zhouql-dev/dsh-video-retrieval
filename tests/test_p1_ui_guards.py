#!/usr/bin/env python3
"""P1-13 控制台 XSS / P1-6 镜像门禁 的静态回归（不需要浏览器）。"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FAILED=[]
def check(n,c,d=""):
    if c: print(f"  ok   {n}")
    else: print(f"  FAIL {n} {d}"); FAILED.append(n)

def main():
    html = open(os.path.join(ROOT, "packages", "vr-plugin", "client", "index.html"), encoding="utf-8").read()
    print("[P1-13] 案例库字段必须 escape")
    for field in ("c.query", "c.query_type", "c.outcome", "c.id"):
        check(f"{field} 走 esc()", f"esc({field}" in html or f"esc({field})" in html)
    import re
    raw = re.findall(r"\$\{\s*c\.(query|query_type|outcome|id)\b(?!\s*,)", html)
    # 允许出现在 esc( 内部；粗判：模板里直接插值且两侧不是 esc(
    direct = re.findall(r"<td>\$\{c\.(query|query_type|outcome)\|\|", html)
    check("没有裸插值 ${c.query/…||}", not direct, str(direct[:3]))
    check("JSON 过程块也 esc()", "esc(JSON.stringify(" in html)

    print("[P1-6] 镜像门禁脚本")
    r = subprocess.run(["bash", os.path.join(ROOT,"scripts","sync.sh"), "--check"],
                       capture_output=True, text=True, timeout=120)
    check("scripts/sync.sh --check 可跑且通过", r.returncode == 0,
          (r.stdout or r.stderr)[-160:])
    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
