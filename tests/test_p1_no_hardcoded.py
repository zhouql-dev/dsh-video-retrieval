#!/usr/bin/env python3
"""P1-9：源码不得出现写死的家目录绝对路径（CI 同款门禁）。"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad=[]
for base in (os.path.join("packages", "vr-backend", "vr_backend"),
             os.path.join("packages", "vr-plugin")):
    for dp,_,fns in os.walk(os.path.join(ROOT, base)):
        for f in fns:
            if not f.endswith((".py", ".js", ".mjs")): continue
            p=os.path.join(dp,f)
            try: txt=open(p,encoding="utf-8").read()
            except Exception: continue
            for i,l in enumerate(txt.splitlines(),1):
                if "/Users/" in l: bad.append(f"{os.path.relpath(p,ROOT)}:{i}")
print("[P1-9] 无硬编码家目录")
if bad:
    print("  FAIL", bad[:5]); sys.exit(1)
print("  ok   0 处")
print("PASS 全部通过"); sys.exit(0)
