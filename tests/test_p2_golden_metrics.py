#!/usr/bin/env python3
"""P2-3：metrics golden —— 手算期望值，改动算法/常数即红。"""
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path[:0]=[os.path.join(ROOT,"server"), os.path.join(ROOT,"server","bench")]
FAILED=[]
def check(n,c,d=""):
    if c: print(f"  ok   {n}")
    else: print(f"  FAIL {n} {d}"); FAILED.append(n)

def main():
    import metrics as M
    fx=json.load(open(os.path.join(HERE,"golden","fixtures.json"),encoding="utf-8"))
    print("[interval_coverage]")
    for c in fx["coverage"]:
        got=M.interval_coverage([tuple(x) for x in c["pred"]], [tuple(x) for x in c["gt"]])
        for k,v in c["expect"].items():
            check(f"{c['name']} · {k}", abs(float(got[k])-float(v))<1e-9, f"{got.get(k)} != {v}")
    print("[cmc]")
    for c in fx["cmc"]:
        got=M.cmc_k(c["is_pos"], ks=(1,5,10))
        for k,v in c["expect"].items():
            val = got.get(k)
            check(f"{c['name']} · {k}", val is not None and abs(float(val)-float(v))<1e-9, f"{got}")
    print("[ocr_match]")
    conf=fx["ocr_confusables"]
    for c in fx["ocr"]:
        got=M.ocr_match(c["gt"], c["pred"], conf)
        check(f"{c['name']}", got==c["expect"], f"{got} != {c['expect']}")
    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED[:4]}"); return 1
    print("PASS 全部通过"); return 0

if __name__=="__main__":
    sys.exit(main())
