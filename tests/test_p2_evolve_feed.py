#!/usr/bin/env python3
"""P2-6：演化数据入口 —— 已确认案例 + rescoring 产物 → candidates。

真实库里 3 条已确认案例都没有 rescoring 的 scored 行，所以 evolve 仍是 no_data。
本用例用一条"完整"案例证明**接线**是通的（candidates > 0），把数据缺口和代码缺陷分开。
"""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "server"), os.path.join(ROOT, "server", "agent")]
FAILED=[]
def check(n,c,d=""):
    if c: print(f"  ok   {n}")
    else: print(f"  FAIL {n} {d}"); FAILED.append(n)

def main():
    import case_log as CL, evolve as EV
    print("[P2-6] 确认案例 → candidates 接线")
    p = os.path.join(tempfile.mkdtemp(), "cases.jsonl")
    case = {
        "id": "case_feed_1", "ts": "2026-11-15T00:00:00+00:00", "query": "白车",
        "query_type": "routine", "inputs": {"video": "v.mp4", "ref": "r.png"},
        "plan": [], "outcome": "resolved", "human_confirmed": True,
        "gt": {"hit_intervals": [[10.0, 12.0]]},
        "attempts": [{
            "tool": "rescoring", "params": {}, "note": "",
            "envelope": {"outputs": {"scored": [
                {"id": "c1", "t_s": 11.0, "results": {"hue": {"score": 0.9, "raw": {}, "evidence": ""}}},
                {"id": "c2", "t_s": 30.0, "results": {"hue": {"score": 0.1, "raw": {}, "evidence": ""}}},
            ]}},
        }],
    }
    ok, msg = CL.record_case(case, path=p)
    check("记录合法案例", ok, msg)
    conf = CL.confirmed_cases(p)
    check("confirmed_cases 能读到", len(conf) == 1, len(conf))
    cands = CL.extract_candidates_from_case(conf[0])
    check("抽出 2 条候选", len(cands) == 2, len(cands))
    check("标签按区间判定", [c["gt"] for c in cands] == [True, False], [c["gt"] for c in cands])
    ds = EV.merge_datasets(cases_path=p)
    check("merge_datasets 的 sources.cases > 0", ds["sources"]["cases"] > 0, ds["sources"])

    print("[P2-6] 真实库现状（数据缺口，不是代码缺陷）")
    real = os.path.expanduser("~/.dsh/video-retrieval/cases.jsonl")
    if os.path.exists(real):
        rows = [json.loads(l) for l in open(real, encoding="utf-8") if l.strip()]
        conf_real = [r for r in rows if r.get("human_confirmed")]
        scored = sum(1 for r in conf_real for a in (r.get("attempts") or [])
                     if (a.get("envelope") or {}).get("outputs", {}).get("scored"))
        print(f"  总 {len(rows)} 条 / 已确认 {len(conf_real)} 条 / 带 scored 的尝试 {scored} 个")
        check("真实库暂时无 scored 数据（预期，需跑一次 rescoring 的确认案例）", scored == 0, scored)

    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
