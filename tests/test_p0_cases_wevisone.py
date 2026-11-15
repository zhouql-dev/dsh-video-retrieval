#!/usr/bin/env python3
"""P0-8 案例并发 / P0-13 wevisone 冷却重试（无 pytest，失败 exit 1）。"""
import json, os, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server")); sys.path.insert(0, os.path.join(ROOT, "server", "agent"))
CASES = os.path.join(tempfile.mkdtemp(), "cases.jsonl")
os.environ["FUSION_CASES"] = CASES
os.environ["FUSION_JOBS"] = tempfile.mkdtemp()
os.environ["FUSION_UPLOADS"] = tempfile.mkdtemp()
os.environ["VTL_WV_COOLDOWN_S"] = "0"


def _case(cid, query, i):
    """满足 case_log.validate 的最小合法案例。"""
    return {"id": cid, "ts": f"2026-11-15T00:{i:02d}:00+00:00", "query": query,
            "query_type": "environment", "inputs": {"video": "v.mp4", "ref": None},
            "plan": [], "attempts": [], "outcome": "resolved", "human_confirmed": False,
            "gt": {}}

FAILED = []
def check(name, cond, detail=""):
    if cond: print(f"  ok   {name}")
    else: print(f"  FAIL {name} {detail}"); FAILED.append(name)

def test_cases_concurrency():
    print("[P0-8] confirm 与 append 并发不丢行")
    import server as S
    import case_log as CL
    from fastapi.testclient import TestClient
    S.CASES_PATH = CASES
    N_SEED, N_APPEND, N_CONFIRM = 40, 15, 15
    with open(CASES, "w", encoding="utf-8") as f:
        for i in range(N_SEED):
            f.write(json.dumps(_case(f"case_{i:04d}", f"q{i}", i), ensure_ascii=False) + "\n")
    c = TestClient(S.app)
    errors = []
    def appender(k):
        try:
            ok, msg = CL.record_case(_case(f"new_{k:03d}", f"n{k}", 30 + k), path=CASES)
            if not ok: errors.append(msg)
        except Exception as e: errors.append(repr(e))
    def confirmer(i):
        try:
            r = c.post(f"/cases/case_{i:04d}/confirm", json={"gt": {"x": i}})
            if r.status_code != 200: errors.append(f"{i}:{r.status_code}")
        except Exception as e: errors.append(repr(e))
    ts = [threading.Thread(target=appender, args=(k,)) for k in range(N_APPEND)]
    ts += [threading.Thread(target=confirmer, args=(i,)) for i in range(N_CONFIRM)]
    for t in ts: t.start()
    for t in ts: t.join()
    rows = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
    ids = {r["id"] for r in rows}
    check("无异常", not errors, str(errors[:3]))
    check(f"行数 = {N_SEED + N_APPEND}（不丢不重）", len(rows) == N_SEED + N_APPEND, f"got {len(rows)}")
    appended = [f"new_{k:03d}" for k in range(N_APPEND)]
    check("追加的行全部幸存", all(x in ids for x in appended), str([x for x in appended if x not in ids][:3]))
    conf = sum(1 for r in rows if r.get("human_confirmed"))
    check(f"确认生效 {N_CONFIRM} 条", conf == N_CONFIRM, f"got {conf}")

def test_wevisone_cooldown():
    print("[P0-13] wevisone 失败计数 + 冷却重试")
    import wevisone_local as W
    W._MAX_FAILS = 3
    W._state.update({"model": None, "failed": False, "reason": "", "fails": 0, "retry_at": 0.0})
    W._note_failure(RuntimeError("boom"))
    check("第 1 次失败不熔断", W._state["failed"] is False and W._state["fails"] == 1)
    W._note_failure(RuntimeError("boom"))
    check("第 2 次失败不熔断", W._state["failed"] is False)
    W._note_failure(RuntimeError("boom"))
    check("第 3 次熔断", W._state["failed"] is True)
    W._state["retry_at"] = time.time() - 1          # 冷却到期
    W._maybe_reset()
    check("冷却后自动复位", W._state["failed"] is False and W._state["fails"] == 0)

def main():
    test_cases_concurrency()
    test_wevisone_cooldown()
    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
