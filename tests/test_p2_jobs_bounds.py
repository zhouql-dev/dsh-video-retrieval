#!/usr/bin/env python3
"""P2-5：JOBS 上限淘汰 + jobs.jsonl 轮转。"""
import json, os, sys, tempfile
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT,"packages","vr-backend","vr_backend"))
os.environ["FUSION_JOBS"]=tempfile.mkdtemp(); os.environ["FUSION_UPLOADS"]=tempfile.mkdtemp()
os.environ.setdefault("FUSION_CASES", os.path.join(tempfile.mkdtemp(),"c.jsonl"))
FAILED=[]
def check(n,c,d=""):
    if c: print(f"  ok   {n}")
    else: print(f"  FAIL {n} {d}"); FAILED.append(n)

def main():
    import server as S
    print("[JOBS 上限]")
    S.MAX_JOBS = 10; S.JOBS.clear()
    for i in range(25):
        S.JOBS[f"j{i}"] = {"status": "done", "finished_at": i, "result": {}}
        S._evict_jobs()
    check("len(JOBS) <= MAX_JOBS", len(S.JOBS) <= 10, len(S.JOBS))
    check("淘汰的是最早的", "j0" not in S.JOBS and "j24" in S.JOBS, list(S.JOBS)[:3])
    # 在跑的不淘汰
    S.JOBS["running"] = {"status": "running"}
    for i in range(30):
        S.JOBS[f"k{i}"] = {"status": "done", "finished_at": 100+i, "result": {}}
        S._evict_jobs()
    check("running 不被淘汰", "running" in S.JOBS, "running" in S.JOBS)

    print("[jobs.jsonl 轮转]")
    S.JOBS_LOG_MAX_MB = 0.0005        # ~512B 触发轮转
    for i in range(200):
        S._persist_job(f"r{i}", status="done")
    logs = sorted(f for f in os.listdir(S.JOBS_DIR) if f.startswith("jobs.jsonl"))
    check("产生 .1 轮转文件", "jobs.jsonl.1" in logs, logs)
    check("当前日志不超过阈值太多",
          os.path.getsize(S._jobs_log()) <= S.JOBS_LOG_MAX_MB*1024*1024 + 4096,
          os.path.getsize(S._jobs_log()))

    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
