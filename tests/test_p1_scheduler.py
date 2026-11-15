#!/usr/bin/env python3
"""P1-4 预算调度器单测（纯计算 + 一次 cv2 probe，无模型无网络）。"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
FAILED=[]
def check(n,c,d=""):
    if c: print(f"  ok   {n}")
    else: print(f"  FAIL {n} {d}"); FAILED.append(n)

def main():
    import scheduler as SC
    print("[profile 解析]")
    for mode, want in [("fast","fast"), ("standard","standard"), ("deep","deep"),
                       (None,"fast"), ("", "fast"), ("bogus","fast")]:
        os.environ.pop("VTL_PROFILE", None)
        got = SC.profile_name(mode)
        check(f"mode={mode!r} -> {want}", got == want, got)
    os.environ["VTL_PROFILE"] = "bogus"
    check("非法 VTL_PROFILE 回落 fast", SC.profile_name(None) == "fast", SC.profile_name(None))
    os.environ.pop("VTL_PROFILE", None)

    print("[plan 表驱动]")
    rows = [(0, "fast"), (30, "fast"), (600, "fast"), (3600, "fast"),
            (30, "standard"), (600, "standard"), (300, "deep"), (1800, "deep")]
    prev = None
    for dur, mode in rows:
        try:
            p = SC.plan("__nonexistent__.mp4", mode=mode, duration_s=dur)
        except TypeError:
            # 兼容不同签名：没有 duration_s 参数时用 monkeypatch 的 probe
            SC.probe_duration = lambda v, _d=dur: _d
            p = SC.plan("__nonexistent__.mp4", mode=mode)
        check(f"dur={dur}s mode={mode} 有 step/imgsz",
              isinstance(p, dict) and p.get("step", 0) > 0 and p.get("imgsz", 0) > 0, p)
        if prev and prev[0] == mode:
            check(f"  时长↑ step 不降（{dur}s vs {prev[1]}s）", p["step"] >= prev[2], (prev, p["step"]))
        prev = (mode, dur, p.get("step", 0))

    print("[探测容错]")
    check("不存在的文件 -> 0.0", SC.probe_duration("/definitely/not/here.mp4") == 0.0,
          SC.probe_duration("/definitely/not/here.mp4"))
    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
