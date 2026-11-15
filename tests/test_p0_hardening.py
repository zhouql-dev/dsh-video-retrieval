#!/usr/bin/env python3
"""P0-5/6/7 回归用例（无 pytest，直接 python3 运行；失败 exit 1）。

对应任务：
  P0-5 上传硬上限（分块落盘，超限 413 且不留残文件）
  P0-6 dispatch 的 stderr 必须同时泵出（否则子进程写满 64KB 管道即死锁）
  P0-7 /video/* 的 path 必须落在允许根内（默认 $HOME + uploads + jobs）
用法：cd dsh-video-retrieval && ../video/bin/python tests/test_p0_hardening.py
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "packages", "vr-backend", "vr_backend"))
os.environ.setdefault("FUSION_CASES", os.path.join(tempfile.mkdtemp(), "cases.jsonl"))
os.environ.setdefault("FUSION_JOBS", tempfile.mkdtemp())
os.environ.setdefault("FUSION_UPLOADS", tempfile.mkdtemp())

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def test_upload_cap():
    print("[P0-5] 上传硬上限")
    import server as S
    from fastapi.testclient import TestClient
    S.MAX_UPLOAD_MB = 1                      # 1 MB
    c = TestClient(S.app)
    big = b"x" * (2 * 1024 * 1024)           # 2 MB
    r = c.post("/search/upload", files={"file": ("big.mp4", big, "video/mp4")})
    check("超限返回 413", r.status_code == 413, f"got {r.status_code}")
    left = [f for f in os.listdir(S.UPLOAD_DIR)]
    check("超限不留残文件", all("big.mp4" not in f for f in left), str(left[:3]))
    small = b"y" * (512 * 1024)
    r2 = c.post("/search/upload", files={"file": ("small.mp4", small, "video/mp4")})
    check("限额内返回 200", r2.status_code == 200, f"got {r2.status_code}")


def test_path_guard():
    print("[P0-7] /video/* 路径约束")
    import server as S
    from fastapi.testclient import TestClient
    c = TestClient(S.app)
    for p, what in [("/etc/passwd", "系统文件"), ("/tmp/outside.mp4", "家目录之外")]:
        r = c.get("/video/file", params={"path": p})
        check(f"{what} → 403", r.status_code == 403, f"got {r.status_code}")
        r2 = c.get("/video/info", params={"path": p})
        check(f"{what} info → 403", r2.status_code == 403, f"got {r2.status_code}")
    # 错误体不回显内部路径（P0-15）
    body = c.get("/video/info", params={"path": "/nonexistent.mp4"}).text
    check("错误体不含绝对路径", "/Users/" not in body and "/private/" not in body, body[:80])
    # 允许根内文件可读
    ok_path = os.path.join(S.UPLOAD_DIR, "inside.mp4")
    open(ok_path, "wb").write(b"\x00" * 1024)
    r3 = c.get("/video/file", params={"path": ok_path})
    check("上传目录内 → 200", r3.status_code == 200, f"got {r3.status_code}")


def test_dispatch_stderr():
    print("[P0-6] dispatch 的 stderr 死锁")
    import toolbox

    d = tempfile.mkdtemp()
    script = os.path.join(d, "fake_engine.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(
            "import sys\n"
            "sys.stderr.write('E' * (4 * 1024 * 1024))\n"   # 4MB > 64KB 管道缓冲
            "sys.stderr.flush()\n"
            "print('[stage1] done')\n"
        )
    toolbox.get_tool = lambda name: {"impl": "subprocess", "cost": {}}
    toolbox._script_for = lambda name: script
    toolbox._build_argv = lambda name, params: [sys.executable, script]

    lines = []
    t0 = time.time()
    res = toolbox.dispatch("fake", {"timeout": 20}, on_line=lines.append)
    dt = time.time() - t0
    check("不超时且 status=ok", res.get("status") == "ok", str(res)[:120])
    check("耗时 < 5s（死锁时会 20s 超时）", dt < 5, f"{dt:.1f}s")
    check("stdout 行进来了", any("stage1" in x for x in lines), str(lines[:3]))
    check("stderr 被收进 error 缓冲", len(res.get("error") or "") >= 0)


def main():
    test_upload_cap()
    test_path_guard()
    test_dispatch_stderr()
    print()
    if FAILED:
        print(f"FAIL {len(FAILED)} 项未通过：{FAILED}")
        return 1
    print("PASS 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
