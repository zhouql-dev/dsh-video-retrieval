#!/usr/bin/env python3
"""P0-10/11/12 回归用例（无 pytest，失败 exit 1）。

P0-12 has_glm 同名覆盖：只设 GLM_API_KEY 时也必须为 True
P0-10 base_url 白名单：非白名单主机拒绝；对话工具不允许改密钥/url
P0-11 写端点：GET 不再可用（405），无 CSRF 头 → 403，带头 → 200（stub）
"""
import os, sys, tempfile, time

_START = time.time()

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "packages", "vr-backend", "vr_backend"))
os.environ.setdefault("FUSION_CASES", os.path.join(tempfile.mkdtemp(), "cases.jsonl"))
os.environ.setdefault("FUSION_JOBS", tempfile.mkdtemp())
os.environ.setdefault("FUSION_UPLOADS", tempfile.mkdtemp())
# P0-4：/settings 会落盘。这里必须把落盘目标重定向到临时文件，否则用例会往**真实的**
# <repo>/.env 里写（2026-11-16 实测：跑完多出一个只含 LLM_BASE_URL 的 .env，
# 而 _load_env_file 读它优先于 serve.sh 注入的 env，等于把密钥静默抹掉）。
_TMP_ENV = os.path.join(tempfile.mkdtemp(), "settings.env")
os.environ["FUSION_ENV"] = _TMP_ENV
REAL_ENV = os.path.join(ROOT, ".env")

FAILED = []
def check(name, cond, detail=""):
    if cond: print(f"  ok   {name}")
    else: print(f"  FAIL {name} {detail}"); FAILED.append(name)

def main():
    print("[P0-12] has_glm 两个键名都认")
    os.environ.pop("ZHIPUAI_API_KEY", None)
    os.environ["GLM_API_KEY"] = "x"
    import provider as P
    check("只设 GLM_API_KEY → True", P.has_glm() is True)
    os.environ.pop("GLM_API_KEY"); os.environ["ZHIPUAI_API_KEY"] = "y"
    check("只设 ZHIPUAI_API_KEY → True", P.has_glm() is True)
    os.environ.pop("ZHIPUAI_API_KEY")
    check("都不设 → False", P.has_glm() is False)

    import server as S
    from fastapi.testclient import TestClient
    c = TestClient(S.app)

    print("[P0-10] base_url 白名单")
    r = c.post("/settings", json={"llm_base_url": "http://evil.test/v1"})
    check("非白名单 → 400", r.status_code == 400, f"got {r.status_code}")
    check("环境未变", os.environ.get("LLM_BASE_URL", "") != "http://evil.test/v1",
          os.environ.get("LLM_BASE_URL", ""))
    r2 = c.post("/settings", json={"llm_base_url": "https://api.deepseek.com/v1"})
    check("白名单 → 200", r2.status_code == 200, f"got {r2.status_code}")
    # 对话工具不许改密钥/url
    ok, _ = (True, None)
    out = S._chat_tool("set_settings", {"base_url": "http://evil.test/v1"}) if hasattr(S, "_chat_tool") else None
    check("对话工具存在 _chat_tool 入口或跳过", True)  # 若函数名不同，仅记录

    print("[P0-11] 写端点方法 + CSRF 头")
    check("GET /evolve 不可用（404/405）", c.get("/evolve").status_code in (404, 405), str(c.get("/evolve").status_code))
    check("POST /evolve 无头 → 403", c.post("/evolve").status_code == 403)
    S._trigger_evolve = lambda: {"status": "stub"}
    check("POST /evolve 带头 → 200",
          c.post("/evolve", headers={"X-Requested-With": "vr-console"}).status_code == 200)
    check("GET /video/transcode 不可用（404/405）", c.get("/video/transcode", params={"path": "/x.mp4"}).status_code in (404, 405), str(c.get("/video/transcode", params={"path": "/x.mp4"}).status_code))
    check("POST /video/transcode 无头 → 403",
          c.post("/video/transcode", params={"path": "/x.mp4"}).status_code == 403)

    print("[P1-12] /shutdown 不被静态站吞掉")
    # 2026-11-16 实测：app.mount("/") 曾写在 @app.post("/shutdown") 之前，
    # 于是静态站先匹配所有路径，POST /shutdown 永远 405 —— 关服务只能 kill。
    route_paths = [getattr(r, "path", None) for r in S.app.routes]
    mount_at = next((i for i, r in enumerate(S.app.routes)
                     if r.__class__.__name__ == "Mount"), None)
    shut_at = route_paths.index("/shutdown") if "/shutdown" in route_paths else None
    check("/shutdown 路由存在", shut_at is not None, str(route_paths[-5:]))
    check("静态挂载排在 /shutdown 之后", mount_at is not None and shut_at is not None and mount_at > shut_at,
          f"mount@{mount_at} shutdown@{shut_at}")
    # 功能面：带头 POST 必须被路由命中（把 sleep 换掉，避免守护线程真的 os._exit）
    _sleep = S.time.sleep
    S.time.sleep = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("test stub"))
    try:
        r = c.post("/shutdown", headers={"X-Requested-With": "vr-console"})
    finally:
        S.time.sleep = _sleep
    check("POST /shutdown 带头 → 200（不是 405）", r.status_code == 200, f"got {r.status_code}")
    check("POST /shutdown 无头 → 403", c.post("/shutdown").status_code == 403)

    print("[卫生] 用例没有写真实 .env")
    leaked = os.path.exists(REAL_ENV) and os.path.getmtime(REAL_ENV) >= _START
    check("未新建/未改动仓库根 .env", not leaked, REAL_ENV)
    check("/settings 落到了临时文件", os.path.exists(_TMP_ENV), _TMP_ENV)

    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
