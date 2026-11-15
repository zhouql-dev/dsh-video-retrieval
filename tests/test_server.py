#!/usr/bin/env python3
"""E2 service tests — TestClient over every endpoint + SSE event sequence.

The agent is stubbed (server.AGENT_FACTORY test seam) so no engines launch:
a canned hit result with a fake annotated frame exercises the job lifecycle,
SSE progress stream, artifact serving, case confirm (evolution feed), config
view, and the manual evolve trigger. Run: ``python3 test_server.py``.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import time

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）

import server as S
import case_log as CL
from fastapi.testclient import TestClient

_passed = _failed = 0

# leak guard: the REAL config dir must not gain files while the suite runs
# (captured BEFORE any _setup redirects FUSION_CONFIG to a temp dir)
_REAL_CONFIG_DIR = S.CFG.config_dir()
_REAL_CONFIG_SNAPSHOT = sorted(os.listdir(_REAL_CONFIG_DIR)) \
    if os.path.isdir(_REAL_CONFIG_DIR) else []
# same for the REAL .env (settings persistence must target temp files only)
_REAL_ENV_FILE = S._env_file()
_REAL_ENV_SNAPSHOT = (open(_REAL_ENV_FILE, encoding="utf-8").read()
                      if os.path.exists(_REAL_ENV_FILE) else None)


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


class StubAgent:
    # P0/P1 契约：_run_job 现在会传 profile=；桩必须接受，否则任务线程 TypeError → 超时
    def __init__(self, workdir, max_steps, on_event, cases_path, **kwargs):
        self.workdir, self.on_event, self.cases_path = workdir, on_event, cases_path
        self.kwargs = kwargs

    def run(self, query, video, ref=None, mode="auto", profile=None):
        for e in ({"type": "start", "query": query, "video": video},
                  {"type": "step", "tool": "preflight", "status": "ok",
                   "summary": {"verdict": "OK"}},
                  {"type": "engine", "tool": "person_search",
                   "line": "[stage1] scanned 10/100 frames"},
                  {"type": "step", "tool": "fusion_run", "status": "ok",
                   "summary": {"metrics.json": {"n_hits": 1, "intervals": [
                       {"start_s": 10.0, "end_s": 12.0}]}}},
                  {"type": "final", "verdict": "hit", "n_intervals": 1}):
            self.on_event(e)
        os.makedirs(self.workdir, exist_ok=True)
        with open(os.path.join(self.workdir, "hit_cluster0.jpg"), "wb") as f:
            f.write(b"fake-jpg-bytes")
        return {"verdict": "hit", "intervals": [{"start_s": 10.0, "end_s": 12.0}],
                "evidence": {"n_hits": 1}, "case_id": None,
                "tried": ["preflight", "fusion_run"],
                "coverage": "全片", "reason": "", "suggestion": "", "log": []}


def _setup():
    # hermetic: everything (cases/jobs/CONFIG) goes to temp dirs — the veto and
    # evolve endpoints must never touch the real fusion/config/
    cfg = tempfile.mkdtemp(prefix="srv-cfg-")
    os.environ["FUSION_CONFIG"] = cfg
    os.environ["EVOLVE_NO_BENCH"] = "1"
    S.CASES_PATH = os.path.join(tempfile.mkdtemp(prefix="srv-cases-"), "cases.jsonl")
    S.JOBS_DIR = tempfile.mkdtemp(prefix="srv-jobs-")
    os.makedirs(S.JOBS_DIR, exist_ok=True)
    S.AGENT_FACTORY = StubAgent
    return TestClient(S.app)


def _wait_done(client, job_id, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/search/{job_id}")
        if r.status_code == 200 and r.json().get("status") == "done":
            return r.json()
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish")


def test_health_and_404():
    print("[health + unknown job]")
    client = _setup()
    r = client.get("/health")
    check("health ok", r.status_code == 200 and r.json()["status"] == "ok", r.text)
    check("unknown job -> 404", client.get("/search/nope").status_code == 404)
    check("unknown job events -> 404",
          client.get("/search/nope/events").status_code == 404)


def test_search_flow_and_artifacts():
    print("[POST /search -> job -> result + artifact file]")
    client = _setup()
    r = client.post("/search", json={"video": "/tmp/x.mp4", "query": "白车"})
    check("search accepted", r.status_code == 200 and r.json().get("job_id"), r.text)
    job_id = r.json()["job_id"]
    done = _wait_done(client, job_id)
    check("result verdict hit", done["result"]["verdict"] == "hit", done)
    check("intervals in result",
          done["result"]["intervals"] == [{"start_s": 10.0, "end_s": 12.0}],
          done["result"].get("intervals"))
    arts = done["result"].get("artifacts", [])
    check("annotated frame artifact listed",
          any(a.endswith("hit_cluster0.jpg") for a in arts), arts)
    img = client.get(arts[0])
    check("artifact served", img.status_code == 200 and img.content == b"fake-jpg-bytes",
          img.status_code)
    trav = client.get("/runs/../etc/passwd")
    check("path traversal blocked", trav.status_code in (403, 404), trav.status_code)


def test_sse_events():
    print("[SSE progress stream]")
    client = _setup()
    job_id = client.post("/search", json={"video": "/tmp/x.mp4",
                                          "query": "白车"}).json()["job_id"]
    r = client.get(f"/search/{job_id}/events")
    check("sse 200 + text/event-stream", r.status_code == 200
          and "text/event-stream" in r.headers.get("content-type", ""), r.text[:80])
    body = r.text
    check("start event present", '"type": "start"' in body, body[:200])
    check("step event present", '"type": "step"' in body, body[:200])
    check("engine progress line present", '"type": "engine"' in body
          and "scanned 10/100" in body, body[:400])
    check("final event present", '"type": "final"' in body, body[:200])
    check("job_done carries result", '"type": "job_done"' in body
          and '"verdict": "hit"' in body, body[-400:])
    check("event order start<step<final",
          body.index('"type": "start"') < body.index('"type": "step"')
          < body.index('"type": "final"'))


def test_cases_and_confirm():
    print("[case list + human confirm -> evolution feed]")
    client = _setup()
    case = {"id": "case_20260814_001", "ts": "2026-08-14T00:00:00+00:00",
            "query": "白车", "query_type": "insufficient",
            "inputs": {"video": "/tmp/x.mp4", "ref": None,
                       "provider_status": {"glm": False, "qwen": False}},
            "plan": ["p"], "attempts": [], "outcome": "unresolved"}
    CL.record_case(case, S.CASES_PATH)
    r = client.get("/cases")
    ids = [c["id"] for c in r.json()["cases"]]
    check("case listed", r.status_code == 200 and "case_20260814_001" in ids, r.text)
    r2 = client.post("/cases/case_20260814_001/confirm",
                     json={"gt": {"hit_intervals": [[9.5, 12.5]]}})
    check("confirm ok", r2.status_code == 200
          and r2.json()["human_confirmed"] is True, r2.text)
    check("gt stored", r2.json()["gt"] == {"hit_intervals": [[9.5, 12.5]]}, r2.json())
    listed = client.get("/cases").json()["cases"]
    confirmed = [c for c in listed if c["id"] == "case_20260814_001"][0]
    check("confirmed persisted", confirmed["human_confirmed"] is True, confirmed)
    check("unknown case confirm -> 404",
          client.post("/cases/nope/confirm", json={}).status_code == 404)


def test_config_and_evolve():
    print("[config view + manual evolve trigger]")
    client = _setup()
    r = client.get("/config")
    d = r.json()
    check("config summary + runtime", r.status_code == 200
          and "summary" in d and "runtime" in d
          and "thresholds" in d["runtime"], d)
    # P0-11：/evolve 已改为 POST + CSRF 头
    r2 = client.post("/evolve", headers={"X-Requested-With": "vr-console"})
    check("evolve returns JSON with status",
          r2.status_code == 200 and "status" in r2.json(), r2.text)
    _r405 = client.get("/evolve")
    check("GET /evolve 不可用", _r405.status_code in (404, 405), _r405.status_code)
    r3 = client.get("/evolutions")
    check("evolutions log JSON (empty ok)",
          r3.status_code == 200 and "n" in r3.json() and "events" in r3.json(),
          r3.text)
    # plumbing: /evolve must hand the ACTIVE cases path to run_once
    seen = {}
    S.EVOLVE_RUNNER = lambda **kw: seen.update(kw) or {"status": "test-hook"}
    try:
        client.post("/evolve", headers={"X-Requested-With": "vr-console"})
    finally:
        S.EVOLVE_RUNNER = None
    check("/evolve passes active cases_path",
          seen.get("cases_path") == S.CASES_PATH, seen)


def test_veto_and_gui():
    print("[.veto toggle + GUI page]")
    client = _setup()
    r = client.post("/config/veto", json={"freeze": True, "reason": "test"})
    check("veto on", r.status_code == 200 and r.json()["vetoed"] is True, r.text)
    check("veto visible in config", client.get("/config").json()["vetoed"] is True)
    r2 = client.post("/config/veto", json={"freeze": False})
    check("veto off", r2.json()["vetoed"] is False, r2.text)
    g = client.get("/")
    check("GUI index served", g.status_code == 200
          and "监控视频检索" in g.text, g.status_code)


def test_jobs_persisted():
    print("[jobs.jsonl persistence (#6)]")
    client = _setup()
    job_id = client.post("/search", json={"video": "/tmp/x.mp4",
                                          "query": "白车"}).json()["job_id"]
    _wait_done(client, job_id)
    log = os.path.join(S.JOBS_DIR, "jobs.jsonl")
    check("jobs.jsonl written", os.path.exists(log), log)
    lines = [json.loads(l) for l in open(log) if l.strip()]
    final = [l for l in lines if l["job_id"] == job_id]
    check("job has running+done records", len(final) >= 2
          and final[-1]["status"] == "done"
          and final[-1]["result"]["verdict"] == "hit", final[-1]["status"] if final else None)


def test_upload():
    print("[multipart upload -> server-side path]")
    client = _setup()
    r = client.post("/search/upload",
                    files={"file": ("路口监控.mp4", b"videobytes", "video/mp4")})
    d = r.json()
    check("upload returns path", r.status_code == 200 and d.get("path"), d)
    check("file saved on disk", os.path.exists(d["path"])
          and open(d["path"], "rb").read() == b"videobytes", d)


def test_video_endpoints():
    print("[video panel: /video/info + /video/file + images + transcode]")
    import cv2, numpy as np
    tmp = tempfile.mkdtemp(prefix="srv-vid-")
    # P0-7 之后 /video/* 只允许 uploads/jobs/$HOME（或 VTL_ALLOWED_ROOTS），
    # 这里把临时目录显式加进白名单，避免用例被 403 挡住。
    os.environ["VTL_ALLOWED_ROOTS"] = tmp
    path = os.path.join(tmp, "tiny.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    # a PNG ref image (the refchip bug: images must be servable)
    png = os.path.join(tmp, "ref.png")
    cv2.imwrite(png, np.zeros((10, 10, 3), np.uint8))
    client = _setup()
    r = client.get("/video/info", params={"path": path})
    d = r.json()
    check("info probed", r.status_code == 200 and d["n_frames"] == 10
          and d["width"] == 64, d)
    check("duration/fps computed", 0 < d["duration_s"] <= 1.5 and d["fps"] > 0, d)
    r2 = client.get("/video/file", params={"path": path})
    check("file served", r2.status_code == 200
          and r2.headers["content-type"].startswith("video/"), r2.status_code)
    rp = client.get("/video/file", params={"path": png})
    check("image served (refchip)", rp.status_code == 200
          and rp.headers["content-type"].startswith("image/"), rp.status_code)
    # P0-7 之后：白名单外的路径一律 403；白名单内不存在才是 404
    check("白名单外 -> 403",
          client.get("/video/info", params={"path": "/nope.mp4"}).status_code == 403)
    check("系统文件 -> 403",
          client.get("/video/file", params={"path": "/etc/passwd"}).status_code == 403)
    check("白名单内不存在 -> 404",
          client.get("/video/info", params={"path": os.path.join(tmp, "nope.mp4")}).status_code == 404)
    # transcode: real MPEG-PS (DVR) file -> browser-safe mp4
    ps = os.path.join(tmp, "dvr.mp4")      # extension lies: MPEG-PS inside
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
                    "-f", "mpegts", ps], capture_output=True, timeout=60)
    ti = client.get("/video/info", params={"path": ps}).json()
    check("container detected (mpegts not playable)",
          "mpeg" in (ti.get("container") or "") and ti["browser_playable"] is False, ti)
    tr = client.post("/video/transcode", params={"path": ps}, headers={"X-Requested-With": "vr-console"}).json()
    check("transcode ok", tr.get("status") in ("ok", "cached") and tr.get("url"), tr)
    serve = client.get(tr["url"])
    check("transcoded served as mp4", serve.status_code == 200
          and serve.headers["content-type"].startswith("video/mp4"), serve.status_code)
    tr2 = client.post("/video/transcode", params={"path": ps}, headers={"X-Requested-With": "vr-console"}).json()
    check("second transcode hits cache", tr2.get("status") == "cached", tr2)
    check("bad transcoded key -> 403",
          client.get("/video/transcoded", params={"key": "../../etc"}).status_code == 403)


def test_chat_endpoint():
    print("[chat: honest without key]")
    saved = {k: os.environ.pop(k, None) for k in
             ("LLM_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY", "DASHSCOPE_API_KEY")}
    try:
        client = _setup()
        r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        d = r.json()
        check("no-key honest note", r.status_code == 200 and d["reply"] is None
              and "key" in d["note"], d)
        check("empty messages -> 400",
              client.post("/chat", json={"messages": []}).status_code == 400)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _fake_chat_factory():
    """Scripted _litellm_chat stand-in: pops one reply per call, records
    (messages, tools) of every call, defaults to {'text': '正常'}.
    NOTE: server.py imports from ``agent.loop`` (package-qualified) — patch
    THAT module, not the top-level ``loop`` alias."""
    import agent.loop as AL
    seen = []

    def fake(messages, tools=None):
        seen.append({"messages": messages, "tools": tools})
        return fake.script.pop(0) if fake.script else {"text": "正常"}
    fake.script = []
    return AL, fake, seen


def test_chat_tool_loop():
    print("[chat tools: natural-language config (set + persist + mask)]")
    envfile = os.path.join(tempfile.mkdtemp(prefix="srv-chat-"), ".env")
    os.environ["FUSION_ENV"] = envfile
    saved = {k: os.environ.get(k) for k in
             ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL")}
    os.environ["LLM_API_KEY"] = "sk-oldkey"
    AL, fake, seen = _fake_chat_factory()
    fake.script = [
        {"tool_calls": [{"id": "c1", "name": "set_settings",
                         "arguments": json.dumps({"model": "test-model",
                                                  "llm_key": "sk-newkey1234567890"})}],
         "reasoning_content": "用户要求换模型"},
        {"text": "已按你的要求保存配置。"},
    ]
    real = AL._litellm_chat
    AL._litellm_chat = fake
    try:
        client = _setup()
        r = client.post("/chat", json={"messages": [{"role": "user",
                                                     "content": "把模型换成 test-model,key 换新"}]})
        d = r.json()
        # P0-10：对话通道只放行 model；密钥/base_url 必须被拒
        check("chat tool 只应用 model、拒绝 key",
              os.environ.get("LLM_MODEL") == "test-model"
              and os.environ.get("LLM_API_KEY") != "sk-newkey1234567890",
              (os.environ.get("LLM_MODEL"), os.environ.get("LLM_API_KEY")))
        check("reply + connectivity verdict",
              d.get("reply") and "连通性自检" in d["reply"], d)
        _env_txt = open(envfile).read() if os.path.exists(envfile) else ""
        check("persisted to .env（且不落 key）",
              "LLM_MODEL=test-model" in _env_txt and "sk-newkey" not in _env_txt,
              _env_txt[:80])
        asst = [m for m in seen[0]["messages"] if m["role"] == "assistant"][0]
        check("wire format + reasoning echoed",
              asst["tool_calls"][0]["type"] == "function"
              and "name" in asst["tool_calls"][0]["function"]
              and asst.get("reasoning_content") == "用户要求换模型", asst)
        tool_msgs = [m["content"] for m in seen[0]["messages"]
                     if m["role"] == "tool"]
        # P0-10：对话通道拒绝改密钥，工具结果里只能看到"被拒"，绝无明文/掩码
        check("对话通道拒绝改密钥（无明文）",
              all("sk-newkey1234567890" not in c for c in tool_msgs)
              and any("被拒" in c for c in tool_msgs), tool_msgs)
    finally:
        AL._litellm_chat = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop("FUSION_ENV", None)


def test_chat_get_and_unknown_tool():
    print("[chat tools: get_settings + unknown tool don't crash]")
    saved = {k: os.environ.get(k) for k in
             ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL")}
    os.environ["LLM_API_KEY"] = "sk-fake"
    os.environ["LLM_MODEL"] = "deepseek-v4-pro"
    AL, fake, seen = _fake_chat_factory()
    fake.script = [
        {"tool_calls": [{"id": "g1", "name": "get_settings", "arguments": "{}"}]},
        {"tool_calls": [{"id": "u1", "name": "hack_all", "arguments": "{}"}]},
        {"text": "好的,当前配置如上。"},
    ]
    real = AL._litellm_chat
    AL._litellm_chat = fake
    try:
        client = _setup()
        r = client.post("/chat", json={"messages": [{"role": "user",
                                                     "content": "现在用什么模型?"}]})
        d = r.json()
        check("multi-round loop completes", r.status_code == 200 and d.get("reply"), d)
        tool_msgs = [m["content"] for m in seen[0]["messages"]
                     if m["role"] == "tool"]
        check("get_settings result readable",
              any("deepseek-v4-pro" in c for c in tool_msgs), tool_msgs)
        check("unknown tool -> honest error, loop continues",
              any("没有这个配置工具" in c for c in tool_msgs), tool_msgs)
    finally:
        AL._litellm_chat = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_settings_roundtrip():
    print("[settings: mask on GET, apply+persist on POST]")
    envfile = os.path.join(tempfile.mkdtemp(prefix="srv-env-"), ".env")
    os.environ["FUSION_ENV"] = envfile
    saved = {k: os.environ.pop(k, None) for k in
             ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")}
    try:
        client = _setup()
        # P0-10：base_url 必须命中白名单（非白名单另有用例断言 400）
        r = client.post("/settings", json={
            "llm_base_url": "https://api.deepseek.com/v1",
            "llm_api_key": "sk-abcdef1234567890",
            "llm_model": "test-model"})
        d = r.json()
        check("applied to process env",
              os.environ.get("LLM_API_KEY") == "sk-abcdef1234567890"
              and os.environ.get("LLM_MODEL") == "test-model", os.environ.get("LLM_MODEL"))
        check("GET masks the key", "sk-ab" in d["llm_api_key_masked"]
              and "7890" in d["llm_api_key_masked"]
              and "123456" not in d["llm_api_key_masked"], d["llm_api_key_masked"])
        check("persisted to .env", "LLM_API_KEY=sk-abcdef1234567890"
              in open(envfile).read(), open(envfile).read())
        # GET reflects (masked), still no raw key anywhere
        g = client.get("/settings").json()
        check("settings GET consistent", g["llm_base_url"] == "https://api.deepseek.com/v1"
              and g["llm_api_key_masked"], g)
        # clearing with empty string
        client.post("/settings", json={"llm_api_key": ""})
        check("empty string clears key", "LLM_API_KEY" not in os.environ
              or os.environ["LLM_API_KEY"] == "", os.environ.get("LLM_API_KEY"))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]
        os.environ.pop("FUSION_ENV", None)


def test_ref_only_search_accepted():
    print("[ref-only search: empty query + ref accepted]")
    client = _setup()
    r = client.post("/search", json={"video": "/tmp/x.mp4", "query": "",
                                     "ref": "/tmp/ref.png"})
    check("empty query + ref accepted", r.status_code == 200
          and r.json().get("job_id"), r.text)
    job_id = r.json()["job_id"]
    done = _wait_done(client, job_id)
    check("ref-only job completed", done["status"] == "done", done["status"])


def test_tracked_clip():
    print("[tracked clip: segment cut + frame-accurate boxes]")
    import cv2
    import numpy as np
    tmp = tempfile.mkdtemp(prefix="srv-clip-")
    # a moving white square on black (1s @ 10fps, 64x48)
    path = os.path.join(tmp, "src.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(10):
        fr = np.zeros((48, 64, 3), np.uint8)
        cv2.rectangle(fr, (i * 3, 10), (i * 3 + 8, 22), (255, 255, 255), -1)
        w.write(fr)
    w.release()
    job = "abc123def456"
    client = _setup()                       # _setup swaps S.JOBS_DIR — call first
    jobdir = os.path.join(S.JOBS_DIR, job)
    os.makedirs(jobdir, exist_ok=True)
    json.dump([{"frame": 2, "t_s": 0.2, "box": [6, 10, 8, 12]},
               {"frame": 6, "t_s": 0.6, "box": [18, 10, 8, 12]}],
              open(os.path.join(jobdir, "matches.json"), "w"))
    S.JOBS[job] = {"status": "done", "events": None, "thread": None,
                   "video": path, "query": "", "created": 0,
                   "result": {"verdict": "hit",
                              "intervals": [{"start_s": 0.2, "end_s": 0.7}],
                              "log": []}}
    r = client.get("/video/tracked-clip", params={"job": job, "i": 0})
    d = r.json()
    check("clip generated", r.status_code == 200 and d.get("status") == "ok"
          and d.get("n_boxes") == 2, d)
    clip = os.path.join(jobdir, "tracked_clip0.mp4")
    check("clip file exists", os.path.exists(clip)
          and os.path.getsize(clip) > 500, clip)
    serve = client.get(d["url"])
    check("clip served via /runs", serve.status_code == 200
          and serve.headers["content-type"].startswith("video/mp4"),
          serve.status_code)
    # clip covers the segment only (~0.2-0.7s + 1s tail ≈ <= 2s)
    c = cv2.VideoCapture(clip)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = c.get(cv2.CAP_PROP_FPS) or 10
    c.release()
    check("clip is segment-length only", 0.5 <= n / fps <= 2.0, n / fps)
    check("bad job id rejected",
          client.get("/video/tracked-clip", params={"job": "nope", "i": 0})
          .status_code in (403, 404))
    check("bad index -> 404",
          client.get("/video/tracked-clip", params={"job": job, "i": 9})
          .status_code == 404)


def test_no_real_config_leak():
    print("[hermetic: real fusion/config + .env untouched]")
    cur = sorted(os.listdir(_REAL_CONFIG_DIR)) \
        if os.path.isdir(_REAL_CONFIG_DIR) else []
    check("no new files in real config dir", cur == _REAL_CONFIG_SNAPSHOT,
          f"{_REAL_CONFIG_SNAPSHOT} -> {cur}")
    env_now = (open(_REAL_ENV_FILE, encoding="utf-8").read()
               if os.path.exists(_REAL_ENV_FILE) else None)
    check("real .env content unchanged", env_now == _REAL_ENV_SNAPSHOT,
          f"{'(missing)' if _REAL_ENV_SNAPSHOT is None else _REAL_ENV_SNAPSHOT[:60]!r} "
          f"-> {env_now[:60] if env_now else env_now!r}")


if __name__ == "__main__":
    test_health_and_404()
    test_search_flow_and_artifacts()
    test_sse_events()
    test_cases_and_confirm()
    test_config_and_evolve()
    test_veto_and_gui()
    test_jobs_persisted()
    test_upload()
    test_video_endpoints()
    test_chat_endpoint()
    test_chat_tool_loop()
    test_chat_get_and_unknown_tool()
    test_settings_roundtrip()
    test_ref_only_search_accepted()
    test_tracked_clip()
    test_no_real_config_leak()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
