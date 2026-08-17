#!/usr/bin/env python3
"""E2 — agent service: FastAPI + uvicorn (default :8787), SSE progress,
case confirm (evolution feed), config view, manual/watched evolution.

Endpoints
---------
    POST /search                  {video, query, ref?, mode?, max_steps?}
                                  -> {job_id}; Agent.run in a worker thread
    GET  /search/{job_id}         poll: {status, result} (done -> artifacts)
    GET  /search/{job_id}/events  SSE: one JSON event per agent step
    POST /search/upload           multipart file -> server-side path (视频不出本机:
                                  the browser uploads locally, never to the cloud)
    GET  /runs/{job_id}/{path}    annotated frames / engine JSONs of a job
    GET  /cases                   case library (待确认/已确认)
    POST /cases/{id}/confirm      {gt:{hit_intervals,positive_reads}} — the ONE
                                  human action that feeds the evolution layer
    GET  /config                  active runtime config (E0 seam, 自进化回填结果)
    GET  /evolve                  manual evolution trigger (E3 run_once)
    GET  /health

``--watch N`` (minutes) embeds the periodic evolution cycle — server-side
timer calling ``agent.evolve.run_once`` (default cycle 30 min via serve.sh).

The old skill GUI (port 7878) is untouched; this service drives the agent
loop with the same FastAPI+SSE pattern.
"""
from __future__ import annotations
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "agent"))

import case_log as CL          # noqa: E402
import config as CFG           # noqa: E402
from agent.loop import Agent   # noqa: E402

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="video-retrieval agent service",
              description="云边协同监控视频检索 — agent loop + 自进化接缝")

# runtime paths (env-overridable; monkeypatched by tests)
CASES_PATH = os.environ.get("FUSION_CASES") or CL.DEFAULT_CASES_PATH
JOBS_DIR = os.environ.get("FUSION_JOBS") or os.path.join(_HERE, "jobs")
UPLOAD_DIR = os.environ.get("FUSION_UPLOADS") or os.path.join(_HERE, "uploads")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# job registry: job_id -> {status, events:Queue, result, thread, query, video}
JOBS: dict[str, dict] = {}

AGENT_FACTORY = Agent          # test seam


def _jobs_log() -> str:
    """jobs.jsonl path under the ACTIVE jobs dir (test seam may swap it)."""
    return os.path.join(JOBS_DIR, "jobs.jsonl")


def _persist_job(job_id: str, **fields):
    """Append one job-state line (crash-safe history; replayed on boot)."""
    try:
        with open(_jobs_log(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"job_id": job_id, "ts": time.time(), **fields},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass                     # persistence is best-effort, never fatal


def _load_jobs():
    """Replay jobs.jsonl into JOBS so history survives restarts (#6)."""
    log = _jobs_log()
    if not os.path.exists(log):
        return
    seen: dict[str, dict] = {}
    try:
        with open(log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jid = rec.get("job_id")
                if not jid:
                    continue
                cur = seen.setdefault(jid, {"status": "running", "events": None,
                                            "result": None, "thread": None,
                                            "query": "", "video": "", "created": 0})
                cur["status"] = rec.get("status", cur["status"])
                cur["result"] = rec.get("result", cur["result"])
                cur["query"] = rec.get("query", cur["query"])
                cur["video"] = rec.get("video", cur["video"])
    except OSError:
        return
    for jid, j in seen.items():
        j["events"] = queue.Queue()      # historical jobs stream job_done only
        if j["status"] == "running":
            # the server that ran it is gone — honest about the interruption
            j["status"] = "interrupted"
            j["result"] = j["result"] or {"verdict": "inconclusive",
                                          "intervals": [], "evidence": {},
                                          "error": "服务重启,任务被中断(未完成)",
                                          "artifacts": []}
        JOBS[jid] = j


_load_jobs()


class SearchRequest(BaseModel):
    video: str
    query: str = ""             # 可空:仅参考图时走以图搜人
    ref: str | None = None
    mode: str = "auto"          # auto | deterministic
    max_steps: int = 12


class ConfirmRequest(BaseModel):
    gt: dict = {}


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

def _list_artifacts(job_id: str, workdir: str) -> list[str]:
    arts = []
    for root, _, files in os.walk(workdir):
        for f in files:
            if f.endswith((".jpg", ".png")):
                arts.append("/runs/" + job_id + "/"
                            + os.path.relpath(os.path.join(root, f), workdir))
    # hit_cluster files must line up with the time-ordered intervals:
    # lexicographic sort puts hit_cluster10 before hit_cluster2 — sort by the
    # numeric cluster index instead.
    def _key(a: str):
        m = re.search(r"hit_cluster(\d+)", a)
        return (int(m.group(1)) if m else 1 << 30, a)
    return sorted(arts, key=_key)[:200]


def _run_job(job_id: str, req: SearchRequest):
    job = JOBS[job_id]
    workdir = os.path.join(JOBS_DIR, job_id)
    try:
        agent = AGENT_FACTORY(workdir=workdir, max_steps=req.max_steps,
                              on_event=job["events"].put, cases_path=CASES_PATH)
        res = agent.run(req.query, req.video, ref=req.ref, mode=req.mode)
        res.setdefault("artifacts", _list_artifacts(job_id, workdir))
        job["result"] = res
        job["status"] = "done"
    except Exception as e:                      # noqa: BLE001 — report, never die
        job["result"] = {"verdict": "inconclusive", "intervals": [],
                         "evidence": {}, "case_id": None,
                         "error": f"{type(e).__name__}: {e}", "artifacts": []}
        job["status"] = "failed"
    finally:
        if job.get("cancelled"):
            job["status"] = "cancelled"
        job["events"].put(None)                 # SSE sentinel
        _persist_job(job_id, status=job["status"], result=job["result"],
                     query=job["query"], video=job["video"])


@app.post("/search/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running job: terminate its engine subprocesses (they carry
    ``jobs/<id>`` in their argv) and mark the job cancelled."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"unknown job {job_id}")
    if job["status"] in ("done", "failed", "cancelled"):
        return {"job_id": job_id, "status": job["status"], "cancelled": False}
    job["cancelled"] = True
    try:
        subprocess.run(["pkill", "-f", f"jobs/{job_id}"], capture_output=True,
                       timeout=15)
    except Exception as e:                      # noqa: BLE001 — best effort
        print(f"[cancel] pkill failed: {e}", flush=True)
    return {"job_id": job_id, "status": "cancelling", "cancelled": True}


@app.post("/search")
def search(req: SearchRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "events": queue.Queue(), "result": None,
                    "thread": None, "query": req.query, "video": req.video,
                    "created": time.time()}
    JOBS[job_id]["thread"] = threading.Thread(
        target=_run_job, args=(job_id, req), daemon=True)
    JOBS[job_id]["thread"].start()
    _persist_job(job_id, status="running", query=req.query, video=req.video)
    return {"job_id": job_id, "status": "running"}


@app.get("/search/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"unknown job {job_id}")
    return {"job_id": job_id, "status": job["status"], "query": job["query"],
            "video": job["video"], "result": job["result"]}


@app.get("/search/{job_id}/events")
def job_events(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"unknown job {job_id}")
    q = job["events"]

    def gen():
        yield "retry: 1000\n\n"
        if job.get("thread") is None:
            # historical job replayed from jobs.jsonl — no live steps to stream
            done = {"type": "job_done", "status": job["status"],
                    "result": job["result"]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            return
        while True:
            try:
                e = q.get(timeout=25)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if e is None:
                break
            yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        done = {"type": "job_done", "status": job["status"], "result": job["result"]}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/search/upload")
async def search_upload(file: UploadFile = File(...)):
    """Browser file upload -> server-side path (video stays on this machine).
    The GUI then POSTs that path to /search."""
    safe = re.sub(r"[^\w.\-]+", "_", file.filename or "upload.bin")
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe}")
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"path": path, "filename": safe}


@app.get("/runs/{job_id}/{path:path}")
def serve_run_file(job_id: str, path: str):
    base = os.path.abspath(JOBS_DIR)
    p = os.path.abspath(os.path.join(base, job_id, path))
    if not p.startswith(base + os.sep):
        raise HTTPException(403, "path traversal blocked")
    if os.path.isfile(p):
        return FileResponse(p)
    raise HTTPException(404, f"no such file {path}")


# --------------------------------------------------------------------------- #
# cases (人工确认 → 演化层养料)
# --------------------------------------------------------------------------- #

@app.get("/cases")
def list_cases():
    cases = list(CL.iter_cases(CASES_PATH))
    return {"n": len(cases),
            "n_confirmed": sum(1 for c in cases if c.get("human_confirmed")),
            "cases": cases[-200:]}


@app.post("/cases/{case_id}/confirm")
def confirm_case(case_id: str, req: ConfirmRequest):
    cases = list(CL.iter_cases(CASES_PATH))
    target = next((c for c in cases if c.get("id") == case_id), None)
    if target is None:
        raise HTTPException(404, f"unknown case {case_id}")
    target["human_confirmed"] = True
    target["gt"] = req.gt or {}
    os.makedirs(os.path.dirname(CASES_PATH), exist_ok=True)
    tmp = CASES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, CASES_PATH)
    return target


# --------------------------------------------------------------------------- #
# config + evolution
# --------------------------------------------------------------------------- #

@app.get("/config")
def get_config():
    return {"summary": CFG.summary(), "runtime": CFG.load_runtime_config(),
            "vetoed": CFG.vetoed()}


EVOLVE_RUNNER = None            # test seam (else lazy-import agent.evolve.run_once)


def _trigger_evolve() -> dict:
    """Lazy import keeps E2 runnable before E3 lands (returns unavailable)."""
    if EVOLVE_RUNNER is not None:
        return EVOLVE_RUNNER(cases_path=CASES_PATH)
    try:
        from agent.evolve import run_once
    except Exception as e:                      # noqa: BLE001
        return {"status": "unavailable", "error": f"{type(e).__name__}: {e}"}
    return run_once(cases_path=CASES_PATH)


@app.get("/evolve")
def evolve_now():
    return _trigger_evolve()


@app.get("/evolutions")
def evolutions():
    """Last 100 lines of the evolution log (fusion/config/evolutions.jsonl)."""
    path = os.path.join(CFG.config_dir(), "evolutions.jsonl")
    if not os.path.exists(path):
        return {"n": 0, "events": []}
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"n": len(events), "events": events[-100:]}


class VetoRequest(BaseModel):
    freeze: bool = True
    reason: str = ""


@app.post("/config/veto")
def config_veto(req: VetoRequest):
    state = CFG.set_veto(req.freeze, req.reason)
    return {"vetoed": state}


@app.get("/health")
def health():
    return {"status": "ok", "jobs": {jid: j["status"] for jid, j in JOBS.items()},
            "cases_path": CASES_PATH, "config": CFG.summary()}


# --------------------------------------------------------------------------- #
# video panel: metadata probe + local playback (视频不出本机: 流式回给本机浏览器)
# --------------------------------------------------------------------------- #

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".dav", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MEDIA_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
                ".mkv": "video/x-matroska", ".webm": "video/webm",
                ".avi": "video/x-msvideo", ".ts": "video/mp2t", ".dav": "application/octet-stream",
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}


def _probe_container(path: str) -> str | None:
    """ffprobe the REAL container (a DVR .mp4 is often MPEG-PS inside)."""
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=format_name", "-of", "csv=p=0",
                            path], capture_output=True, text=True, timeout=30)
        names = (r.stdout or "").strip().split(",")
        return names[0] if names else None
    except Exception:                       # noqa: BLE001 — ffprobe is best-effort
        return None


@app.get("/video/info")
def video_info(path: str):
    """cv2 probe: duration / fps / frames / resolution + REAL container."""
    import cv2
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(404, f"video not found: {path}")
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    dur = (n / fps) if (n and fps) else 0.0
    container = _probe_container(path)
    ext = os.path.splitext(path)[1].lower()
    browser_playable = (container in ("mp4", "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm",
                                      "webm") or container == "mov")
    return {"path": path, "duration_s": round(dur, 1), "fps": round(fps, 2),
            "n_frames": n, "width": w, "height": h,
            "size_mb": round(os.path.getsize(path) / 1048576, 1),
            "container": container, "extension": ext,
            "browser_playable": bool(browser_playable)}


@app.get("/video/file")
def video_file(path: str):
    """Stream a local video OR image to THIS browser (localhost only; the file
    never leaves the machine — playback source for the panels + ref thumbnails)."""
    ext = os.path.splitext(path or "")[1].lower()
    if ext not in _VIDEO_EXTS and ext not in _IMAGE_EXTS:
        raise HTTPException(404, "not a playable local file")
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"))


# --------------------------------------------------------------------------- #
# transcode fallback: DVR files (MPEG-PS/TS, G711 audio) -> browser-safe mp4
# --------------------------------------------------------------------------- #

_TRANSCODE_DIR = os.path.join(JOBS_DIR, ".transcode")
os.makedirs(_TRANSCODE_DIR, exist_ok=True)


@app.get("/video/transcode")
def video_transcode(path: str):
    """Remux/transcode to a browser-safe h264+aac mp4 (cached by path hash).
    MPEG-PS/DVR containers cannot play in any browser — this is the fix."""
    import hashlib
    import subprocess
    if not os.path.isfile(path):
        raise HTTPException(404, f"video not found: {path}")
    key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
    out = os.path.join(_TRANSCODE_DIR, key + ".mp4")
    if os.path.exists(out):
        return {"status": "cached", "url": f"/video/transcoded?key={key}",
                "note": "已转封装(缓存)"}
    tmp = out[:-4] + ".part.mp4"           # keep .mp4 ext so ffmpeg infers the muxer
    # -c:v copy 保留画质;音频转 AAC(G711 浏览器不支持);faststart 保证可流式播放
    cmd = ["ffmpeg", "-y", "-i", path, "-map", "0:v:0", "-map", "0:a?",
           "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "转封装超时(>30 分钟)"}
    if r.returncode != 0 or not os.path.exists(tmp):
        # fall back to full re-encode (some DVR h264 streams refuse copy)
        cmd2 = ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", tmp]
        try:
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "转封装超时(>60 分钟)"}
        if r2.returncode != 0:
            return {"status": "failed",
                    "error": (r2.stderr or "ffmpeg failed")[-400:]}
    os.replace(tmp, out)
    return {"status": "ok", "url": f"/video/transcoded?key={key}",
            "note": "转封装完成(h264+aac, 可播放)"}


@app.get("/video/transcoded")
def video_transcoded(key: str):
    """Serve a cached transcode. key is a sha1 fragment — traversal-proof."""
    import re
    if not re.fullmatch(r"[0-9a-f]{8,64}", key):
        raise HTTPException(403, "bad key")
    path = os.path.join(_TRANSCODE_DIR, key + ".mp4")
    if not os.path.isfile(path):
        raise HTTPException(404, "transcode not found — call /video/transcode first")
    return FileResponse(path, media_type="video/mp4")


# --------------------------------------------------------------------------- #
# tracked segment clip: 目标出现→消失的跟踪片段(逐帧 bounding box 叠加)
# --------------------------------------------------------------------------- #

@app.get("/video/tracked-clip")
def tracked_clip(job: str, i: int):
    """Cut interval i of job's result into a standalone mp4 with the target's
    bounding boxes drawn frame-accurately (boxes from the engines' matches.json).
    The clip starts when the target appears and ends when it disappears."""
    import subprocess
    if not re.fullmatch(r"[0-9a-f]{8,32}", job):
        raise HTTPException(403, "bad job id")
    rec = JOBS.get(job)
    if not rec or not rec.get("result"):
        raise HTTPException(404, f"unknown/finished job {job}")
    intervals = rec["result"].get("intervals") or []
    if not (0 <= i < len(intervals)):
        raise HTTPException(404, f"no interval {i}")
    iv = intervals[i]
    start, end = float(iv["start_s"]), float(iv["end_s"])
    video = rec.get("video")
    if not video or not os.path.isfile(video):
        raise HTTPException(404, "job video unavailable")

    # collect per-frame boxes from every engine matches.json under the job dir
    jobdir = os.path.join(JOBS_DIR, job)
    boxes = []                       # (frame, [x,y,w,h])
    for root, _, files in os.walk(jobdir):
        if "matches.json" not in files:
            continue
        try:
            for m in json.load(open(os.path.join(root, "matches.json"))):
                fr, ts, bx = m.get("frame"), m.get("t_s"), m.get("box")
                if fr is not None and ts is not None and bx \
                        and start - 0.5 <= float(ts) <= end + 0.5:
                    boxes.append((int(fr), [int(v) for v in bx[:4]]))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    boxes = sorted(set((fr, tuple(bx)) for fr, bx in boxes))
    boxes = [(fr, list(bx)) for fr, bx in boxes]
    fps = 25.0
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
    except Exception:                       # noqa: BLE001
        pass

    # box persistence: each box is drawn from its frame until the next one.
    # Input-seek (-ss before -i) is fast but renumbers output frames from 0 —
    # shift every box frame by the seek floor so enable=between(n,...) matches.
    # A minimum 3s window (1.5s before/after) keeps single-frame hits watchable.
    seek_start = max(0.0, start - 1.5)
    clip_len = (end - seek_start) + 1.5
    floor_f = int(seek_start * fps)
    vf_parts = []
    for k, (fr, bx) in enumerate(boxes):
        nxt = boxes[k + 1][0] - 1 if k + 1 < len(boxes) else 1 << 30
        # matches.json boxes are [x1,y1,x2,y2] (xyxy) — convert to drawbox xywh
        x1, y1, x2, y2 = bx[:4]
        x, y, w, h = int(x1), int(y1), max(0, int(x2 - x1)), max(0, int(y2 - y1))
        vf_parts.append(
            f"drawbox=x={x}:y={y}:w={w}:h={h}:color=red@0.85:t=4"
            f":enable='between(n,{max(0, fr - floor_f)},{max(0, nxt - floor_f)})'")
    vf = ",".join(vf_parts) or "null"

    out = os.path.join(jobdir, f"tracked_clip{i}.mp4")
    cmd = ["ffmpeg", "-y", "-ss", str(seek_start), "-i", video,
           "-t", str(clip_len), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", out]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "片段生成超时"}
    if r.returncode != 0 or not os.path.exists(out):
        return {"status": "failed", "error": (r.stderr or "ffmpeg failed")[-300:]}
    return {"status": "ok", "url": f"/runs/{job}/tracked_clip{i}.mp4",
            "n_boxes": len(boxes), "start_s": round(start, 1),
            "end_s": round(end, 1),
            "note": (f"跟踪片段(逐帧框 {len(boxes)} 个)" if boxes
                     else "该命中无逐帧框数据,输出为片段裁切")}


# --------------------------------------------------------------------------- #
# chat (与 LLM 直接对话 — 中栏对话模式的通用通道)
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    messages: list[dict]          # [{role, content}] — history kept client-side


_SYS_AGENT = None                # cached for /chat system-prompt assembly


def _chat_system_prompt() -> str:
    global _SYS_AGENT
    if _SYS_AGENT is None:
        from agent.loop import Agent
        _SYS_AGENT = Agent(workdir="/tmp/chat-context")
    return ("你是监控视频检索智能体(云边协同,视频不出本机)。可以聊检索方法、"
            "解读检索结果、给出检索建议;实际检索任务通过「检索」按钮/工具执行。\n\n"
            "**配置能力**:你可以帮用户查看/修改 LLM 配置(模型/接口地址/密钥)——"
            "用户用自然语言提出要求即可,如'把模型换成 xxx'、'现在用什么模型'。"
            "修改立即生效并保存;密钥绝不回显明文,只显示掩码。\n\n"
            + _SYS_AGENT._system_prompt()[:9000])


_CHAT_TOOLS = [
    {"type": "function", "function": {
        "name": "get_settings",
        "description": "查看当前 LLM/VLM 配置(接口地址、模型、各 key 的掩码状态)",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_settings",
        "description": "修改 LLM/VLM 配置并立即生效+持久化。字段可空:base_url、"
                       "model、llm_key、zhipu_key(智谱VLM)、dashscope_key(阿里兜底)。"
                       "传空字符串=清除该 key",
        "parameters": {"type": "object",
                       "properties": {
                           "base_url": {"type": "string"},
                           "model": {"type": "string"},
                           "llm_key": {"type": "string"},
                           "zhipu_key": {"type": "string"},
                           "dashscope_key": {"type": "string"}}}}},
]


def _run_chat_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute one chat tool. Returns (json_text, changed_llm_channel)."""
    if name == "get_settings":
        s = get_settings()
        return (json.dumps({"配置": {
                    "llm_base_url": s["llm_base_url"] or "(未设置,用默认)",
                    "llm_model": s["llm_model"] or "(未设置)",
                    "llm_api_key": s["llm_api_key_masked"] or "(未配置)",
                    "zhipuai_api_key": s["zhipuai_api_key_masked"] or "(未配置)",
                    "dashscope_api_key": s["dashscope_api_key_masked"] or "(未配置)",
                }}, ensure_ascii=False), False)
    if name == "set_settings":
        updates = {}
        if args.get("base_url") is not None:
            updates["LLM_BASE_URL"] = str(args["base_url"]).strip()
        if args.get("model") is not None:
            updates["LLM_MODEL"] = str(args["model"]).strip()
        if args.get("llm_key") is not None:
            updates["LLM_API_KEY"] = str(args["llm_key"]).strip()
        if args.get("zhipu_key") is not None:
            updates["ZHIPUAI_API_KEY"] = str(args["zhipu_key"]).strip()
        if args.get("dashscope_key") is not None:
            updates["DASHSCOPE_API_KEY"] = str(args["dashscope_key"]).strip()
        if not updates:
            return (json.dumps({"error": "没有给出要修改的字段"},
                               ensure_ascii=False), False)
        apply_settings(updates)
        s = get_settings()
        changed = bool(updates.keys() & {"LLM_BASE_URL", "LLM_API_KEY",
                                         "LLM_MODEL"})
        return (json.dumps({"已保存": {
                    "llm_base_url": s["llm_base_url"],
                    "llm_model": s["llm_model"],
                    "llm_api_key": s["llm_api_key_masked"] or "(已清除)",
                    "zhipuai_api_key": s["zhipuai_api_key_masked"] or "(未配置)",
                    "dashscope_api_key": s["dashscope_api_key_masked"] or "(未配置)",
                }}, ensure_ascii=False), changed)
    return (json.dumps({"error": f"没有这个配置工具: {name}"},
                       ensure_ascii=False), False)


def _connectivity_check() -> str:
    """One tiny call on the CURRENT channel; returns a short verdict."""
    from agent.loop import _litellm_chat, LITELM_LAST_ERROR, llm_key_present
    if not llm_key_present():
        return "未配置 key,跳过测试"
    try:
        out = _litellm_chat([{"role": "user", "content": "只回复两个字:正常"}])
    except Exception as e:              # noqa: BLE001
        return f"测试失败: {type(e).__name__}: {str(e)[:80]}"
    if out and out.get("text"):
        return "正常"
    return f"测试失败: {LITELM_LAST_ERROR[:80] or '无响应'}"


@app.post("/chat")
def chat(req: ChatRequest):
    """Chat with tool-calling: the dialog can get/set LLM config in natural
    language (agent-style operation). Up to 4 tool rounds, then a text reply."""
    from agent.loop import _litellm_chat, llm_key_present
    if not req.messages:
        raise HTTPException(400, "messages required")
    if not llm_key_present():
        return {"reply": None,
                "note": "未配置 LLM key —— 直接对话告诉我'设置 key 为 sk-xxx'也可以,"
                        "但需要先通过配置页或 .env 给一个可用的通道。"
                        "(检索功能不受影响:无 key 走确定性剧本。)"}
    messages = [{"role": "system", "content": _chat_system_prompt()}] + req.messages[-20:]
    changed_channel = False
    reply: str | None = None
    for _ in range(4):
        out = _litellm_chat(messages, tools=_CHAT_TOOLS)
        if out is None:
            from agent.loop import LITELM_LAST_ERROR
            return {"reply": None,
                    "note": "LLM 通道无响应"
                            + (f"({LITELM_LAST_ERROR[:120]})" if LITELM_LAST_ERROR else "")
                            + " —— 可对话让我换模型/key(如'把模型换成 glm-4-flash')。"}
        tcs = out.get("tool_calls")
        if not tcs:
            reply = out.get("text") or ""
            break
        wire = [{"id": tc.get("id", f"c{i}"), "type": "function",
                 "function": {"name": tc.get("name", ""),
                              "arguments": tc.get("arguments", "{}")}}
                for i, tc in enumerate(tcs)]
        asst = {"role": "assistant", "content": "", "tool_calls": wire}
        if out.get("reasoning_content"):
            asst["reasoning_content"] = out["reasoning_content"]
        messages.append(asst)
        for tc, w in zip(tcs, wire):
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            text, did_change = _run_chat_tool(tc.get("name", ""), args)
            messages.append({"role": "tool", "tool_call_id": w["id"],
                             "content": text})
            changed_channel = changed_channel or did_change
    if reply is None:                       # loop exhausted — one final text call
        out = _litellm_chat(messages)
        reply = (out or {}).get("text") or ""
    if changed_channel:
        reply = (reply + f"\n\n> 连通性自检: {_connectivity_check()}").strip()
    return {"reply": reply}


# --------------------------------------------------------------------------- #
# runtime settings (LLM/VLM key & url — 配置页)
# --------------------------------------------------------------------------- #

def _env_file() -> str:
    """The .env path settings persist to (test seam may swap FUSION_ENV)."""
    return os.environ.get("FUSION_ENV") or os.path.join(
        os.path.dirname(_HERE), ".env")

_SETTING_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
                 "ZHIPUAI_API_KEY", "DASHSCOPE_API_KEY")


def _mask(v: str) -> str:
    return (v[:5] + "…" + v[-4:]) if v and len(v) > 12 else ("已配置" if v else "")


@app.get("/settings")
def get_settings():
    cur = {k: os.environ.get(k, "") for k in _SETTING_KEYS}
    return {"llm_base_url": cur["LLM_BASE_URL"], "llm_model": cur["LLM_MODEL"],
            "llm_api_key_masked": _mask(cur["LLM_API_KEY"]),
            "zhipuai_api_key_masked": _mask(cur["ZHIPUAI_API_KEY"]),
            "dashscope_api_key_masked": _mask(cur["DASHSCOPE_API_KEY"]),
            "env_file": _env_file()}


class SettingsRequest(BaseModel):
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    zhipuai_api_key: str | None = None
    dashscope_api_key: str | None = None


_FIELD_MAP = {"llm_base_url": "LLM_BASE_URL", "llm_api_key": "LLM_API_KEY",
              "llm_model": "LLM_MODEL",
              "zhipuai_api_key": "ZHIPUAI_API_KEY",
              "dashscope_api_key": "DASHSCOPE_API_KEY"}


def apply_settings(updates: dict) -> dict:
    """Apply {env_name: value} immediately (os.environ) AND persist to .env.
    Shared by the HTTP endpoint and the chat tool so both behave identically.
    Empty string clears a key. Returns the masked settings view."""
    if not updates:
        raise HTTPException(400, "no fields to update")
    for k, v in updates.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    # persist: rewrite .env preserving unrelated lines
    lines = []
    if os.path.exists(_env_file()):
        with open(_env_file(), encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines()
                     if l.strip() and not l.startswith("#")
                     and l.split("=", 1)[0].strip() not in updates]
    lines += [f"{k}={v}" for k, v in updates.items() if v]
    os.makedirs(os.path.dirname(_env_file()), exist_ok=True)
    with open(_env_file(), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return get_settings()


@app.post("/settings")
def set_settings(req: SettingsRequest):
    updates = {env: (val or "").strip()
               for f, env in _FIELD_MAP.items()
               if (val := getattr(req, f)) is not None}
    return apply_settings(updates)


# GUI (single page, embedded CSS/JS, no external deps) — mounted LAST so the
# API routes above keep precedence.
WEB_DIR = os.environ.get("FUSION_WEB") or os.path.join(_HERE, "web")
os.makedirs(WEB_DIR, exist_ok=True)
from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


# --------------------------------------------------------------------------- #
# entry point (--watch embeds the periodic evolution cycle)
# --------------------------------------------------------------------------- #

def _watch_loop(interval_min: float):
    while True:
        time.sleep(max(interval_min, 0.05) * 60)
        try:
            print("[watch] evolve:", _trigger_evolve(), flush=True)
        except Exception as e:                  # noqa: BLE001
            print("[watch] evolve error:", e, flush=True)


def _warmup():
    """Preload YOLO-World + CLIP on MPS/CUDA with a 5-frame dummy video (#3):
    without this the FIRST user job pays the ~90 s model-load cost."""
    import cv2
    import numpy as np
    t0 = time.time()
    wdir = os.path.join(JOBS_DIR, ".warmup")
    os.makedirs(wdir, exist_ok=True)
    video = os.path.join(wdir, "warmup.mp4")
    try:
        w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                            (320, 240))
        for _ in range(5):
            w.write(np.zeros((240, 320, 3), np.uint8))
        w.release()
    except Exception as e:                      # noqa: BLE001
        print(f"[warmup] dummy video failed: {e}", flush=True)
        return
    env = _trigger_engine_warmup(video, wdir)
    print(f"[warmup] engine preload done in {time.time()-t0:.0f}s -> {env}",
          flush=True)


def _trigger_engine_warmup(video: str, wdir: str):
    from agent.loop import Agent
    agent = Agent(workdir=wdir, on_event=lambda e: None, cases_path=CASES_PATH)
    return agent.dispatch_fn("verify_target",
                             {"video": video, "query": "person",
                              "out": os.path.join(wdir, "vt"), "timeout": 600},
                             wdir)


def main(argv=None):
    ap = argparse.ArgumentParser(description="video-retrieval agent service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--watch", type=float, default=0,
                    help="内嵌演化周期(分钟; 0=禁用. serve.sh 默认传 30)")
    ap.add_argument("--warmup", action="store_true",
                    help="启动时后台预热引擎模型(消除首个任务的冷启动)")
    ap.add_argument("--cases", default=None, help="案例 JSONL 路径")
    ap.add_argument("--jobs", default=None, help="任务工作目录")
    args = ap.parse_args(argv)
    global CASES_PATH, JOBS_DIR
    if args.cases:
        CASES_PATH = args.cases
    if args.jobs:
        JOBS_DIR = args.jobs
        os.makedirs(JOBS_DIR, exist_ok=True)
    if args.watch:
        print(f"[watch] evolution cycle every {args.watch} min", flush=True)
        threading.Thread(target=_watch_loop, args=(args.watch,),
                         daemon=True).start()
    if args.warmup:
        print("[warmup] preloading engine models in background…", flush=True)
        threading.Thread(target=_warmup, daemon=True).start()
    import uvicorn
    print(f"[serve] http://{args.host}:{args.port}  (GUI: /, health: /health)",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
