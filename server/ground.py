#!/usr/bin/env python3
"""Cloud candidate generation — the wide mouth of the funnel.

``ground_candidates(video, query)`` asks ``qwen-vl-max`` (DashScope, native
``video_url``) for the *temporal windows* where the target appears, and returns
them as coarse candidates ``[{start_s, end_s, score, reason}]``. The local
engines then verify *inside* those windows only — this is the report-1 design:
cloud-generates-candidates + local-verifies, instead of trusting the cloud's
"all yes" hallucination or the edge's over-matching embeddings.

Why it must degrade to empty (§3 / report-2):
  * report-2 (车牌 京Q1G728): the cloud grounding returned **empty** (it can't
    read a 70×27 plate), and the cloud OCR even mis-read 京 as 苏E7589. So a
    precise-text query MUST NOT be locked to grounding's output — an empty
    result triggers a **full-video local OCR scan** instead (see run.py).
  * Any failure (no DASHSCOPE_API_KEY, local file too large to base64, parse
    miss, network) returns ``[]``. Callers treat ``[]`` as "no cloud priors;
    scan the whole video" — never as "target absent".

Local-file input: DashScope's OpenAI-compatible mode does NOT accept a local
path or file-id, only an ``http(s)`` URL or a base64 data URL. A 15-min 1080p
clip is too large to base64 in one POST, so for big local files we return []
and expect the caller to pass a URL (or pre-upload to OSS); this honest cap is
``GROUND_VIDEO_MAX_MB`` (default 64). http(s) URLs pass through untouched.
"""
from __future__ import annotations
import base64
import json
import os
import re
from typing import Optional

# Max local-file size we'll base64 into a single request. Files above this are
# TRANSCODED down (ffmpeg: fit ~448², fps cap, CRF25) — the qwen-mm-plugins
# strategy that fits ~9 min of surveillance into a 10MB inline body. Only when
# even the transcode exceeds the budget does the caller fall back to a full
# local scan. Override with GROUND_VIDEO_MAX_MB. http(s) URLs bypass it.
GROUND_VIDEO_MAX_MB = float(os.environ.get("GROUND_VIDEO_MAX_MB", "64"))
GROUND_MAX_B64_BYTES = int(os.environ.get("GROUND_MAX_B64_BYTES", str(int(7.5 * 1024 * 1024))))
DEFAULT_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max")
DEFAULT_FPS = float(os.environ.get("QWEN_GROUND_FPS", "1"))   # frames/sec sampled by DashScope


def _transcode_for_upload(video: str, out_dir: str,
                          max_bytes: int = GROUND_MAX_B64_BYTES,
                          fps: float = DEFAULT_FPS,
                          max_side: int = 448, crf: int = 25) -> Optional[str]:
    """Transcode a local video into an inline-able mp4 (fit ~448², fps cap,
    CRF) whose base64 stays under ``max_bytes``. Returns the tmp path, or None
    if ffmpeg is missing or the budget cannot be met (caller falls back)."""
    import shutil
    import subprocess
    import tempfile
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ground_transcode.mp4")
    for crf_v, fps_v in ((crf, fps), (crf + 6, max(0.2, fps / 2)), (crf + 12, max(0.2, fps / 4))):
        try:
            r = subprocess.run(
                [ffmpeg, "-y", "-i", video, "-vf",
                 f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease,"
                 f"fps={fps_v}", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf_v),
                 "-c:a", "aac", "-b:a", "32k", "-ac", "1", out],
                capture_output=True, timeout=900)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and os.path.exists(out):
            if os.path.getsize(out) * 4 // 3 <= max_bytes:   # base64 inflates ~1/3
                return out
    return None


def _video_url(video: str, max_mb: float) -> tuple[Optional[str], Optional[str]]:
    """Resolve ``video`` to a ``video_url`` value for DashScope. Returns
    (url_value, error_reason); on error url_value is None and the caller
    returns []. http(s) passes through; local files up to GROUND_VIDEO_MAX_MB
    inline directly, larger ones are transcoded down first (ffmpeg), and only
    when that also fails do we report the error (caller scans the full video)."""
    if str(video).startswith(("http://", "https://")):
        return video, None
    p = os.path.abspath(video)
    if not os.path.exists(p):
        return None, f"video not found: {video}"
    mb = os.path.getsize(p) / 1e6
    if mb <= max_mb:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:video/mp4;base64,{b64}", None
    # large local file -> transcode to fit the inline budget (qwen-mm-plugins
    # strategy; ~9 min of surveillance fits at 448²/fps1/CRF25)
    tmp = _transcode_for_upload(p, os.path.join(os.path.dirname(p), ".ground_tmp")
                                if os.access(os.path.dirname(p), os.W_OK) else "/tmp/ground_tmp")
    if tmp is None:
        return None, (f"local file {mb:.0f}MB > cap and transcode failed; "
                      f"pass an http(s) URL or raise GROUND_VIDEO_MAX_MB")
    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:video/mp4;base64,{b64}", None


def _ts_to_seconds(ts: str) -> Optional[float]:
    """'01:23:45' / '23:45' / '45.0' -> seconds."""
    if ts is None:
        return None
    t = str(ts).strip()
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", t)
    if m:
        h = int(m.group(1) or 0); mi = int(m.group(2)); s = float(m.group(3))
        return h * 3600 + mi * 60 + s
    try:
        return float(t)
    except ValueError:
        return None


def _parse_segments(text: str, query: str) -> list[dict]:
    """Pull timestamped segments out of a model reply. Accepts either a JSON
    object ``{"events":[{start_time,end_time,event,confidence?}]}`` (the shape
    the DashScope doc recommends) or a bare ``[{...}]`` list."""
    if not text:
        return []
    # Find the outermost {...} or [...].
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
    except Exception:
        return []
    events = data.get("events", data) if isinstance(data, dict) else data
    if not isinstance(events, list):
        return []
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        s = _ts_to_seconds(e.get("start_time") or e.get("start") or e.get("start_s"))
        en = _ts_to_seconds(e.get("end_time") or e.get("end") or e.get("end_s"))
        if s is None:
            continue
        if en is None or en < s:
            en = s + 2.0                      # default 2s window if end missing
        conf = e.get("confidence") or e.get("score") or e.get("conf")
        try:
            score = float(conf)
        except (TypeError, ValueError):
            score = 0.6                       # flat default; local verifier re-scores
        reason = str(e.get("event") or e.get("reason") or e.get("description") or "")
        out.append({"start_s": round(s, 2), "end_s": round(en, 2),
                    "score": round(max(0.0, min(1.0, score)), 3), "reason": reason})
    return out


def _sanitize_segments(segs: list[dict], duration_s: Optional[float]) -> list[dict]:
    """Clamp model-returned timestamps onto the real time axis. qwen-vl-max's
    chat grounding has been observed returning frame numbers or milliseconds as
    'seconds' (e.g. 33180s on a 902s clip) — try the plausible unit fixes
    (÷fps, ÷1000) and drop segments that still fall outside [0, duration]."""
    if duration_s is None:
        return segs
    out = []
    for s in segs:
        st, en = s["start_s"], s["end_s"]
        for fix in (lambda v: v, lambda v: v / 1000.0, lambda v: v / 25.0):
            a, b = fix(st), fix(en)
            if 0 <= a < duration_s and 0 < b <= duration_s and b > a:
                st, en = a, b
                break
        else:
            continue                        # drop: never lands on the real axis
        if en - st < 1.0:
            continue                        # drop sub-second noise windows
        out.append({**s, "start_s": round(st, 2), "end_s": round(min(en, duration_s), 2)})
    return out


def _video_duration_s(video: str) -> Optional[float]:
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if total > 0:
            return total / fps
    except Exception:
        pass
    return None


def ground_candidates(video: str, query: str, *, model: str = DEFAULT_MODEL,
                      fps: float = DEFAULT_FPS, max_mb: float = GROUND_VIDEO_MAX_MB,
                      max_tokens: int = 2048, timeout: int = 180) -> list[dict]:
    """Cloud temporal grounding via qwen-vl-max. Returns coarse candidate
    windows ``[{start_s, end_s, score, reason}]``; **empty list on any failure
    or absence — the caller must then fall back to a full scan.**

    ``fps`` is DashScope's frame-sampling rate for the video (higher = more
    thorough but slower/costlier). This is the *only* cloud call in the
    semantic branch; everything downstream is local verification.
    """
    from provider import qwen_chat, has_qwen
    if not has_qwen():
        print("[ground] DASHSCOPE_API_KEY unset -> no cloud grounding; "
              "caller should fall back to a full scan")
        return []

    url, err = _video_url(video, max_mb)
    if url is None:
        print(f"[ground] cannot send video to DashScope ({err}); falling back")
        return []

    duration = _video_duration_s(video)
    prompt = (
        "你在看一段监控视频。请定位画面中与下列目标描述相符的时间段，"
        "只返回真正出现目标的区间，若整段视频没有出现则返回空列表。\n"
        f"视频总长约 {duration:.0f} 秒" + ("。" if duration else "") +
        "时间戳必须用视频内的相对秒数(0 到视频总长)，不是帧号也不是毫秒。\n"
        "只返回JSON，格式：{\"events\":[{\"start_time\":\"HH:MM:SS\","
        "\"end_time\":\"HH:MM:SS\",\"confidence\":0到1的数字,\"event\":\"简述\"}]}\n"
        f"目标描述：{query}")
    messages = [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": url}, "fps": fps},
        {"type": "text", "text": prompt},
    ]}]
    print(f"[ground] asking {model} to ground {query!r} (fps={fps}, dur≈{duration and round(duration) or '?'}s) ...")
    out = qwen_chat(messages, max_tokens=max_tokens, timeout=timeout, model=model)
    if not out:
        print("[ground] no reply from Qwen -> falling back")
        return []
    segs = _parse_segments(out, query)
    segs = _sanitize_segments(segs, duration)
    print(f"[ground] {len(segs)} candidate window(s): "
          + (", ".join(f"[{s['start_s']}-{s['end_s']}s]" for s in segs) or "(empty)"))
    return segs


def windows_to_frames(windows: list[dict], fps: float, total_frames: int,
                      pad_s: float = 1.0) -> list[tuple[int, int]]:
    """Turn grounded ``[{start_s,end_s}]`` into ``[(lo_frame, hi_frame)]``
    scan ranges, clamped to the video and padded by ``pad_s``. Empty windows
    -> single full-range tuple ``[(0, total_frames)]`` (the fallback scan)."""
    if not windows:
        return [(0, max(0, total_frames - 1))]
    ranges = []
    for w in windows:
        lo = max(0, int((w["start_s"] - pad_s) * fps))
        hi = min(total_frames - 1, int((w["end_s"] + pad_s) * fps))
        if hi >= lo:
            ranges.append((lo, hi))
    return ranges or [(0, max(0, total_frames - 1))]


if __name__ == "__main__":
    # Self-test of the pure parsers (no network / no key needed).
    fake_reply = ('{"events":[{"start_time":"00:09:40","end_time":"00:09:50",'
                  '"confidence":0.9,"event":"穿粉色外套女性经过"}]}')
    print("parse:", _parse_segments(fake_reply, "粉色外套女性"))
    print("windows->frames:",
          windows_to_frames([{"start_s": 580, "end_s": 600}], fps=25, total_frames=22566))
    print("empty->frames:", windows_to_frames([], fps=25, total_frames=22566))
