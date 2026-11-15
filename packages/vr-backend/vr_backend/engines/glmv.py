#!/usr/bin/env python3
"""GLM-5V-Turbo client (OpenAI-compatible REST via requests, no extra deps).

Two roles in the edge pipeline (the "API semantic" half of PDF Solution-4):
  1) refine_query()  — turn a free-form natural-language description into a
     concise object class phrase that YOLO-World can ground well.
  2) disambiguate()  — given several candidate crops from a frame, return the
     index of the one the query refers to (seed selection).

All calls are env-gated: silent no-op if ZHIPUAI_API_KEY is unset or the network
fails, so the local-only baseline never hard-depends on the API.
"""
from __future__ import annotations
import base64
import json
import os
import re
from typing import Optional

import requests

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
# Default to glm-4v-flash (free tier, no balance needed) — paid models
# (glm-5v-turbo / glm-4v-plus / glm-4.5v) return 429 "余额不足" on many accounts.
# Override with GLM_VISION_MODEL=glm-5v-turbo for stronger reasoning if your quota allows.
MODEL = os.environ.get("GLM_VISION_MODEL", "glm-4v-flash")


def _key() -> Optional[str]:
    return os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")


def _chat(messages, max_tokens=2048, timeout=60, model=None):
    # GLM-5 family are reasoning models: completion_tokens include reasoning_tokens,
    # so we need a generous budget or `content` comes back empty.
    key = _key()
    if not key:
        return None
    model = model or MODEL
    try:
        r = requests.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                             "temperature": 0.1}),
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def refine_query(query: str) -> Optional[str]:
    """Free-form description -> short noun phrase for open-vocab detection."""
    out = _chat([{"role": "user", "content":
        f"Extract the single most salient target object noun phrase (<=4 words, English) "
        f"that a vision detector should localize for this query. Reply ONLY the phrase.\n"
        f"Query: {query}"}], model="glm-5.1")
    if out:
        out = re.sub(r'["\'.\n]', "", out).strip()
    return out or None


def disambiguate(query: str, crops_bgr: list) -> Optional[int]:
    """Pick the crop index (0-based) best matching query, or None."""
    if not crops_bgr:
        return None
    content = [{"type": "text", "text":
        f"I show {len(crops_bgr)} candidate object crops from a surveillance frame, "
        f"numbered 1..{len(crops_bgr)}. Which ONE best matches: \"{query}\"? "
        f"Reply ONLY the integer index."}]
    import cv2
    for i, c in enumerate(crops_bgr, 1):
        ok, buf = cv2.imencode(".jpg", c, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode()
        content.append({"type": "text", "text": f"candidate {i}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    out = _chat([{"role": "user", "content": content}], max_tokens=2048)
    if out:
        m = re.search(r"\d+", out)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(crops_bgr):
                return idx
    return None


def verify_target(query: str, crop_bgr, max_tokens: int = 2048) -> Optional[dict]:
    """Verify whether a single crop matches a *specific* target description and
    read any identifying text (e.g. a license plate). Returns a dict
    {match, read, note} or None on failure.

    This is the per-instance check used by the two-stage retrieve->verify engine
    (verify_target.py), needed because the single-frame seed disambiguate() in
    locate.py cannot reliably identify a specific individual/plate across a clip
    with several similar objects.
    """
    import cv2, json
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode()
    prompt = (
        f"这是监控画面中一个目标物体的裁剪图。判断它是否与以下目标描述相符：\"{query}\"。\n"
        f"若目标是车辆/人员等带有可读标识（如车牌、衣着颜色）的物体，请尽量准确读出标识文字。\n"
        f"只返回JSON：{{\"match\": \"exact|partial|no|unclear\", \"read\": \"读到的标识或空\", \"note\": \"简短说明\"}}")
    out = _chat([{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        max_tokens=max_tokens)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {"match": "unclear", "read": "", "note": out[:200]}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"match": "unclear", "read": "", "note": out[:200]}


def match_image(ref_bgr, crop_bgr, max_tokens: int = 2048) -> Optional[dict]:
    """Image-to-image identity verification via GLM-5V-Turbo. Sends BOTH the
    reference subject image and a candidate crop, asks whether they are the same
    person/face. This is the VLM fallback for person_search when no embedding
    backend (insightface/OSNet) is available. Returns
    {match, conf, note} or None on failure.
    """
    import cv2, json
    content = [{"type": "text", "text":
        "图1是一张参考人物图像(人脸或人形),图2是监控视频中的一张人物裁剪图。"
        "请判断图2中的人物是否与图1中的参考人物是同一个人。"
        "综合考虑人脸、体型、衣着、姿态等线索。"
        '只返回JSON:{"match":"exact|partial|no|unclear","conf":0到1的数字,"note":"简短说明"}'}]
    for tag, img in (("图1 参考", ref_bgr), ("图2 候选", crop_bgr)):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            continue
        content.append({"type": "text", "text": tag + ":"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"}})
    out = _chat([{"role": "user", "content": content}], max_tokens=max_tokens)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {"match": "unclear", "conf": 0.0, "note": out[:200]}
    try:
        d = json.loads(m.group(0))
        d.setdefault("conf", 0.0); d.setdefault("match", "unclear")
        return d
    except Exception:
        return {"match": "unclear", "conf": 0.0, "note": out[:200]}


if __name__ == "__main__":
    # quick smoke test
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "a man in a red jacket carrying a black backpack"
    print("refine_query ->", refine_query(q))
    print("model:", MODEL, "key_set:", bool(_key()))
