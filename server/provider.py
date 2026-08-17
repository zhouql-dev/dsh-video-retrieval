#!/usr/bin/env python3
"""Unified VLM client — the cloud-edge seam (the Q1 融合落点).

This is the Phase-1 evolution of ``mvp/glmv.py``: the same four public
functions with identical signatures and return contracts, but the single
OpenAI-compatible transport now routes between two providers via
``VLM_PROVIDER=glm|qwen``:

  * **glm** — ZhipuAI bigmodel.cn (``glm-4v-flash`` / ``glm-5.1``). This is the
    proven edge baseline from ``mvp/glmv.py``; behavior is preserved byte-for-byte.
  * **qwen** — Alibaba DashScope OpenAI-compatible endpoint (``qwen-vl-max`` /
    ``qwen-plus``). Unlocks the Q2/Q3 channel (native video, OCR, temporal
    grounding).

Routing + degradation contract (unchanged from the field-tested design):
  * The *active* provider (``VLM_PROVIDER``, default ``glm``) is tried first.
  * On any failure — no key, non-200, quota (429), timeout, JSON/regex miss —
    the call transparently falls back to the *other* provider if it has a key.
  * If neither provider can answer, returns ``None`` (or ``{}``-shaped dict for
    the JSON functions), so the local-only baseline never hard-depends on the
    cloud. This is the env-gated silent degradation glmv.py already used; we
    only widen it from one provider to two.

Both providers speak OpenAI-compatible ``/chat/completions`` and accept the same
``image_url`` data-URL content items, so message bodies are provider-agnostic.
"""
from __future__ import annotations
import base64
import json
import os
import re
from typing import Optional

import requests

# --------------------------------------------------------------------------- #
# Provider configuration
# --------------------------------------------------------------------------- #

# GLM (ZhipuAI) ----------------------------------------------------------------
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
# glm-4v-flash = free tier (no balance needed); paid glm-5v-turbo / glm-4v-plus
# 429 "余额不足" on many accounts — override with GLM_VISION_MODEL.
GLM_VISION_MODEL = os.environ.get("GLM_VISION_MODEL", "glm-4v-flash")
GLM_TEXT_MODEL = os.environ.get("GLM_TEXT_MODEL", "glm-5.1")

# Qwen (DashScope, OpenAI-compatible) -----------------------------------------
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# qwen-vl-max: flagship, native video via video_url; qwen-vl-ocr for OCR-only.
QWEN_VISION_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max")
QWEN_TEXT_MODEL = os.environ.get("QWEN_TEXT_MODEL", "qwen-plus")


def _glm_key() -> Optional[str]:
    return os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")


def _qwen_key() -> Optional[str]:
    return os.environ.get("DASHSCOPE_API_KEY")


# Active provider selection. ``auto`` = GLM if its key is set, else Qwen.
VLM_PROVIDER = (os.environ.get("VLM_PROVIDER") or "glm").lower()

# Last provider that actually answered a call (for metrics/logging in run.py).
_LAST_PROVIDER: Optional[str] = None

# E0 runtime-config seam: fusion/config/prompts.json may override the
# refine/verify/match prompt templates (keys: refine, verify, match; a
# template may use ``{query}``). Missing/malformed config degrades silently
# to the inline texts below — zero behavior change when absent.
try:
    from config import prompt_overrides  # type: ignore
    _PROMPT_OVERRIDES = prompt_overrides() or {}
except Exception:                       # noqa: BLE001 — degrade to inline texts
    _PROMPT_OVERRIDES = {}


def _tpl(key: str, default: str) -> str:
    """The (possibly overridden) prompt template for ``key``."""
    return _PROMPT_OVERRIDES.get(key, default)


def _fmt(tpl: str, **kw) -> str:
    """Format a template; templates without the placeholders are used as-is."""
    try:
        return tpl.format(**kw)
    except (KeyError, IndexError):
        return tpl


def active_provider() -> str:
    """The provider ``chat`` tries first (``VLM_PROVIDER``)."""
    return VLM_PROVIDER


def has_glm() -> bool:
    return bool(_glm_key())


def has_qwen() -> bool:
    return bool(_qwen_key())


def provider_status() -> dict:
    """Snapshot of provider availability — cheap, no network."""
    return {"active": VLM_PROVIDER, "glm": has_glm(), "qwen": has_qwen(),
            "last": _LAST_PROVIDER}


def last_provider() -> Optional[str]:
    """Provider that answered the most recent successful ``chat`` call."""
    return _LAST_PROVIDER


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

def _cfg(name: str):
    """Return (key, base_url, vision_model, text_model) for a provider, or None
    if the provider has no key configured (env-gated)."""
    if name == "qwen":
        k = _qwen_key()
        return (k, QWEN_BASE_URL, QWEN_VISION_MODEL, QWEN_TEXT_MODEL) if k else None
    # glm (default + fallback spelling tolerant)
    k = _glm_key()
    return (k, GLM_BASE_URL, GLM_VISION_MODEL, GLM_TEXT_MODEL) if k else None


def _post(base_url: str, key: str, model: str, messages, max_tokens: int,
          timeout: int, temperature: float) -> Optional[str]:
    """One OpenAI-compatible POST. Returns the assistant content string, or None
    on any failure (non-200, network, missing content). Never raises."""
    try:
        r = requests.post(
            base_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps({"model": model, "messages": messages,
                             "max_tokens": max_tokens, "temperature": temperature}),
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _order() -> list[str]:
    """Provider try-order: active first, then the other (if available)."""
    act = VLM_PROVIDER if VLM_PROVIDER in ("glm", "qwen") else "glm"
    other = "qwen" if act == "glm" else "glm"
    order = [act]
    if other not in order:
        order.append(other)
    return order


def chat(messages, max_tokens: int = 2048, timeout: int = 60,
         model: Optional[str] = None, kind: str = "vision",
         temperature: float = 0.1) -> Optional[str]:
    """Send a chat completion through the active provider, with one-shot
    transparent fallback to the other provider on any failure.

    ``kind`` ("vision"|"text") selects the provider's default model family when
    ``model`` is None; an explicit ``model`` always wins (so the existing
    glmv.py call sites that pass a GLM model name keep working when the active
    provider is glm, and are auto-substituted to the Qwen family when it is
    qwen). Returns the content string or None (both providers unavailable /
    failed) — env-gated, never raises.
    """
    global _LAST_PROVIDER
    for name in _order():
        cfg = _cfg(name)
        if not cfg:
            continue
        key, base, vis_m, txt_m = cfg
        # Resolve model: explicit override, else per-provider default for kind.
        if model:
            m = _substitute_model(model, name) if not _model_known(model, name) else model
        else:
            m = vis_m if kind == "vision" else txt_m
        out = _post(base, key, m, messages, max_tokens, timeout, temperature)
        if out:
            _LAST_PROVIDER = name
            return out
    return None


def _model_known(model: str, provider: str) -> bool:
    """True if ``model`` looks like a model id native to ``provider`` (avoids
    sending a GLM name like 'glm-5.1' to DashScope)."""
    g = model.lower().startswith(("glm", "charglm", "cogvlm"))
    q = model.lower().startswith(("qwen", "qwq"))
    if provider == "glm":
        return g or not q
    return q or not g


def qwen_chat(messages, max_tokens: int = 4096, timeout: int = 120,
              model: Optional[str] = None, temperature: float = 0.1) -> Optional[str]:
    """Force the Qwen (DashScope) provider — no GLM fallback. Used by
    ``ground_candidates`` (native ``video_url`` capability, which GLM's
    chat endpoint lacks). Returns content or None (env-gated: no
    DASHSCOPE_API_KEY -> None, so the caller falls back to a full scan)."""
    global _LAST_PROVIDER
    cfg = _cfg("qwen")
    if not cfg:
        return None
    key, base, vis_m, _ = cfg
    out = _post(base, key, model or vis_m, messages, max_tokens, timeout, temperature)
    if out:
        _LAST_PROVIDER = "qwen"
    return out


def _substitute_model(model: str, provider: str) -> str:
    """Map a model requested for the *other* family onto this provider's
    equivalent family (text->text, vision->vision). Keeps cross-provider
    fallback coherent when an explicit model name was passed."""
    # Heuristic: text-family requests (refine_query uses glm-5.1) -> text model.
    textish = any(t in model.lower() for t in ("5.1", "plus", "max-text", "turbo"))
    if provider == "qwen":
        return QWEN_TEXT_MODEL if textish else QWEN_VISION_MODEL
    return GLM_TEXT_MODEL if textish else GLM_VISION_MODEL


# Backwards-compat alias so existing call sites (``from provider import _key``)
# and glmv.py imports both work inside the fusion package.
def _key() -> Optional[str]:
    return _glm_key() or _qwen_key()


# --------------------------------------------------------------------------- #
# The four glmv.py contracts (signatures + return shapes preserved)
# --------------------------------------------------------------------------- #

def refine_query(query: str) -> Optional[str]:
    """Free-form description -> short noun phrase for open-vocab detection.
    ``str -> str | None``."""
    tpl = _tpl("refine",
               "Extract the single most salient target object noun phrase (<=4 words, English) "
               "that a vision detector should localize for this query. Reply ONLY the phrase.\n"
               "Query: {query}")
    out = chat([{"role": "user", "content": _fmt(tpl, query=query)}],
               model="glm-5.1", kind="text")
    if out:
        out = re.sub(r'["\'.\n]', "", out).strip()
    return out or None


def disambiguate(query: str, crops_bgr: list) -> Optional[int]:
    """Pick the crop index (0-based) best matching query, or None.
    ``list[ndarray] -> int | None``."""
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
    out = chat([{"role": "user", "content": content}], max_tokens=2048)
    if out:
        m = re.search(r"\d+", out)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(crops_bgr):
                return idx
    return None


def verify_target(query: str, crop_bgr, max_tokens: int = 2048) -> Optional[dict]:
    """Verify whether a single crop matches a *specific* target description and
    read any identifying text (e.g. a license plate). ``-> {match,read,note} |
    None``. The per-instance hit gate."""
    import cv2, json
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode()
    tpl = _tpl("verify",
               "这是监控画面中一个目标物体的裁剪图。判断它是否与以下目标描述相符：\"{query}\"。\n"
               "若目标是车辆/人员等带有可读标识（如车牌、衣着颜色）的物体，请尽量准确读出标识文字。\n"
               "只返回JSON：{\"match\": \"exact|partial|no|unclear\", \"read\": \"读到的标识或空\", \"note\": \"简短说明\"}")
    prompt = _fmt(tpl, query=query)
    out = chat([{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        max_tokens=max_tokens)
    if not out:
        return None
    return _parse_json(out, fallback={"match": "unclear", "read": "", "note": out[:200]})


def match_image(ref_bgr, crop_bgr, max_tokens: int = 2048) -> Optional[dict]:
    """Image-to-image identity verification (person_search VLM fallback).
    ``-> {match,conf,note} | None``. The §11.2 VLM 图对图终审 backend."""
    import cv2, json
    content = [{"type": "text", "text": _fmt(_tpl("match",
        "图1是一张参考人物图像(人脸或人形),图2是监控视频中的一张人物裁剪图。"
        "请判断图2中的人物是否与图1中的参考人物是同一个人。"
        "综合考虑人脸、体型、衣着、姿态等线索。"
        '只返回JSON:{"match":"exact|partial|no|unclear","conf":0到1的数字,"note":"简短说明"}' ))}]
    for tag, img in (("图1 参考", ref_bgr), ("图2 候选", crop_bgr)):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            continue
        content.append({"type": "text", "text": tag + ":"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"}})
    out = chat([{"role": "user", "content": content}], max_tokens=max_tokens)
    if not out:
        return None
    d = _parse_json(out, fallback={"match": "unclear", "conf": 0.0, "note": out[:200]})
    d.setdefault("conf", 0.0)
    d.setdefault("match", "unclear")
    return d


def _parse_json(text: str, fallback: dict) -> dict:
    """Extract the first {...} block from a model reply and json.loads it;
    return ``fallback`` (with the raw text folded into ``note``) on any miss."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return fallback
    try:
        return json.loads(m.group(0))
    except Exception:
        return fallback


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Smoke test: refine_query across both providers' availability.
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "a man in a red jacket carrying a black backpack"
    print("status:", provider_status())
    print("refine_query ->", refine_query(q))
