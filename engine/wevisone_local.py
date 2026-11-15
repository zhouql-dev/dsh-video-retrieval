#!/usr/bin/env python3
"""Local WeVisOne-VL backend — the local semantic verifier (第四证人).

Wraps Tencent WeVisOne-VL (2B/4B, image-level unified perception VLM) as a
lazy, env-gated singleton for the fusion pipeline. Role in the architecture:

  * referring + reliable rejection (None for absent targets, WeRef 96.4%)
    becomes the LOCAL arbiter for candidate crops — no API key, frames never
    leave the machine, and it does not share the cloud VLM's "all yes"
    hallucination mode.
  * It is IMAGE-ONLY: it never replaces temporal grounding or the per-frame
    YOLO/BoT-SORT fast scan. It only ever sees deduped candidate crops
    (tens of images), which is where the speed/accuracy tradeoff lands.

Environment gates (all optional, all degrade silently):

  WEVISONE_ENABLED   auto|on|off   (default auto = on iff import+weights ok)
  WEVISONE_MODEL     HF id or path (default tencent/WeVisOne-VL-2B)
  WEVISONE_DEVICE    auto|mps|cuda:0|cpu (default auto)
  WEVISONE_MAX_PIXELS  crops are small; default 128*32*32 keeps tokens low

Failure contract: every public function returns ``None`` (or
``match="unavailable"`` at the signal layer) when the model is disabled,
missing, or errors — callers treat it as an abstaining signal, exactly like
an offline cloud provider. The local baseline never hard-depends on this.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

_ENABLED_ENV = "WEVISONE_ENABLED"
_MODEL_ENV = "WEVISONE_MODEL"
_DEVICE_ENV = "WEVISONE_DEVICE"
_MAXPIX_ENV = "WEVISONE_MAX_PIXELS"

_DEFAULT_MODEL = "tencent/WeVisOne-VL-2B"
_DEFAULT_MAX_PIXELS = 128 * 32 * 32      # crops are small; keep visual tokens low

# Offline-first weight resolution: prefer a local HF snapshot under
# <repo>/weights/hf-home (stage-0 download location) so the runtime never
# needs HF_ENDPOINT/cache env vars and works fully offline once downloaded.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_HOME = os.environ.get("HF_HOME") or os.path.join(_REPO_ROOT, "weights", "hf-home")
_HUB = os.path.join(_HF_HOME, "hub")


def _resolve_model_path(model_id: str) -> str:
    """Return a local snapshot dir if one exists, else the HF id (online)."""
    if os.path.isdir(model_id):
        return model_id
    safe = "models--" + model_id.replace("/", "--")
    snap_root = os.path.join(_HUB, safe, "snapshots")
    if os.path.isdir(snap_root):
        snaps = sorted(os.listdir(snap_root))
        if snaps:
            return os.path.join(snap_root, snaps[-1])
    return model_id

_lock = threading.Lock()
_state: dict = {"model": None, "failed": False, "reason": ""}


def _parse_referring(response: str) -> tuple[list, bool]:
    """Parse a referring response: ``(boxes_normalized, said_none)``.

    Field-verified format (2B, greedy): the model echoes the expression then
    either emits ``<box><x1><y1><x2><y2></box>`` (coords as <0>..<1000> tokens)
    or ``<box>none</box>`` for an absent target. The official
    ``parse_seg_boxes`` only handles the <OBJ>-suffixed matting variant, which
    bare referring does not produce — hence this local parser.
    """
    import re
    box_re = re.compile(r"<box>(.*?)(?:</box>|$)", re.S | re.I)
    coord_re = re.compile(r"<(\d{1,4})>")
    boxes: list[list[float]] = []
    said_none = False
    for body in box_re.findall(response or ""):
        if "none" in body.lower():
            said_none = True
            continue
        nums = [int(v) for v in coord_re.findall(body)]
        for i in range(0, len(nums) - 3, 4):
            x1, y1, x2, y2 = nums[i:i + 4]
            if all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                boxes.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return boxes, said_none


def enabled() -> bool:
    """auto|on|off gate. ``off`` hard-disables without touching imports."""
    mode = (os.environ.get(_ENABLED_ENV) or "auto").lower()
    if mode in ("off", "0", "false", "no"):
        return False
    return True          # auto == on (availability is decided at load time)


def available() -> bool:
    """True iff a usable local model can be loaded (cheap after first call)."""
    if not enabled():
        return False
    if _state["model"] is not None or _state["failed"]:
        return not _state["failed"]
    try:
        _get_model()
        return _state["model"] is not None
    except Exception:
        return False


def status() -> dict:
    """Diagnostics for reports/logs without forcing a load."""
    return {"enabled": enabled(), "loaded": _state["model"] is not None,
            "failed": _state["failed"], "reason": _state["reason"],
            "model": os.environ.get(_MODEL_ENV) or _DEFAULT_MODEL}


def _get_model():
    """Load the engine once (thread-safe). Any failure is cached as failed."""
    with _lock:
        if _state["model"] is not None or _state["failed"]:
            return _state["model"]
        model_id = os.environ.get(_MODEL_ENV) or _DEFAULT_MODEL
        device = os.environ.get(_DEVICE_ENV) or ""
        if not device:
            # The official _auto_device() only knows cuda/npu -> falls back to
            # CPU on Apple Silicon. MPS is ~5x faster for this workload.
            import torch
            device = "mps" if torch.backends.mps.is_available() else "auto"
        max_pixels = int(os.environ.get(_MAXPIX_ENV) or _DEFAULT_MAX_PIXELS)
        try:
            from wevisone_vl import WeVisOneVL
            model_path = _resolve_model_path(model_id)
            t0 = time.perf_counter()
            m = WeVisOneVL(
                model_path,
                backend="transformers",
                device=device,
                max_pixels=max_pixels,
                min_pixels=16 * 32 * 32,
            )
            _state["model"] = m
            print(f"[wevisone] loaded {model_id} (resolved: {model_path}) on {m.device} "
                  f"in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:                     # noqa: BLE001 — degrade
            _state["failed"] = True
            _state["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[wevisone] unavailable -> signals abstain ({_state['reason']})",
                  flush=True)
        return _state["model"]


def refer(query: str, crop_bgr, max_tokens: int = 256) -> Optional[dict]:
    """Referring localization on one crop — the local presence test.

    ``-> {"match": "exact"|"no"|"unclear", "boxes": [[x1,y1,x2,y2](normalized)],
          "latency_s": float, "response": str} | None``

    ``None`` (or match="unclear") means the caller should treat the signal as
    abstaining. "no" is a *decision* (the model rejected the target), not an
    abstention — that distinction is the whole point of the rejection feature.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    m = _get_model()
    if m is None:
        return None
    try:
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)
        t0 = time.perf_counter()
        result = m.predict(pil, task="referring", query=query,
                           decode_matting=False, max_new_tokens=max_tokens)
        response = str(result.response or "").strip()
        boxes, said_none = _parse_referring(response)
        latency = time.perf_counter() - t0
        if boxes:
            return {"match": "exact", "boxes": boxes,
                    "latency_s": round(latency, 3), "response": response[:200]}
        return {"match": "no" if said_none else "unclear", "boxes": [],
                "latency_s": round(latency, 3), "response": response[:200]}
    except Exception as exc:                         # noqa: BLE001 — degrade
        _state["failed"] = True
        _state["reason"] = f"{type(exc).__name__}: {exc}"
        return None


def refer_batch(query: str, crops_bgr, max_tokens: int = 256) -> list[Optional[dict]]:
    """Batched ``refer`` over many crops (single generation pass — much faster
    than calling refer() in a loop when MPS launch overhead dominates).
    Returns one result per input (None where the input crop was invalid)."""
    valid_idx = [i for i, c in enumerate(crops_bgr or [])
                 if c is not None and c.size]
    m = _get_model()
    if m is None or not valid_idx:
        return [None] * len(crops_bgr or [])
    try:
        import cv2
        from PIL import Image
        phrase = query.strip().rstrip(".!?")
        prompt = ("Locate the instances that match the following description: "
                  f"{phrase}.")
        pils = [Image.fromarray(cv2.cvtColor(crops_bgr[i], cv2.COLOR_BGR2RGB))
                for i in valid_idx]
        t0 = time.perf_counter()
        responses = m.generate_text_batch(pils, [prompt] * len(pils),
                                          max_new_tokens=max_tokens)
        latency = round((time.perf_counter() - t0) / max(1, len(pils)), 3)
        out: list[Optional[dict]] = [None] * len(crops_bgr)
        for i, response in zip(valid_idx, responses):
            response = str(response or "").strip()
            boxes, said_none = _parse_referring(response)
            if boxes:
                out[i] = {"match": "exact", "boxes": boxes,
                          "latency_s": latency, "response": response[:200]}
            else:
                out[i] = {"match": "no" if said_none else "unclear", "boxes": [],
                          "latency_s": latency, "response": response[:200]}
        return out
    except Exception as exc:                         # noqa: BLE001 — degrade
        _state["failed"] = True
        _state["reason"] = f"{type(exc).__name__}: {exc}"
        return [None] * len(crops_bgr or [])


if __name__ == "__main__":
    # Smoke: no image -> just report availability diagnostics.
    import numpy as np
    print(status())
    gray = np.full((64, 64, 3), 128, np.uint8)
    print("refer(gray, 'a red jacket') ->", refer("a red jacket", gray))
