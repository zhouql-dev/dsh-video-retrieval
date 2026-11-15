#!/usr/bin/env python3
"""Task router — the key loop the two eval reports forced (§3 of the guide).

Why a router: the two reports showed the cloud and the edge win on *different*
query types, and forcing one path for both is strictly worse:

  * **precise_text** (a license plate ``京Q1G728``, an ID number): the cloud
    grounding model can't read a 70×27 plate (report-2: returns empty; cloud
    OCR even mis-reads 京 as 苏E7589). The winner is local OCR + a character
    confusable table (``fast_plate_scan.py``) — but the cloud's emptiness must
    *not* lock the search; it must trigger a full-video OCR scan instead.

  * **semantic** (a person "in a pink coat and white pants"): local embeddings
    over-match (OSNet 0.55 → 16 candidates, 15 false positives) while the cloud
    grounding model confidently *hallucinates* (report-1: 0.95 on wrong pink/red
    person). The winner is cloud-generates-candidates + local-verifies +
    multi-signal consensus — the §11.2 path.

``classify_query`` is therefore the first stage of ``run.py``: it decides which
of the two branches executes. It is pure regex/heuristic — no network, so it is
unit-testable without keys.

Contract
--------
``classify_query(q) -> {"branch": "semantic"|"unsupported", "target": str|None,
                        "query": str, "kind": str?, "msg": str?}``
  * W1 预算红线：precise_text 分支已整体移除（实测 15 分钟视频全片 OCR 双遍
    >130min，超预算 8 倍以上）。车牌/证件号查询不再启动任何扫描，返回
    ``unsupported`` + 引导文案，由调用方（agent loop / GUI）直接给出建议。
  * ``target``: 命中的标识串（如 ``Q1G728``），仅用于引导展示。
"""
from __future__ import annotations
import re

# Chinese plate alphanumerics, with or without the 京 prefix, allowing spaces.
# A plate is a province letter + 5-6 alnum, e.g. 京Q1G728 / 京P3LD03.
_PLATE_RE = re.compile(r"京\s*([A-Z][A-Z0-9]{4,6})")
# Bare plate-like token (no province), e.g. "Q1G728" — 5-7 alnum with a leading
# letter and at least one digit, to avoid matching ordinary English words.
_BARE_PLATE_RE = re.compile(r"\b([A-Z][A-Z0-9]{4,6}\d[A-Z0-9]*)\b|\b([A-Z]\d[A-Z0-9]{3,5})\b")
# 18-digit Chinese ID card number (last char may be X).
_IDCARD_RE = re.compile(r"\b\d{17}[\dXx]\b")
# Recognized keywords that strongly imply a precise-text lookup.
_PRECISE_KW = ("车牌", "牌号", "号牌", "身份证", "证件号", "plate", "license")


def extract_plate(query: str) -> str | None:
    """Pull the plate alphanumerics out of a natural-language query.
    '车牌号 京Q1G728 的小轿车' -> 'Q1G728'; 'white plate 京P3LD03' -> 'P3LD03'.
    Mirrors verify_target.extract_plate_target so run.py and the OCR engine
    agree on the target string."""
    q = query.upper().replace("京", " 京")
    m = _PLATE_RE.search(query) or _PLATE_RE.search(q)
    if m:
        return m.group(1)
    m = _BARE_PLATE_RE.search(q)
    return m.group(1) or m.group(2) if m else None


_UNSUPPORTED_MSG = ("全片车牌/证件号检索已按时长预算移除(实测超出红线 8 倍以上)。"
                    "请改用外观描述,例如「白色皮卡车」「黑色轿车」;"
                    "命中片段内可人工读取车牌。")


def classify_query(query: str) -> dict:
    """Route a free-form query. Plate / ID-number / precise-keyword queries
    return ``branch="unsupported"`` with guidance text (W1: the full-video
    OCR branch was cut for blowing the duration budget 8x)."""
    if not query or not query.strip():
        return {"branch": "semantic", "target": None, "query": query or ""}

    plate = extract_plate(query)
    idm = _IDCARD_RE.search(query)
    ql = query.lower()
    kw = any(k in ql for k in ("plate", "license")) or any(k in query for k in _PRECISE_KW)

    if plate or idm or kw:
        target = plate or (idm.group(0).upper() if idm else None)
        return {"branch": "unsupported", "target": target, "query": query,
                "kind": "id" if (idm and not plate) else "plate",
                "msg": _UNSUPPORTED_MSG}

    return {"branch": "semantic", "target": None, "query": query}


if __name__ == "__main__":
    import sys, json
    for q in (sys.argv[1:] or [
        "车牌号 京Q1G728 的小轿车",
        "a woman in a pink coat and white pants",
        "white plate 京P3LD03 sedan",
        "身份证 110101199003071234 的男子",
        "a red bicycle parked near the gate",
        "白色轿车",
    ]):
        print(f"{q!r:55} -> {json.dumps(classify_query(q), ensure_ascii=False)}")
