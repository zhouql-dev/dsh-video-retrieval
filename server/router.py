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
``classify_query(q) -> {"branch": "semantic"|"precise_text", "target": str|None,
                        "query": str}``
  * ``target``: the normalized identifier string for precise_text (e.g.
    ``Q1G728``), else None. Forwarded to ``fast_plate_scan`` as ``--target``.
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


def classify_query(query: str) -> dict:
    """Route a free-form query to the semantic or precise_text branch.

    Returns ``{"branch", "target", "query"}``. ``branch == "precise_text"``
    iff a plate / ID number (or a strong precise keyword) is present; the
    extracted identifier is returned in ``target`` for the OCR scan."""
    if not query or not query.strip():
        return {"branch": "semantic", "target": None, "query": query or ""}

    plate = extract_plate(query)
    if plate:
        return {"branch": "precise_text", "target": plate, "query": query}

    if _IDCARD_RE.search(query):
        m = _IDCARD_RE.search(query)
        return {"branch": "precise_text", "target": m.group(0).upper(),
                "query": query}

    ql = query.lower()
    if any(k in ql for k in ("plate", "license")) or any(k in query for k in _PRECISE_KW):
        # precise keyword present but no extractable token — still precise_text,
        # target None lets run.py decide whether an OCR scan is meaningful.
        return {"branch": "precise_text", "target": None, "query": query}

    return {"branch": "semantic", "target": None, "query": query}


if __name__ == "__main__":
    import sys, json
    for q in (sys.argv[1:] or [
        "车牌号 京Q1G728 的小轿车",
        "a woman in a pink coat and white pants",
        "white plate 京P3LD03 sedan",
        "身份证 110101199003071234 的男子",
        "a red bicycle parked near the gate",
    ]):
        print(f"{q!r:55} -> {json.dumps(classify_query(q), ensure_ascii=False)}")
