#!/usr/bin/env python3
"""Hard-case record — the loop-closer between the agent layer and the
evolution layer.

The agent's 临场发挥 is only as valuable as the data it leaves behind. This
module defines the recording protocol (schema: case.schema.json), a JSONL
writer/loader, and the converter that turns *human-confirmed* cases into the
dataset contract of dataset.py — so gepa/Optuna can later distill the agent's
improvised solutions into rules/thresholds that get baked back into the
deterministic core. 闭环: 疑难 → 记录 → 人工确认 → 数据集 → 演化 → 核心变强。

Dependency-free (stdlib only) so agents can record cases in any environment.
"""
from __future__ import annotations
import json
import os
from datetime import datetime

_REQUIRED = ["id", "ts", "query", "query_type", "inputs", "plan", "attempts", "outcome"]
_QUERY_TYPES = {"composite", "insufficient", "environment", "reference_quality",
                "combo_retrieval", "novel_charset", "routine"}
_OUTCOMES = {"resolved", "unresolved", "user_input", "escalated"}

DEFAULT_CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "cases.jsonl")


def new_case_id() -> str:
    return f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def validate(case: dict) -> list[str]:
    """Return the list of problems (empty = valid). Lightweight structural
    check — no jsonschema dependency; case.schema.json is the formal spec."""
    problems = []
    for k in _REQUIRED:
        if k not in case:
            problems.append(f"missing required field: {k}")
    if "query_type" in case and case["query_type"] not in _QUERY_TYPES:
        problems.append(f"query_type must be one of {sorted(_QUERY_TYPES)}")
    if "outcome" in case and case["outcome"] not in _OUTCOMES:
        problems.append(f"outcome must be one of {sorted(_OUTCOMES)}")
    if "attempts" in case and not isinstance(case["attempts"], list):
        problems.append("attempts must be a list")
    return problems


def record_case(case: dict, path: str = DEFAULT_CASES_PATH) -> tuple[bool, str]:
    """Validate and append one case to the JSONL log. Returns (ok, msg)."""
    problems = validate(case)
    if problems:
        return False, "; ".join(problems)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")
    return True, f"recorded {case.get('id')}"


def iter_cases(path: str = DEFAULT_CASES_PATH):
    """Yield every case in the log (skipping unparsable lines)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def confirmed_cases(path: str = DEFAULT_CASES_PATH) -> list[dict]:
    """Only cases a human has confirmed with ground truth — the ones eligible
    to become training data for the evolution layer."""
    return [c for c in iter_cases(path)
            if c.get("human_confirmed") is True and c.get("gt")]


# --------------------------------------------------------------------------- #
# Extraction into the dataset.py contract
# --------------------------------------------------------------------------- #

def extract_candidates_from_case(case: dict) -> list[dict]:
    """Convert one confirmed case into person-style dataset candidates.

    Pulls the rescoring attempt's scored rows (each carries per-signal results)
    and labels each candidate ``gt=True`` iff its t_s falls inside a confirmed
    hit interval. Mirrors the dataset.py person contract:
        {id, t_s, gt, results:{hue/temporal/vlm:{score,raw,evidence}}}"""
    gt = case.get("gt") or {}
    intervals = gt.get("hit_intervals") or []
    cands = []
    for a in case.get("attempts", []):
        env = a.get("envelope") or {}
        scored = env.get("outputs", {}).get("scored") or []
        for c in scored:
            t = c.get("t_s")
            is_hit = any(lo <= float(t) <= hi for lo, hi in intervals) if t is not None else False
            row = {"id": f"{case['id']}:{c.get('id')}", "t_s": t,
                   "gt": is_hit, "results": c.get("results", {})}
            if row["results"]:
                cands.append(row)
    return cands


def extract_ocr_rows_from_case(case: dict) -> list[dict]:
    """Convert one confirmed case into plate-OCR dataset rows.

    Labels each OCR read by the confirmed positive reads (gt.positive_reads)
    if given, else by t_s inside gt.hit_intervals. Reads come from the
    fast_plate_scan attempt's all_ocr.json. Mirrors dataset.py plate contract:
        {ocr, t_s, frame, gt}"""
    gt = case.get("gt") or {}
    positive_reads = {s.upper().replace(" ", "") for s in (gt.get("positive_reads") or [])}
    intervals = gt.get("hit_intervals") or []
    rows = []
    for a in case.get("attempts", []):
        env = a.get("envelope") or {}
        all_ocr_path = (env.get("outputs") or {}).get("all_ocr.json")
        if not all_ocr_path or not os.path.exists(all_ocr_path):
            continue
        for r in json.load(open(all_ocr_path)):
            txt = (r.get("ocr") or "").upper().replace(" ", "")
            if positive_reads:
                label = txt in positive_reads
            else:
                t = r.get("t_s")
                label = any(lo <= float(t) <= hi for lo, hi in intervals) if t is not None else False
            rows.append({"ocr": txt, "t_s": r.get("t_s"), "frame": r.get("frame"), "gt": label})
    return rows


def cases_to_dataset(path: str = DEFAULT_CASES_PATH) -> dict:
    """Merge all confirmed cases into the two dataset shapes the evolution
    layer consumes: {"candidates": [...], "rows": [...]}. This is the bridge
    that closes the 智能体→演化 循环."""
    candidates, rows = [], []
    for c in confirmed_cases(path):
        candidates += extract_candidates_from_case(c)
        rows += extract_ocr_rows_from_case(c)
    return {"candidates": candidates, "rows": rows}


if __name__ == "__main__":
    # self-test: fabricate a confirmed hard case, record it, convert it back.
    case = {
        "id": new_case_id(), "ts": datetime.now().isoformat(timespec="seconds"),
        "query": "找 3 点蓝门附近穿红衣的人", "query_type": "composite",
        "inputs": {"video": "/tmp/clip.mp4", "ref": None, "provider_status": {"glm": False}},
        "plan": ["定位蓝门锚点", "锁定时间窗", "窗内找红衣人", "身份核验"],
        "attempts": [
            {"tool": "rescoring", "params": {"agree": 2},
             "note": "候选 t=438 色相+曲线双 fire",
             "envelope": {"outputs": {"scored": [
                 {"id": "f438", "t_s": 438.0,
                  "results": {"hue": {"score": 0.9, "raw": {"dhue": 12.0}, "evidence": ""},
                              "temporal": {"score": 0.7, "raw": {"peak": 0.83, "bellness": 0.5, "span_s": 2.0, "n": 5}, "evidence": ""}}},
                 {"id": "f120", "t_s": 120.0,
                  "results": {"hue": {"score": 0.1, "raw": {"dhue": 95.0}, "evidence": ""},
                              "temporal": {"score": 0.4, "raw": {"peak": 0.62, "bellness": 0.1, "span_s": 0.0, "n": 3}, "evidence": ""}}}]}}},
        ],
        "outcome": "resolved", "agent_reasoning": "拆步后两信号一致锁定 t=438",
        "human_confirmed": True,
        "gt": {"hit_intervals": [[437.0, 439.0]]},
    }
    ok, msg = record_case(case)
    print(msg)
    ds = cases_to_dataset()
    print("converted candidates:", len(ds["candidates"]),
          "| positives:", sum(c["gt"] for c in ds["candidates"]))
    print("sample:", json.dumps(ds["candidates"][0], ensure_ascii=False)[:180], "...")
