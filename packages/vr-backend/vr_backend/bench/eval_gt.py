#!/usr/bin/env python3
"""Harness 测评的纯逻辑评分层(可单测,无引擎)。

对一次检索结果(agent / playbook / pipeline 三种模式都归一成同一形状的
result dict)按 GT 打分:

  * gt_hit       任一命中区间与 GT 区间重叠(±TOL 秒容差 / IoU≥0.5);
  * best_gap_s   最接近 GT 的命中区间与 GT 中心的距离(未命中=None);
  * honesty      输出不含"目标不存在"类谎报;
  * n_intervals / case_recorded / verdict 直接计数。

GT 契约: 每任务 {"id", "kind", "video", "query", "ref", "gt": [[s,e]...]}
gt=[] 表示"目标不在视频中"的反例任务——期望 honest no_hit。
"""
from __future__ import annotations

TOL_S = 2.0                       # 命中区间与 GT 的时间容差(秒)


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU of two time intervals; 0.0 when disjoint."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def hit_against_gt(interval: dict, gt: list) -> tuple[bool, float, float]:
    """(hit, iou, center_gap_s) of one result interval vs the GT set.
    Tolerance: the interval contains a GT center within TOL_S, or IoU>=0.5."""
    s, e = float(interval["start_s"]), float(interval["end_s"])
    best_iou, best_gap = 0.0, float("inf")
    for g in gt:
        gs, ge = float(g[0]), float(g[1])
        iou = interval_overlap((s, e), (gs, ge))
        if iou > best_iou:
            best_iou = iou
        center = (gs + ge) / 2
        gap = 0.0 if gs - TOL_S <= s <= ge + TOL_S or s <= center <= e \
            else min(abs(s - center), abs(e - center))
        if gap < best_gap:
            best_gap = gap
    contained = any((float(g[0]) - TOL_S) <= s <= (float(g[1]) + TOL_S)
                    for g in gt)
    hit = best_iou >= 0.5 or contained or best_gap <= TOL_S
    return hit, round(best_iou, 3), (round(best_gap, 1)
                                     if best_gap != float("inf") else None)


def score_result(result: dict, gt: list) -> dict:
    """Normalize one run's result into comparable metrics."""
    intervals = result.get("intervals") or []
    gt = gt or []
    hits = [hit_against_gt(iv, gt) for iv in intervals]
    any_hit = any(h for h, _, _ in hits)
    best_iou = max((iou for _, iou, _ in hits), default=0.0)
    best_gap = min((gap for _, _, gap in hits if gap is not None),
                   default=None)
    text = " ".join(str(result.get(k, "")) for k in ("reason", "note", "suggestion"))
    return {
        "verdict": result.get("verdict"),
        "n_intervals": len(intervals),
        "gt_hit": bool(any_hit),
        "best_iou": best_iou,
        "best_gap_s": best_gap,
        "honest": "不存在" not in text,
        "case_recorded": bool(result.get("case_id")),
        "tried": list(result.get("tried") or []),
        "wall_s": result.get("wall_s"),
        "llm_steps": result.get("llm_steps"),
    }


def score_all(results: dict, tasks: dict) -> dict:
    """results[task_id][mode] -> rows per mode + summary."""
    rows = {}
    for task_id, task in tasks.items():
        for mode, res in (results.get(task_id) or {}).items():
            rows[f"{task_id}/{mode}"] = score_result(res, task["gt"])
    return rows


def eval_report(results: dict, tasks: dict) -> str:
    """Compact markdown comparison table."""
    lines = ["# 检索模式对比测评(同一 GT)\n",
             "| 任务/模式 | 判定 | 命中GT | 区间数 | 距GT(s) | 耗时(s) | 引擎调用 | LLM步 | 案例 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task_id, task in tasks.items():
        gt_txt = str(task["gt"]) if task["gt"] else "无目标(反例)"
        for mode in ("agent", "playbook", "pipeline"):
            key = f"{task_id}/{mode}"
            r = (results.get(task_id) or {}).get(mode)
            if not r:
                lines.append(f"| {key} | 未运行 | — | — | — | — | — | — | — |")
                continue
            m = score_result(r, task["gt"])
            hit = "✅" if m["gt_hit"] else ("—" if gt_txt.startswith("无目标") else "❌")
            lines.append(
                f"| {key} | {m['verdict']} | {hit} | {m['n_intervals']} | "
                f"{m['best_gap_s'] if m['best_gap_s'] is not None else '—'} | "
                f"{round(m['wall_s'] or 0, 1)} | {len(m['tried'])} | "
                f"{m['llm_steps'] or 0} | {'✓' if m['case_recorded'] else '—'} |")
    return "\n".join(lines)
