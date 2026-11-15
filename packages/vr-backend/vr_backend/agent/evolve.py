#!/usr/bin/env python3
"""E3 — the evolution controller: confirmed cases in, better config out.

Implements §3 of the harness plan, one run through the whole loop:

    ① merge datasets   confirmed cases (case_log, environment-type filtered
                        — env failures aren't retrieval labels) + bench PRW
                        candidates when the artifact exists + CCPD-GT rows
                        (E4 export) when present;
    ② optimize         Optuna threshold sweep (no cloud, µs/trial) +
                        Layer-1 confusable reflection — deterministic from
                        CCPD-GT confusion pairs (keyless) when available,
                        else the LLM reflective loop (needs a text key,
                        budget-capped); no key -> honestly "skipped";
    ③ regression gate  evaluate old vs new on a random 20% HOLDOUT the
                        optimizers never saw; adopt only if
                        new_macro_f1 >= old_macro_f1 - 0.005;
    ④ backfill         fusion/config/{thresholds,confusables}.json with
                        _meta provenance + rollback snapshot (E0 seam);
    ⑤ record           one JSON line in fusion/config/evolutions.jsonl.

Safety (unchanged from the plan): a ``.veto`` file freezes everything;
evolution writes ONLY fusion/config/ + logs, never core code; every step
degrades honestly (missing optuna/rows/keys -> "skipped" + reason, never a
fabricated improvement).

CLI:  python fusion/agent/evolve.py --once | --watch N (minutes)
HTTP: GET /evolve (server) triggers ``run_once``.
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import case_log as CL        # noqa: E402
import evaluator as E       # noqa: E402
import config as CFG        # noqa: E402
import optimize_thresholds as OT   # noqa: E402
try:                                    # keep runs readable; the record has the numbers
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass
sys.path.insert(0, os.path.join(_PARENT, "bench"))
import cycle as CYC         # noqa: E402

# tunables (env-overridable budgets)
MIN_CANDIDATES = int(os.environ.get("EVOLVE_MIN_CANDIDATES", 10))
MIN_ROWS = int(os.environ.get("EVOLVE_MIN_ROWS", 10))
HOLDOUT_FRAC = 0.2
GATE_TOL = 0.005
DEFAULT_TRIALS = int(os.environ.get("OPTUNA_TRIALS", 40))
DEFAULT_LAYER1_CALLS = int(os.environ.get("GEPA_MAX_CALLS", 40))

BENCH_PRW = os.path.join(_PARENT, "bench_out", "prw_x1_full", "candidates.json")
CCPD_ROWS = os.path.join(_PARENT, "bench_out", "ccpd_gt", "rows.json")


def _text_key() -> str | None:
    return (os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY"))


# --------------------------------------------------------------------------- #
# ① dataset merge
# --------------------------------------------------------------------------- #

def _bench_enabled() -> bool:
    """Bench-GT merge switch (tests set EVOLVE_NO_BENCH=1 for hermetic runs)."""
    return os.environ.get("EVOLVE_NO_BENCH", "") != "1"


def merge_datasets(cases_path: str = None, bench_candidates: str = None) -> dict:
    """Confirmed cases + bench GT -> the dataset.py contract shapes.

    ``environment``-type cases are excluded (#8): they record infrastructure
    failures (unreadable video, missing tesseract), not retrieval labels."""
    candidates, rows = [], []
    for c in CL.confirmed_cases(cases_path or CL.DEFAULT_CASES_PATH):
        if c.get("query_type") == "environment":
            continue
        candidates += CL.extract_candidates_from_case(c)
        rows += CL.extract_ocr_rows_from_case(c)
    sources = {"cases": len(candidates), "case_rows": len(rows)}
    path = bench_candidates if bench_candidates is not None else BENCH_PRW
    if _bench_enabled() and path and os.path.exists(path):
        try:
            extra = json.load(open(path))
            extra = [c for c in extra if isinstance(c, dict) and c.get("results")]
            candidates += extra
            sources["bench_prw"] = len(extra)
        except Exception:                   # noqa: BLE001 — silent skip
            sources["bench_prw"] = "unreadable"
    return {"candidates": candidates, "rows": rows, "sources": sources}


def split_holdout(items: list, frac: float = HOLDOUT_FRAC,
                  seed: int = 0) -> tuple[list, list]:
    """Deterministic train/holdout split (the gate's unseen 20%)."""
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_hold = max(1, int(round(len(items) * frac))) if len(items) > 1 else 0
    return items[n_hold:], items[:n_hold]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _next_version(name: str) -> int:
    meta = CFG.load_runtime_config()["meta"].get(f"{name}_meta") or {}
    return int(meta.get("version", 0)) + 1


def _majority_target(rows: list) -> str:
    """The target plate for monitor-OCR rows: the most common positive read."""
    from collections import Counter
    pos = [r["ocr"] for r in rows if r.get("gt") and r.get("ocr")]
    return Counter(pos).most_common(1)[0][0] if pos else ""


def _log(record: dict) -> None:
    os.makedirs(CFG.config_dir(), exist_ok=True)
    path = os.path.join(CFG.config_dir(), "evolutions.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# ②③④ the two optimization blocks
# --------------------------------------------------------------------------- #

def _evolve_thresholds(candidates: list, n_trials: int, seed: int) -> dict:
    """Optuna sweep + holdout gate + backfill. Returns the block record."""
    train, hold = split_holdout(candidates, seed=seed)
    current = CFG.threshold_overrides()
    old = E.threshold_metrics(hold, current, agree=2)
    try:
        res = OT.run(train, n_trials=n_trials, seed=seed)
    except ImportError as e:
        return {"status": "skipped", "reason": f"optuna unavailable: {e}",
                "old_macro_f1": old["macro_f1"]}
    best = {k: v for k, v in res["best"].items() if k != "agree"}
    agree = int(res["best"].get("agree", 2))
    new = E.threshold_metrics(hold, best, agree=agree)
    passed = new["macro_f1"] >= old["macro_f1"] - GATE_TOL
    block = {"status": "adopted" if passed else "rejected",
             "n_train": len(train), "n_holdout": len(hold), "trials": n_trials,
             "before": current or "(defaults)",
             "after": best, "agree": agree,
             "old_macro_f1": old["macro_f1"], "new_macro_f1": new["macro_f1"],
             "gate": f"new >= old - {GATE_TOL}"}
    if passed:
        CFG.save_runtime_config("thresholds", best, source="evolve:optuna",
                                metrics={"holdout_macro_f1": new["macro_f1"],
                                         "previous_macro_f1": old["macro_f1"],
                                         "n_train": len(train),
                                         "n_holdout": len(hold)},
                                version=_next_version("thresholds"))
    return block


def _evolve_layer1(rows: list, seed: int) -> dict:
    """Confusable-table reflection, holdout-gated.

    Priority: (a) deterministic reflection from CCPD-GT confusion pairs
    (real GT, no key, no network); (b) LLM reflective loop over the monitor
    OCR rows (needs a text key); neither possible -> honestly skipped."""
    # (a) CCPD-GT confusion-pair reflection (E4 export present?)
    if _ccpd_rows_available():
        try:
            sys.path.insert(0, os.path.join(_PARENT, "bench"))
            from run_ccpd_gt import reflect_confusables, tolerant_metrics
            ccpd = json.load(open(CCPD_ROWS))
            train, hold = split_holdout(ccpd, seed=seed)
            table0 = CFG.confusables_table() or E.DEFAULT_CONFUSABLES
            cand_table = reflect_confusables(train, table0)
            old = tolerant_metrics(hold, table0)
            new = tolerant_metrics(hold, cand_table)
            passed = (new["tolerant_acc"] >= old["tolerant_acc"] - GATE_TOL
                      and new["false_equal_rate"] <= old["false_equal_rate"] + GATE_TOL)
            block = {"mode": "ccpd_reflect", "n_train": len(train),
                     "n_holdout": len(hold),
                     "old": old, "new": new,
                     "status": "adopted" if passed else "rejected"}
            if passed:
                CFG.save_runtime_config("confusables", cand_table,
                                        source="evolve:ccpd_reflect",
                                        metrics={"tolerant_acc": new["tolerant_acc"],
                                                 "previous": old["tolerant_acc"]},
                                        version=_next_version("confusables"))
            return block
        except Exception as e:              # noqa: BLE001 — fall through honestly
            ccpd_note = f"ccpd reflection failed: {type(e).__name__}: {e}"
    else:
        ccpd_note = None
    # (b) monitor OCR rows via the LLM reflective loop
    if len(rows) >= MIN_ROWS:
        key = _text_key()
        if not key:
            return {"status": "skipped",
                    "reason": "no text key (Layer1 reflection needs "
                              "ZHIPUAI/GLM/DASHSCOPE for the reflective loop)"
                              + (f"; {ccpd_note}" if ccpd_note else "")}
        target = _majority_target(rows)
        train, hold = split_holdout(rows, seed=seed)
        outdir = os.path.join(CFG.config_dir(), "layer1_work")
        os.makedirs(outdir, exist_ok=True)
        report = CYC.layer1(train, target, outdir,
                            max_metric_calls=DEFAULT_LAYER1_CALLS)
        table = report.get("best_table") or E.DEFAULT_CONFUSABLES
        old = E.text_metrics(hold, target, E.DEFAULT_CONFUSABLES)
        new = E.text_metrics(hold, target, table)
        passed = new["macro_f1"] >= old["macro_f1"] - GATE_TOL
        block = {"mode": report.get("mode", "reflective"),
                 "n_train": len(train), "n_holdout": len(hold), "target": target,
                 "old_macro_f1": old["macro_f1"], "new_macro_f1": new["macro_f1"],
                 "status": "adopted" if passed else "rejected"}
        if passed:
            CFG.save_runtime_config("confusables", table,
                                    source="evolve:layer1_reflect",
                                    metrics={"holdout_macro_f1": new["macro_f1"],
                                             "previous": old["macro_f1"]},
                                    version=_next_version("confusables"))
        return block
    return {"status": "skipped",
            "reason": (ccpd_note or
                       f"no labeled OCR rows (need >= {MIN_ROWS}) and no "
                       f"CCPD-GT export at {CCPD_ROWS}")}


def _ccpd_rows_available() -> bool:
    return _bench_enabled() and os.path.exists(CCPD_ROWS)


# --------------------------------------------------------------------------- #
# ⑤ one full evolution run
# --------------------------------------------------------------------------- #

def run_once(trigger: str = "manual", cases_path: str = None,
             n_trials: int = None, seed: int = 0) -> dict:
    """The whole §3 loop. Never raises for missing data/keys — the record
    tells the truth instead. Returns the record (also in evolutions.jsonl)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "trigger": trigger}
    if CFG.vetoed():
        record.update(status="skipped", reason=".veto present — evolution frozen")
        _log(record)
        return record

    ds = merge_datasets(cases_path)
    record["dataset"] = {**ds["sources"],
                         "candidates": len(ds["candidates"]),
                         "rows": len(ds["rows"])}
    if len(ds["candidates"]) < MIN_CANDIDATES and len(ds["rows"]) < MIN_ROWS \
            and not _ccpd_rows_available():
        record.update(status="no_data",
                      reason=f"need >= {MIN_CANDIDATES} candidates or "
                             f">= {MIN_ROWS} confirmed OCR rows "
                             f"(got {len(ds['candidates'])}/{len(ds['rows'])})")
        _log(record)
        return record

    record["thresholds"] = (_evolve_thresholds(ds["candidates"],
                                               n_trials or DEFAULT_TRIALS, seed)
                            if len(ds["candidates"]) >= MIN_CANDIDATES
                            else {"status": "skipped",
                                  "reason": f"insufficient candidates "
                                            f"({len(ds['candidates'])} < {MIN_CANDIDATES})"})
    record["layer1"] = _evolve_layer1(ds["rows"], seed)
    adopted = [b for b in ("thresholds", "layer1")
               if record.get(b, {}).get("status") == "adopted"]
    rejected = [b for b in ("thresholds", "layer1")
                if record.get(b, {}).get("status") == "rejected"]
    record["status"] = ("evolved" if adopted else
                        "rejected" if rejected else "no_improvement")
    record["config_diff"] = {b: {"after": record[b].get("after"),
                                 "best_table_keys": list(
                                     (record[b].get("new") or {}).get("table", {})
                                     or (record[b].get("after") or {}))[:8]}
                             for b in adopted}
    record["note"] = ("report refresh deferred — run `make report` for the "
                      "HTML SOTA view (bench reruns are heavy)")
    _log(record)
    return record


def watch(interval_min: float, cases_path: str = None):
    """Periodic evolution (server --watch / launchd / CLI --watch)."""
    print(f"[evolve:watch] cycle every {interval_min} min", flush=True)
    while True:
        time.sleep(max(interval_min, 0.05) * 60)
        try:
            rec = run_once(trigger="watch", cases_path=cases_path)
            print("[evolve:watch]", json.dumps(
                {k: rec.get(k) for k in ("ts", "status", "config_diff")},
                ensure_ascii=False), flush=True)
        except Exception as e:              # noqa: BLE001 — keep the loop alive
            print("[evolve:watch] error:", e, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="evolution controller (E3)")
    ap.add_argument("--once", action="store_true", help="run one evolution cycle")
    ap.add_argument("--watch", type=float, default=0, metavar="N",
                    help="run every N minutes (standalone daemon)")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.watch:
        watch(args.watch, cases_path=args.cases)
    else:
        rec = run_once(trigger="cli", cases_path=args.cases,
                       n_trials=args.n_trials, seed=args.seed)
        print(json.dumps(rec, ensure_ascii=False, indent=2, default=str))
