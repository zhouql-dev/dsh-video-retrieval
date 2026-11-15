#!/usr/bin/env python3
"""E3 evolution-controller tests — merge → optimize → gate → backfill → log.

No engines, no network. Optuna runs real 1–10-trial sweeps on fabricated
labeled candidates (microseconds each); the LLM layer is exercised through
fakes. Verifies the §3 architecture end-to-end:

  * .veto present -> evolution skipped entirely;
  * no confirmed cases -> honest no_data (never fabricates an evolution);
  * 20 confirmed cases -> Optuna runs -> gate evaluates on the 20% holdout
    -> thresholds.json backfilled (or honestly rejected) -> evolutions.jsonl
    appended -> fresh-process scorer reads the new value;
  * a gate-worsening proposal is REJECTED (no backfill, rollback untouched);
  * environment-type cases are filtered out of the dataset (#8);
  * no text key -> Layer1 records "skipped" while Optuna still runs;
  * text key + monitor rows -> reflective Layer1 through a fake cycle.layer1,
    gated on the holdout;
  * bench PRW candidates merge in when the artifact exists.

Run: ``python3 test_evolve.py`` (exit code 0 = all pass).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）
_PARENT = _ROOT

import evolve as EV           # noqa: E402
import case_log as CL        # noqa: E402
import dataset as DS         # noqa: E402

_passed = _failed = 0
_KEY_VARS = ("ZHIPUAI_API_KEY", "GLM_API_KEY", "DASHSCOPE_API_KEY",
             "LLM_API_KEY")


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def _no_keys():
    saved = {k: os.environ.pop(k, None) for k in _KEY_VARS}
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _tmp_env():
    """Fresh FUSION_CONFIG + cases path; no keys; bench merge DISABLED so the
    tests are hermetic (real bench_out PRW otherwise merges in)."""
    d = tempfile.mkdtemp(prefix="evolve-test-")
    os.environ["FUSION_CONFIG"] = d
    os.environ["EVOLVE_NO_BENCH"] = "1"
    return d, os.path.join(d, "cases.jsonl")


def _person_case(idx: int) -> dict:
    """One confirmed case whose scored rows come from dataset.person_dataset
    (1 TP + 5 FPs, distinct t_s) with GT covering only the TP."""
    tps = DS.person_dataset(n_false_positives=5)
    gt_t = tps[0]["t_s"]
    return {
        "id": f"case_20260814_{idx:03d}",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": f"嫌疑人 {idx}", "query_type": "insufficient",
        "inputs": {"video": "/tmp/v.mp4", "ref": None, "provider_status": {}},
        "plan": ["定式", "rescoring"],
        "attempts": [{"tool": "rescoring", "params": {"agree": 2},
                      "note": "",
                      "envelope": {"outputs": {"scored": tps}}}],
        "outcome": "resolved", "agent_reasoning": "",
        "human_confirmed": True,
        "gt": {"hit_intervals": [[gt_t - 1.0, gt_t + 1.0]]},
    }


def _plate_case(idx: int, all_ocr_path: str) -> dict:
    return {
        "id": f"case_20260814_p{idx:03d}",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": "车牌号 京Q1G728", "query_type": "routine",
        "inputs": {"video": "/tmp/v.mp4", "ref": None, "provider_status": {}},
        "plan": ["fast_plate_scan"],
        "attempts": [{"tool": "fast_plate_scan", "params": {},
                      "note": "",
                      "envelope": {"outputs": {"all_ocr.json": all_ocr_path}}}],
        "outcome": "resolved", "agent_reasoning": "",
        "human_confirmed": True,
        "gt": {"positive_reads": ["Q1G728"]},
    }


def _write_all_ocr(path: str):
    rows = [{"ocr": t, "t_s": i, "frame": i}
            for i, t in enumerate(["Q1G728", "11G728", "419071G57", "AN17119",
                                   "WAUHLOTZS", "2A5", "562", "88C91",
                                   "P0LICE", "0X4K9", "ZZTOP4", "M3GANE"], 1)]
    json.dump(rows, open(path, "w"))
    return path


def _fresh_scorer_reads(cfg_dir: str, key: str):
    code = (f"import sys; sys.path.insert(0, {_CODE!r}); "
            f"import scorer; print(scorer.DEFAULT_THRESHOLDS.get({key!r}))")
    r = subprocess.run([sys.executable, "-c", code],
                       env={**os.environ, "FUSION_CONFIG": cfg_dir},
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip() if r.returncode == 0 else f"<err {r.stderr[-200:]}>"


def test_veto_skips():
    print("[.veto freezes evolution]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        open(os.path.join(cfg, ".veto"), "w").write("frozen")
        rec = EV.run_once(trigger="test", cases_path=cases)
        check("status skipped", rec["status"] == "skipped", rec)
        log = json.load if False else open(os.path.join(cfg, "evolutions.jsonl"))
        lines = [l for l in log.read().splitlines() if l.strip()]
        check("skipped run still logged", any(
            json.loads(l).get("status") == "skipped" for l in lines), lines)
        check("no thresholds.json written",
              not os.path.exists(os.path.join(cfg, "thresholds.json")))
    finally:
        _restore(saved)


def test_no_data():
    print("[no confirmed cases -> no_data]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        rec = EV.run_once(trigger="test", cases_path=cases)
        check("status no_data", rec["status"] == "no_data", rec)
        check("no config written",
              not os.path.exists(os.path.join(cfg, "thresholds.json")))
    finally:
        _restore(saved)


def test_full_evolution_flow():
    print("[20 confirmed cases -> optuna -> gate -> backfill -> scorer reads]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        for i in range(20):
            CL.record_case(_person_case(i), cases)
        rec = EV.run_once(trigger="test", cases_path=cases, n_trials=10, seed=1)
        check("run completed honestly", rec["status"]
              in ("evolved", "rejected", "no_improvement"), rec.get("status"))
        th = rec.get("thresholds", {})
        check("thresholds block ran", th.get("status") in ("adopted", "rejected"),
              th)
        check("holdout gate evaluated",
              "old_macro_f1" in th and "new_macro_f1" in th, th)
        check("dataset counts recorded",
              rec["dataset"]["candidates"] >= 100, rec["dataset"])
        check("evolutions.jsonl appended", os.path.exists(
            os.path.join(cfg, "evolutions.jsonl")))
        if th.get("status") == "adopted":
            tj = json.load(open(os.path.join(cfg, "thresholds.json")))
            check("thresholds.json holds best params",
                  all(k in tj for k in th["after"]), tj.keys())
            check("_meta stamped by evolve",
                  tj["_meta"]["source"].startswith("evolve"), tj["_meta"])
            key = next(iter(th["after"]))
            check("fresh scorer reads new value",
                  _fresh_scorer_reads(cfg, key) == str(th["after"][key]),
                  f"{_fresh_scorer_reads(cfg, key)} != {th['after'][key]}")
        else:
            check("rejection leaves no thresholds.json",
                  not os.path.exists(os.path.join(cfg, "thresholds.json")))
        check("gate never degrades holdout",
              th.get("status") != "adopted"
              or th["new_macro_f1"] >= th["old_macro_f1"] - EV.GATE_TOL, th)
        # layer1 skip (no rows, no key)
        check("layer1 honest without rows/key",
              rec.get("layer1", {}).get("status") == "skipped", rec.get("layer1"))
    finally:
        _restore(saved)


def test_gate_rejects_worse():
    print("[gate rejects a worsening proposal]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        for i in range(20):
            CL.record_case(_person_case(i), cases)

        class FakeOT:
            @staticmethod
            def run(cands, n_trials=10, seed=0, search_space=None, **kw):
                # hue_dhue collapsed to the space floor kills the hue signal
                # for every FP with dhue>=15 -> holdout degrades hard.
                return {"best": {"hue_dhue": 15.0, "temporal_peak": 0.80,
                                 "temporal_bellness": 0.70,
                                 "temporal_span_s": 2.00, "agree": 3},
                        "best_macro_f1": 0.0, "n_trials": n_trials,
                        "history": []}

        real = EV.OT
        EV.OT = FakeOT
        try:
            rec = EV.run_once(trigger="test", cases_path=cases, n_trials=3)
        finally:
            EV.OT = real
        th = rec["thresholds"]
        check("rejected", th["status"] == "rejected", th)
        check("no backfill on rejection",
              not os.path.exists(os.path.join(cfg, "thresholds.json")))
        check("rejection recorded in log", any(
            json.loads(l).get("thresholds", {}).get("status") == "rejected"
            for l in open(os.path.join(cfg, "evolutions.jsonl"))
            if l.strip()))
    finally:
        _restore(saved)


def test_environment_cases_filtered():
    print("[environment-type cases filtered from dataset (#8)]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        c = _person_case(99)
        c["query_type"] = "environment"       # e.g. the preflight-argv bug case
        CL.record_case(c, cases)
        ds = EV.merge_datasets(cases)
        check("environment case excluded",
              len(ds["candidates"]) == 0 and len(ds["rows"]) == 0
              and ds["sources"]["cases"] == 0,
              {k: (v if not isinstance(v, list) else len(v))
               for k, v in ds.items()} | ds["sources"])
    finally:
        _restore(saved)


def test_layer1_no_key_skips():
    print("[no text key -> Layer1 skipped, Optuna still runs]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        ocr = _write_all_ocr(os.path.join(cfg, "all_ocr.json"))
        for i in range(20):
            CL.record_case(_person_case(i), cases)
        CL.record_case(_plate_case(1, ocr), cases)
        rec = EV.run_once(trigger="test", cases_path=cases, n_trials=5)
        check("optuna still ran", rec["thresholds"]["status"]
              in ("adopted", "rejected"), rec["thresholds"])
        check("layer1 skipped with reason",
              rec["layer1"]["status"] == "skipped"
              and "key" in rec["layer1"]["reason"].lower(), rec["layer1"])
    finally:
        _restore(saved)


def test_layer1_with_key_faked():
    print("[text key + monitor rows -> reflective Layer1 (faked), holdout-gated]")
    os.environ["ZHIPUAI_API_KEY"] = "fake-key-for-wiring"
    try:
        cfg, cases = _tmp_env()
        ocr = _write_all_ocr(os.path.join(cfg, "all_ocr.json"))
        for i in range(3):
            CL.record_case(_plate_case(i, ocr), cases)

        better_table = {"Q": "O0D9G16", "1": "ILT7Q", "G": "6C0Q19"}

        def fake_layer1(rows, target, out, max_metric_calls=40):
            return {"layer": "layer1_text", "mode": "reflective_direct",
                    "n_rows": len(rows), "target": target,
                    "before_macro_f1": 0.5, "after_macro_f1": 0.9,
                    "best_table": better_table, "history": []}

        real = EV.CYC
        EV.CYC = type("FakeCyc", (), {"layer1": staticmethod(fake_layer1)})
        try:
            rec = EV.run_once(trigger="test", cases_path=cases, n_trials=3)
        finally:
            EV.CYC = real
        l1 = rec["layer1"]
        check("layer1 ran via fake cycle", l1["status"] in ("adopted", "rejected"),
              l1)
        if l1["status"] == "adopted":
            cj = json.load(open(os.path.join(cfg, "confusables.json")))
            check("confusables.json backfilled", cj["Q"] == better_table["Q"], cj)
    finally:
        os.environ.pop("ZHIPUAI_API_KEY", None)


def test_prw_merge():
    print("[bench PRW candidates merge when artifact exists]")
    saved = _no_keys()
    try:
        cfg, cases = _tmp_env()
        os.environ.pop("EVOLVE_NO_BENCH", None)   # re-enable bench merge
        prw = os.path.join(_PARENT, "bench_out", "prw_x1_full", "candidates.json")
        if not os.path.exists(prw):
            print("  (PRW artifact absent — merge path covered by shape check)")
            ds = EV.merge_datasets(cases, bench_candidates="/nonexistent.json")
            check("merge works without bench artifact",
                  len(ds["candidates"]) == 0 and ds["sources"]["cases"] == 0,
                  ds["sources"])
            return
        ds = EV.merge_datasets(cases)
        check("PRW candidates merged", ds["sources"].get("bench_prw", 0) > 100,
              ds["sources"])
        check("merged rows keep dataset contract",
              all("gt" in c and "results" in c for c in ds["candidates"][:20]))
    finally:
        os.environ["EVOLVE_NO_BENCH"] = "1"
        _restore(saved)


if __name__ == "__main__":
    test_veto_skips()
    test_no_data()
    test_full_evolution_flow()
    test_gate_rejects_worse()
    test_environment_cases_filtered()
    test_layer1_no_key_skips()
    test_layer1_with_key_faked()
    test_prw_merge()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
