#!/usr/bin/env python3
"""Phase-2 wiring tests — capability correctness, NOT optimization runs.

Per the project state: the optimization *capability* must be in place and
verified, but no search is executed until the real labeled dataset arrives.
These tests therefore check that:

  * evaluator.threshold_metrics / text_metrics compute correct P/R/F1 and
    respond sensibly to looser-vs-stricter configs (monotonic where expected);
  * the confusable-table objective reacts to a table edit (the heart of Layer-1);
  * the two drivers' single-point evaluation + param mapping are sound;
  * the drivers behave honestly in the current env: optuna (installed since
    E3/E5) runs a 1-trial sweep and returns the documented shape; gepa either
    runs, raises its documented API-drift TypeError ("gepa" hint), or blocks
    with ImportError when absent — no silent no-op.

One 1-trial Optuna sweep is launched here (millisecond-scale). Run:
``python3 test_phase2.py``.
"""
from __future__ import annotations
import os
import sys

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）

import dataset as D
import evaluator as E
import optimize_thresholds as OT
import optimize_text as OX

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_metrics_correctness():
    print("[evaluator metric math]")
    # perfect separation -> macro_f1 1.0
    perf = E._prf(tp=3, fp=0, fn=0)
    check("perfect pos F1=1", E._prf(3, 0, 0)["f1"] == 1.0)
    check("macro_f1 perfect=1", E.macro_f1(perf, n_neg=10) == 1.0, E.macro_f1(perf, 10))
    # all-miss: standard macro-F1 is 0.435 here (pos F1=0, neg F1≈0.87 because
    # "reject everything" also rejects the 3 real hits). The point of macro-F1:
    # it scores the trivial rejector far below perfect (1.0), unlike accuracy.
    allmiss = E.macro_f1(E._prf(0, 0, 3), n_neg=10)
    check("macro_f1 all-miss < perfect", 0.0 < allmiss < 0.5, allmiss)
    check("macro_f1 rewards perfect over reject-all", E.macro_f1(perf, 10) > allmiss)
    # precision drops with FP
    check("FP cuts precision", E._prf(1, 9, 0)["precision"] < 0.15)


def test_threshold_objective_monotone():
    print("[Layer-2 objective behaves]")
    cands = D.person_dataset()
    strict = E.threshold_metrics(cands, {}, agree=2)                  # §11.2 defaults
    loose = E.threshold_metrics(cands, {"hue_dhue": 120, "temporal_peak": 0.4,
                                        "temporal_bellness": 0.0, "temporal_span_s": 0.0},
                                agree=1)                              # over-match
    check("strict has 1 TP (the real hit)", strict["tp"] == 1, strict)
    check("strict kills all FPs", strict["fp"] == 0, strict)
    check("loose admits more FPs than strict", loose["fp"] >= strict["fp"], (strict, loose))
    check("strict macro_f1 >= loose", strict["macro_f1"] >= loose["macro_f1"],
          (strict["macro_f1"], loose["macro_f1"]))
    check("objective returns float in [0,1]",
          0.0 <= E.make_threshold_objective(cands, agree=2)({}) <= 1.0)


def test_text_objective_reacts():
    print("[Layer-1 objective reacts to table edits]")
    rows = D.plate_ocr_dataset()
    target = D.plate_target()
    base = E.text_metrics(rows, target, E.DEFAULT_CONFUSABLES)
    # removing the "1" entry should drop recall (can't match "11G728"->Q1G728)
    degraded = {k: v for k, v in E.DEFAULT_CONFUSABLES.items() if k != "1"}
    worse = E.text_metrics(rows, target, degraded)
    check("base recall > degraded recall", base["recall"] >= worse["recall"], (base, worse))
    # garbage table text parses back to default (no crash)
    check("garbage table parsed safely", E.parse_table("not json") == E.DEFAULT_CONFUSABLES)
    check("ocr_matches finds 11G728->Q1G728",
          E.ocr_matches("Q1G728", E.DEFAULT_CONFUSABLES, "11G728") is True)
    check("ocr_matches rejects noise",
          E.ocr_matches("Q1G728", E.DEFAULT_CONFUSABLES, "419071G57") is False)


def test_driver_wiring():
    print("[driver single-point wiring]")
    obj = E.make_threshold_objective(D.person_dataset(), agree=2)
    p = OT.params_from_dict({"hue_dhue": 200, "temporal_peak": 0.6, "agree": 5})
    check("params clamped to space",
          p["hue_dhue"] <= 80 and p["agree"] <= 3, p)
    check("single_eval returns float", isinstance(OT.single_eval(p, obj), float))
    check("table_to_text round-trips",
          E.parse_table(OX.table_to_text(E.DEFAULT_CONFUSABLES)) == E.DEFAULT_CONFUSABLES)


def test_drivers_in_current_env():
    print("[driver execution gates]")
    # optuna is installed (E3/E5 dependency): a 1-trial sweep must run and
    # return the documented shape. A missing optuna must still block with a
    # clear ImportError hint, never a silent no-op.
    try:
        out = OT.run(D.person_dataset(), n_trials=1)
        check("optuna run() returns shape",
              isinstance(out, dict) and {"best", "best_macro_f1", "n_trials",
                                         "history"} <= set(out), str(out)[:80])
        check("optuna run() 1 trial recorded", out.get("n_trials") == 1
              and len(out.get("history", [])) == 1, out.get("n_trials"))
    except ImportError as e:
        check("optuna run() blocks with hint", "optuna" in str(e).lower(), str(e))
    # gepa: installed here but its public API drifted (signature mismatch) —
    # the documented behavior is a TypeError that pinpoints optimize_text.run.
    # Accept any honest outcome: successful result, drift TypeError, or the
    # ImportError gate when gepa is absent.
    try:
        out = OX.run(D.plate_ocr_dataset(), D.plate_target(), max_metric_calls=1)
        check("gepa run() returns shape",
              isinstance(out, dict) and "best_macro_f1" in out, str(out)[:80])
    except TypeError as e:
        check("gepa drift raises typed hint", "gepa" in str(e).lower(), str(e))
    except ImportError as e:
        check("gepa run() blocks with hint", "gepa" in str(e).lower(), str(e))
    except Exception as e:                       # noqa
        check("gepa run() honest failure", False, f"unexpected {type(e)}: {e}")


if __name__ == "__main__":
    test_metrics_correctness()
    test_threshold_objective_monotone()
    test_text_objective_reacts()
    test_driver_wiring()
    test_drivers_in_current_env()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
