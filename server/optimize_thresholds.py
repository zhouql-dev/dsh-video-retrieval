#!/usr/bin/env python3
"""Layer-2 optimizer — numerical threshold sweep (Optuna).

CAPABILITY ONLY — does not auto-execute. Call ``run()`` explicitly once a REAL
labeled candidate set is in place (see dataset.py's contract). Until then this
module exposes the search space and a unit-testable single-trial evaluator so
the wiring is verified without running a sweep.

How it optimizes (the mechanism)
--------------------------------
The §11.2 signal thresholds (Δhue, OSNet peak, bellness, span, ``--agree``)
live in ``scorer.DEFAULT_THRESHOLDS``. Each one is a continuous/integer knob.
``evaluator.make_threshold_objective`` turns a candidate set into a pure
function ``f(thresholds) -> macro_f1`` that re-scores the cached per-candidate
signal payloads — no CV, no network, microseconds per call. That cheapness is
what makes a Bayesian sweep affordable:

  Optuna runs N trials. Each trial:
    1. suggests a point in SEARCH_SPACE (TPE after a few random warmups — it
       builds a posterior over which threshold regions score well, so it
       converges far faster than grid/random on a 5-D space);
    2. calls the objective -> macro_f1;
    3. records it. The best trial's thresholds are the optimized config, which
       drop straight back into ``run.py --hue-dhue … --agree …``.

This is the §11.2 "REID 0.55→?" question answered numerically: instead of
hand-picking 0.65, the sweep finds the threshold that maximizes macro-F1 on
labeled data. Layer-1 (gepa) uses the SAME evaluator style for text artifacts.
"""
from __future__ import annotations
from typing import Callable

import evaluator

# name -> (low, high, kind). kind 'int' -> suggest_int, else suggest_float.
SEARCH_SPACE = {
    "hue_dhue":          (15.0, 80.0, "float"),   # 35 = §11.2 same-person line
    "temporal_peak":     (0.40, 0.80, "float"),   # OSNet sim gate (report: 0.55)
    "temporal_bellness": (0.10, 0.70, "float"),   # sustained-elevation gate
    "temporal_span_s":   (0.00, 2.00, "float"),   # rejects single-frame spikes
    "agree":             (1, 3, "int"),           # min firing signals
}


def params_from_trial(trial, space: dict = None) -> dict:
    """Map an Optuna Trial onto a thresholds dict (+ agree) via the search
    space. Also works with any object exposing ``suggest_float``/``suggest_int``."""
    space = space or SEARCH_SPACE
    out = {}
    for name, (lo, hi, kind) in space.items():
        if kind == "int":
            out[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            out[name] = trial.suggest_float(name, lo, hi)
    return out


def params_from_dict(d: dict, space: dict = None) -> dict:
    """Clamp a plain dict to the search space bounds (for wiring tests / manual
    point evaluation without Optuna)."""
    space = space or SEARCH_SPACE
    out = {}
    for name, (lo, hi, kind) in space.items():
        if name in d:
            v = float(d[name])
            out[name] = int(round(min(max(v, lo), hi))) if kind == "int" else round(min(max(v, lo), hi), 4)
    return out


def single_eval(params: dict, objective: Callable[[dict], float], space: dict = None) -> float:
    """Evaluate one point (dict) — the unit-testable core of a trial."""
    return objective(params_from_dict(params, space))


# per-benchmark search spaces (name -> (lo, hi, kind))
PRW_SPACE = {                     # two-signal space: reid sim + clothing hue
    "reid_sim": (0.50, 0.92, "float"),
    "hue_dhue": (15.0, 90.0, "float"),
}


def run(candidates: list[dict], n_trials: int = 100, seed: int = 0,
        sampler: str = "tpe", search_space: dict = None) -> dict:
    """Run the Optuna sweep. REQUIRES ``optuna`` AND a real labeled candidate
    set. Returns {"best": thresholds, "best_macro_f1": float, "n_trials": int,
    "history": [(macro_f1, thresholds), ...]}.

    This is deliberately NOT called on import or by any test — invoke it only
    once the production dataset (dataset.py contract) is ready.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Layer-2 optimizer needs optuna — pip install optuna. "
            "(Capability is in place; only the search execution is blocked.)") from e
    space = search_space or SEARCH_SPACE

    def _agree_for(p):
        return int(p.get("agree", 2))
    # agree is swept, so the objective must be rebuilt per agree value.
    def objective(trial):
        params = params_from_trial(trial, space)
        agree = _agree_for(params)
        thr = {k: v for k, v in params.items() if k != "agree"}
        return evaluator.make_threshold_objective(candidates, agree=agree)(thr)

    if sampler == "random":
        sam = optuna.samplers.RandomSampler(seed=seed)
    else:
        sam = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sam)
    study.optimize(objective, n_trials=n_trials)
    return {"best": dict(study.best_params), "best_macro_f1": round(study.best_value, 4),
            "n_trials": n_trials,
            "history": [(round(t.value, 4), dict(t.params)) for t in study.trials]}


if __name__ == "__main__":
    # Wiring check only (NOT an optimization run): evaluate a couple of points
    # by hand to confirm the objective + param mapping behave.
    import dataset as D
    obj = evaluator.make_threshold_objective(D.person_dataset(), agree=2)
    for label, p in [("§11.2 defaults", {"hue_dhue": 35, "temporal_peak": 0.55,
                                          "temporal_bellness": 0.4, "temporal_span_s": 0.5,
                                          "agree": 2}),
                     ("permissive",     {"hue_dhue": 80, "temporal_peak": 0.4,
                                          "temporal_bellness": 0.1, "temporal_span_s": 0.0,
                                          "agree": 1})]:
        print(f"  {label:16} -> macro_f1={single_eval(p, obj)}")
    print("(run() is capability-only; call it with a real dataset + optuna installed.)")
