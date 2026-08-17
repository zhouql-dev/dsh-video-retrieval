#!/usr/bin/env python3
"""E0 runtime config layer — the seam that makes self-evolution possible.

Reads ``fusion/config/`` (override with the ``FUSION_CONFIG`` env var):

    thresholds.json    flat {threshold: value} layered OVER the hardcoded
                       scorer.DEFAULT_THRESHOLDS at import time (CLI flags
                       still win per-call — priority CLI > config > default);
    confusables.json   {char: confusable-string} read by fast_plate_scan via
                       ``FUSION_CONFUSABLES`` / its default path;
    prompts.json       {refine|verify|match: template} overriding the
                       provider prompt texts (templates may use ``{query}``).

Every file may carry a top-level ``_meta`` record (version / updated_at /
source / metrics) stamped by the evolution controller (E3). ``_meta`` is
never merged into consumer dicts.

Hard guarantees (this is the safety valve for evolution):
  * missing or malformed files degrade SILENTLY to current behavior;
  * ``save_runtime_config`` snapshots the previous file into
    ``rollback/<utc-ts>/`` before overwriting — one-command rollback;
  * a ``.veto`` file freezes the layer (evolution checks ``vetoed()``);
  * nothing here ever writes core code — only these JSON files + logs.
"""
from __future__ import annotations
import json
import os
import shutil
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

CONFIG_FILES = ("thresholds", "confusables", "prompts")


def config_dir() -> str:
    """The active config directory (``FUSION_CONFIG`` env overrides default)."""
    return os.environ.get("FUSION_CONFIG") or _DEFAULT_DIR


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: str):
    """Return (data, error). data={} when missing/malformed/not an object."""
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}, f"not a JSON object: {path}"
        return data, None
    except Exception as e:                       # noqa: BLE001 — degrade silently
        return {}, f"{type(e).__name__}: {e}"


def _without_meta(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "_meta"}


# --------------------------------------------------------------------------- #
# loading (silent degradation is the contract)
# --------------------------------------------------------------------------- #

def load_runtime_config() -> dict:
    """Load all three files. Never raises; absent/bad files degrade to {}.

    Returns ``{thresholds, confusables, prompts, meta}`` where meta carries
    the active dir, each file's ``_meta`` provenance, and per-file errors.
    """
    d = config_dir()
    out = {"thresholds": {}, "confusables": {}, "prompts": {},
           "meta": {"dir": d, "errors": {}}}
    for name in CONFIG_FILES:
        data, err = _read_json(os.path.join(d, f"{name}.json"))
        out[name] = _without_meta(data)
        if "_meta" in data:
            out["meta"][f"{name}_meta"] = data["_meta"]
        if err:
            out["meta"]["errors"][f"{name}.json"] = err
    return out


def threshold_overrides() -> dict:
    """The thresholds.json values (no _meta). Layered over the hardcoded
    scorer.DEFAULT_THRESHOLDS by scorer.py at import time."""
    return load_runtime_config()["thresholds"]


def confusables_table() -> dict:
    """The confusables.json values (no _meta) for OCR tolerance."""
    return load_runtime_config()["confusables"]


def prompt_overrides() -> dict:
    """The prompts.json values (no _meta) for provider prompt templates."""
    return load_runtime_config()["prompts"]


def summary() -> dict:
    """One-line observability for run.py / server logs."""
    rc = load_runtime_config()
    return {
        "dir": rc["meta"]["dir"],
        "thresholds_loaded": bool(rc["thresholds"]),
        "confusables_loaded": bool(rc["confusables"]),
        "prompts_loaded": bool(rc["prompts"]),
        "errors": rc["meta"].get("errors") or None,
        "vetoed": vetoed(),
    }


# --------------------------------------------------------------------------- #
# writing (evolution backfill) + rollback snapshots
# --------------------------------------------------------------------------- #

def _rollback_dir() -> str:
    return os.path.join(config_dir(), "rollback")


def snapshot_current(name: str) -> str | None:
    """Copy the current ``name``.json into rollback/<utc-ts>/; None if absent."""
    src = os.path.join(config_dir(), f"{name}.json")
    if not os.path.exists(src):
        return None
    snap_dir = os.path.join(_rollback_dir(), _iso_now())
    os.makedirs(snap_dir, exist_ok=True)
    dst = os.path.join(snap_dir, f"{name}.json")
    shutil.copy2(src, dst)
    return dst


def save_runtime_config(name: str, data: dict, *, source: str = "manual",
                        metrics: dict = None, version: int = 1) -> str:
    """Backfill ``name``.json (one of CONFIG_FILES) with _meta provenance.

    Snapshots the previous file into rollback/ first (the one-command
    rollback safety net). Returns the written path.
    """
    if name not in CONFIG_FILES:
        raise ValueError(f"unknown config file {name!r}; use one of {CONFIG_FILES}")
    os.makedirs(config_dir(), exist_ok=True)
    snapshot_current(name)
    payload = {**data, "_meta": {"version": version, "updated_at": _iso_now(),
                                 "source": source, "metrics": metrics or {}}}
    path = os.path.join(config_dir(), f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def rollback_latest() -> list[str]:
    """Restore the newest rollback snapshot of each file. Returns names."""
    rb = _rollback_dir()
    if not os.path.isdir(rb):
        return []
    stamps = sorted(os.listdir(rb), reverse=True)
    restored = []
    for name in CONFIG_FILES:
        for stamp in stamps:
            snap = os.path.join(rb, stamp, f"{name}.json")
            if os.path.exists(snap):
                shutil.copy2(snap, os.path.join(config_dir(), f"{name}.json"))
                restored.append(name)
                break
    return restored


def vetoed() -> bool:
    """True while the config layer is frozen by an operator (.veto file)."""
    return os.path.exists(os.path.join(config_dir(), ".veto"))


def set_veto(frozen: bool, reason: str = "") -> bool:
    """Create/remove the .veto freeze file. Returns the new vetoed() state."""
    path = os.path.join(config_dir(), ".veto")
    if frozen:
        os.makedirs(config_dir(), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(reason or "frozen by operator")
    elif os.path.exists(path):
        os.remove(path)
    return vetoed()
