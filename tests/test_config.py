#!/usr/bin/env python3
"""E0 runtime-config-layer tests — load / merge / degrade / rollback.

Validates the self-evolution seam (fusion/config.py) without any engines:

  * missing files -> silent degradation to current behavior (empty dicts);
  * valid files -> thresholds/confusables/prompts load, _meta preserved;
  * malformed JSON -> silent degradation + error recorded in meta;
  * save_runtime_config snapshots the old file into rollback/ and stamps
    _meta {version, updated_at, source, metrics}; rollback_latest restores;
  * .veto freeze flag;
  * the three consumer wirings, each via a fresh subprocess import so the
    import-time merge is what is tested:
      - scorer.DEFAULT_THRESHOLDS layers fusion/config/thresholds.json;
      - provider._PROMPT_OVERRIDES layers fusion/config/prompts.json;
      - fast_plate_scan.CONFUSABLES reads FUSION_CONFUSABLES / default path.

Run: ``python3 test_config.py`` (exit code 0 = all pass).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）

import config as C

SKILL_SCRIPTS = os.environ.get(
    "SKILL_SCRIPTS",
    "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def _tmp_config_dir():
    d = tempfile.mkdtemp(prefix="fusion-config-test-")
    os.environ["FUSION_CONFIG"] = d
    return d


def _run_py(code, cwd=None, extra_env=None):
    """Run a one-liner under the venv python with FUSION_CONFIG inherited."""
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd or _HERE,
                          env=env, capture_output=True, text=True, timeout=120)


def test_missing_degrades():
    print("[missing config files degrade silently]")
    d = _tmp_config_dir()
    rc = C.load_runtime_config()
    check("empty thresholds", rc["thresholds"] == {}, rc["thresholds"])
    check("empty confusables", rc["confusables"] == {}, rc["confusables"])
    check("empty prompts", rc["prompts"] == {}, rc["prompts"])
    check("meta records dir", rc["meta"].get("dir") == d, rc["meta"])
    check("no errors for absent files", not rc["meta"].get("errors"), rc["meta"])


def test_valid_files_load():
    print("[valid files load with _meta preserved]")
    d = _tmp_config_dir()
    for name, payload in {
        "thresholds": {"hue_dhue": 41.5, "reid_sim": 0.61,
                       "_meta": {"version": 3, "source": "evolve"}},
        "confusables": {"Q": "O0X", "_meta": {"version": 1, "source": "layer1"}},
        "prompts": {"verify": "CUSTOM {query}", "_meta": {"version": 2}},
    }.items():
        with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    rc = C.load_runtime_config()
    check("thresholds loaded", rc["thresholds"] == {"hue_dhue": 41.5,
                                                    "reid_sim": 0.61}, rc["thresholds"])
    check("confusables loaded", rc["confusables"] == {"Q": "O0X"}, rc["confusables"])
    check("prompts loaded", rc["prompts"] == {"verify": "CUSTOM {query}"}, rc["prompts"])
    check("thresholds _meta preserved",
          rc["meta"]["thresholds_meta"] == {"version": 3, "source": "evolve"},
          rc["meta"].get("thresholds_meta"))
    check("convenience accessors",
          C.threshold_overrides() == rc["thresholds"]
          and C.confusables_table() == rc["confusables"]
          and C.prompt_overrides() == rc["prompts"])
    check("summary reports loaded", C.summary().get("thresholds_loaded") is True,
          C.summary())


def test_bad_json_degrades():
    print("[malformed JSON degrades silently]")
    d = _tmp_config_dir()
    with open(os.path.join(d, "thresholds.json"), "w") as f:
        f.write("{not valid json")
    with open(os.path.join(d, "confusables.json"), "w") as f:
        f.write('["a", "list", "not", "an", "object"]')
    rc = C.load_runtime_config()
    check("bad thresholds -> {}", rc["thresholds"] == {}, rc["thresholds"])
    check("list confusables -> {}", rc["confusables"] == {}, rc["confusables"])
    check("errors recorded per file",
          "thresholds.json" in rc["meta"].get("errors", {})
          and "confusables.json" in rc["meta"].get("errors", {}), rc["meta"])


def test_save_snapshot_rollback():
    print("[save stamps _meta + snapshots old file; rollback restores]")
    d = _tmp_config_dir()
    p1 = C.save_runtime_config("thresholds", {"hue_dhue": 40.0}, source="evolve",
                               metrics={"macro_f1": 0.9})
    check("save writes file", os.path.exists(p1), p1)
    first = json.load(open(p1, encoding="utf-8"))
    check("_meta stamped", first["_meta"].get("source") == "evolve"
          and first["_meta"].get("version") == 1
          and "updated_at" in first["_meta"], first["_meta"])
    check("_meta metrics kept", first["_meta"].get("metrics") == {"macro_f1": 0.9},
          first["_meta"])
    C.save_runtime_config("thresholds", {"hue_dhue": 50.0}, source="evolve",
                          metrics={"macro_f1": 0.7})
    check("file now holds v2", json.load(open(p1, encoding="utf-8"))["hue_dhue"] == 50.0)
    rb = os.path.join(d, "rollback")
    snaps = sorted(os.listdir(rb)) if os.path.isdir(rb) else []
    check("snapshot dir created", bool(snaps), rb)
    restored = json.load(open(os.path.join(rb, snaps[-1], "thresholds.json"),
                              encoding="utf-8"))
    check("snapshot holds pre-write value", restored["hue_dhue"] == 40.0, restored)
    names = C.rollback_latest()
    check("rollback restores v1", "thresholds" in names
          and json.load(open(p1, encoding="utf-8"))["hue_dhue"] == 40.0, names)
    check("summary reflects config dir", C.summary().get("dir") == d)


def test_veto_flag():
    print("[.veto freeze flag]")
    d = _tmp_config_dir()
    check("no veto by default", C.vetoed() is False)
    with open(os.path.join(d, ".veto"), "w") as f:
        f.write("frozen by operator\n")
    check("veto detected", C.vetoed() is True)
    os.remove(os.path.join(d, ".veto"))
    check("veto cleared", C.vetoed() is False)


def test_scorer_import_merge():
    print("[scorer.DEFAULT_THRESHOLDS layers thresholds.json at import]")
    d = _tmp_config_dir()
    with open(os.path.join(d, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"hue_dhue": 41.5}, f)
    r = _run_py("import scorer; print(scorer.DEFAULT_THRESHOLDS.get('hue_dhue'))")
    check("scorer sees config value", r.returncode == 0 and r.stdout.strip() == "41.5",
          r.stdout + r.stderr)
    # empty dir -> hardcoded default untouched
    empty = tempfile.mkdtemp(prefix="fusion-config-empty-")
    r2 = _run_py("import scorer; print(scorer.DEFAULT_THRESHOLDS.get('hue_dhue'))",
                 extra_env={"FUSION_CONFIG": empty})
    check("scorer keeps default without config",
          r2.returncode == 0 and r2.stdout.strip() == "35.0", r2.stdout + r2.stderr)


def test_provider_prompt_override():
    print("[provider._PROMPT_OVERRIDES layers prompts.json]")
    d = _tmp_config_dir()
    with open(os.path.join(d, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump({"verify": "CUSTOM VERIFY {query}", "refine": "CUSTOM REFINE {query}",
                   "match": "CUSTOM MATCH"}, f)
    r = _run_py("import provider; print(provider._PROMPT_OVERRIDES.get('verify'));"
                "print(provider._PROMPT_OVERRIDES.get('match'))")
    lines = r.stdout.strip().splitlines()
    check("verify overridden", r.returncode == 0 and lines[0] == "CUSTOM VERIFY {query}",
          r.stdout + r.stderr)
    check("match overridden", len(lines) > 1 and lines[1] == "CUSTOM MATCH",
          r.stdout + r.stderr)
    r2 = _run_py("import provider; print(bool(provider._PROMPT_OVERRIDES))",
                 extra_env={"FUSION_CONFIG": tempfile.mkdtemp()})
    check("no prompts.json -> empty overrides",
          r2.returncode == 0 and r2.stdout.strip() == "False", r2.stdout + r2.stderr)


def test_fast_plate_scan_confusables():
    print("[fast_plate_scan.CONFUSABLES reads external JSON]")
    tmp = tempfile.mkdtemp(prefix="fusion-confusables-")
    path = os.path.join(tmp, "confusables.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"Q": "O0X", "_meta": {"version": 7}}, f)
    code = ("import fast_plate_scan as f; "
            "print(f.CONFUSABLES.get('Q'), f.CONFUSABLES.get('_meta'))")
    r = _run_py(code, cwd=SKILL_SCRIPTS, extra_env={"FUSION_CONFUSABLES": path})
    check("env path wins over hardcode", r.returncode == 0
          and r.stdout.strip() == "O0X None", r.stdout + r.stderr)
    r2 = _run_py("import fast_plate_scan as f; print(f.CONFUSABLES.get('Q'))",
                 cwd=SKILL_SCRIPTS,
                 extra_env={"FUSION_CONFUSABLES": os.path.join(tmp, "nope.json")})
    check("missing path falls back to hardcode", r2.returncode == 0
          and r2.stdout.strip() == "O0D9G16", r2.stdout + r2.stderr)


if __name__ == "__main__":
    test_missing_degrades()
    test_valid_files_load()
    test_bad_json_degrades()
    test_save_snapshot_rollback()
    test_veto_flag()
    test_scorer_import_merge()
    test_provider_prompt_override()
    test_fast_plate_scan_confusables()
    print(f"\n{'='*48}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
