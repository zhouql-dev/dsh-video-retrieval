---
name: video-retrieval
description: Surveillance video retrieval inside DeepSeek Harness — locate or verify a described or pictured target (person / vehicle / plate) in a video via the native vr_* tools, producing temporal presence intervals, per-frame bounding-box trajectories, and annotated artifacts, with plan-then-code workflow and evidence-based guardrails ("insufficient" is a verdict; engine/cloud failure is never "target absent"; heavy CV runs through vr_* tools, never bash).
whenToUse: Find, track, or verify a person/vehicle/object in a surveillance video by natural-language description or reference image; retrieve a vehicle by plate; pre-flight and recover unreadable DVR/NVR exports; run the self-improvement loop (vr_evolve) over confirmed cases and benchmarks.
---

# video-retrieval (DSH mode)

Operating manual for the `video-retrieval` DSH mode. The deterministic core is the
vendored engine under `dsh/engine/`; this skill is what the agent follows.

## Tools (all native, job-based unless noted)

| Tool | Purpose |
|---|---|
| `vr_preflight` | **ALWAYS first** (blocking, seconds): decode check, sample frame, ffmpeg recovery. |
| `vr_search` | Generic retrieval job (query and/or reference image; backend routes precise-text vs semantic). |
| `vr_locate` | General-target job (open-vocab grounding + tracking + VLM seed disambiguation). |
| `vr_verify_target` | Specific-identifier job (detect-everywhere + per-crop VLM verification, local OCR fallback for plates). |
| `vr_person_search` | Person/face job from a reference image (ArcFace/OSNet/CLIP + three-signal vote). |
| `vr_fast_plate_scan` | Whole-clip plate job (local OCR + character-confusion table; no cloud key needed). |
| `vr_job_status` / `vr_job_result` | Poll / collect a job (jobs run minutes — poll, don't block). |
| `vr_cases` / `vr_case_confirm` | Case library; confirmation is the ONE human action feeding evolution. |
| `vr_config` / `vr_veto` | Read active evolution config; freeze/unfreeze the evolution loop. |
| `vr_evolve` | ONE self-evolution cycle (Optuna + reflection → holdout gate → rollback → hot reload). |
| `vr_report` / `vr_calibrate` | Benchmark report; per-scene threshold calibration. |

Backend: the vendored fusion server (`http://127.0.0.1:8788`) is spawned lazily by the
tools — no manual service start needed. The console GUI (same backend) opens via
`dsh/scripts/serve.sh`.

## Protocol

1. **Plan first** (plan mode is available in this mode): route precise-text vs semantic,
   choose signals, budget time/API.
2. **Preflight** the source. `SUSPECT` → inspect the sample frame; `UNREADABLE` → recover
   and re-preflight the recovered file.
3. **Search** via the vr_* tools; always cite engine artifacts
   (`intervals.json` / `matches.json` / annotated frames / `report.html`).
4. **Guardrails**: a cloud or engine failure is never "target absent" — it means switch to
   a cheaper local path or report exactly what was tried; `insufficient` is a verdict, not
   a miss; never claim a hit without the engine's interval/box evidence.

## Self-improvement (recursive)

Confirmed cases feed `vr_evolve` (thresholds via Optuna, confusables/prompts reflection),
gated on a holdout set with rollback snapshots under `dsh/config/rollback/`. The agent may
edit this SKILL.md, `dsh/config/*.json`, and (carefully, after review) the preset
composition itself — those edits are the mode upgrading itself.
