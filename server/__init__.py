"""Cloud-edge fusion harness (Phase 1 of EXECUTION-GUIDE).

Cloud-edge funnel + evolution layer for surveillance video spatiotemporal
target localization. The package wires together the *cloud candidate
generation* (Qwen-VL via DashScope) and the *local verification* engines
(video-target-localize skill) behind a single task router and a multi-signal
scorer, so a single end-to-end CLI can express the Q1/Q2/Q3 architecture.

Modules
-------
provider  Unified VLM client (VLM_PROVIDER=glm|qwen), mutual fallback,
          env-gated silent degradation. Replicates the four glmv.py contracts:
          refine_query / disambiguate / verify_target / match_image.
router    classify_query(q) -> "semantic" | "precise_text".
signals   §11.2 three disambiguation signals (hue / temporal_curve / vlm_arbiter).
scorer    Multi-signal voting (--agree N).
ground    ground_candidates(video, query) via qwen-vl-max native video;
          returns [] on any failure -> caller falls back to a full scan.
run       End-to-end CLI: router -> ground -> local detect/crop -> signals
          -> score.

The local-only baseline never hard-depends on the cloud: every network call
returns None/[] when no key is set, and the caller degrades to on-device
detection.
"""
