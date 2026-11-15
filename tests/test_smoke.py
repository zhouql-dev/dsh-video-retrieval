#!/usr/bin/env python3
"""No-key, no-model unit smoke tests for the fusion harness.

Validates the pure-logic layer of every Phase-1 module so the cloud-edge
contracts can be checked without DASHSCOPE/ZHIPUAI keys, torch, ultralytics, or
tesseract installed. Run: ``python3 test_smoke.py`` (exit code 0 = all pass).

What this covers
----------------
router    plate / person / ID / semantic routing + target extraction.
signals   hue same-vs-different, temporal bell-vs-spike, vlm abstention.
scorer    agree-rule, abstention shrinks the denominator, ``insufficient``.
ground    JSON segment parsing + window->frame ranges (empty => full scan).
provider  env-gating: no key -> chat/refine_query return None; status honest;
          oxalpha routing (active-first try-order + cross-family fallback).
"""
from __future__ import annotations
import os
import sys

import numpy as np

_HERE_FILE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE_FILE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE_FILE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_HERE = _CODE          # 兼容：旧用例把 _HERE 当"代码目录"用（如 subprocess 的 cwd）

import router
import signals as S
import scorer
import ground as G
import provider

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def swatch(h_deg, s=160, v=180):
    import cv2
    h = int(h_deg / 2)
    return cv2.cvtColor(np.full((40, 40, 3), [h, s, v], np.uint8), cv2.COLOR_HSV2BGR)


# --------------------------------------------------------------------------- #
def test_router():
    print("[router]")
    r = router.classify_query("车牌号 京Q1G728 的小轿车")
    check("plate -> unsupported（预算红线后的产品口径）", r["branch"] == "unsupported", r)
    check("plate target extracted", r["target"] == "Q1G728", r)
    r = router.classify_query("a woman in a pink coat and white pants")
    check("person -> semantic", r["branch"] == "semantic", r)
    r = router.classify_query("身份证 110101199003071234 的男子")
    check("idcard -> unsupported（预算红线后的产品口径）", r["branch"] == "unsupported", r)
    check("idcard target", r["target"] == "110101199003071234", r)
    r = router.classify_query("a red bicycle parked near the gate")
    check("bicycle -> semantic", r["branch"] == "semantic", r)
    check("bare plate Q1G728", router.classify_query("find plate Q1G728")["target"] == "Q1G728")
    check("empty query safe", router.classify_query("")["branch"] == "semantic")


def test_signals():
    print("[signals]")
    pink, mag, cyan = swatch(330), swatch(300), swatch(180)
    h_same = S.hue_consistency(pink, pink)
    h_diff = S.hue_consistency(pink, cyan)
    check("hue same score=1.0", h_same["score"] == 1.0, h_same)
    check("hue diff score low", h_diff["score"] <= 0.1, h_diff)
    check("hue same dhue small", h_same["raw"]["dhue"] <= 35.0)
    check("hue diff dhue large", h_diff["raw"]["dhue"] >= 60.0)
    # low-color crop abstains (neutral 0.5, colorful=False)
    gray = np.full((40, 40, 3), 200, np.uint8)
    h_gray = S.hue_consistency(pink, gray)
    check("hue low-color neutral", h_gray["score"] == 0.5 and h_gray["raw"]["colorful"] is False, h_gray)

    bell = S.temporal_curve([(0, .35), (1, .55), (2, .83), (3, .55), (4, .35)])
    spike = S.temporal_curve([(0, .30), (1, .83), (2, .32)])
    check("bell beats spike", bell["score"] > spike["score"], (bell, spike))
    check("bell span>=0.5", bell["raw"]["span_s"] >= 0.5)
    check("spike span=0", spike["raw"]["span_s"] == 0.0, spike)
    check("too few -> low score", S.temporal_curve([(0, .9)])["score"] <= 0.4)

    # No VLM key present in this test env -> arbiter abstains.
    v = S.vlm_arbiter(pink, pink)
    check("vlm abstains w/o key", v["raw"]["match"] == "unavailable" and v["score"] == 0.0, v)


def test_scorer():
    print("[scorer]")
    pink = swatch(330)
    # genuine hit: hue + temporal fire, vlm abstains -> still a hit at agree=2.
    hit = {"id": "t438", "results": {
        "hue": S.hue_consistency(pink, swatch(328)),
        "temporal": S.temporal_curve([(0, .35), (1, .55), (2, .83), (3, .55), (4, .35)]),
        "vlm": {"score": 0.0, "raw": {"match": "unavailable"}, "evidence": "no key"}}}
    v = scorer.score_candidate(hit["results"], agree=2)
    check("hit w/ vlm abstain", v["verdict"] == "hit", v)
    check("vlm abstained", "vlm" in v["abstained"], v)
    check("2 eligible, both fire", set(v["fired"]) == {"hue", "temporal"}, v)

    # false positive: nothing fires.
    fp = {"id": "t120", "results": {
        "hue": S.hue_consistency(pink, swatch(120)),
        "temporal": S.temporal_curve([(0, .30), (1, .83), (2, .31)]),
        "vlm": {"score": 0.0, "raw": {"match": "no"}, "evidence": "no"}}}
    v = scorer.score_candidate(fp["results"], agree=2)
    check("false positive -> miss", v["verdict"] == "miss", v)

    # insufficient: only one non-abstaining signal, agree=2.
    thin = {"id": "x", "results": {
        "hue": S.hue_consistency(pink, gray := np.full((40, 40, 3), 200, np.uint8)),  # abstain
        "temporal": S.temporal_curve([(0, .35), (1, .83), (2, .35)]),
        "vlm": {"score": 0.0, "raw": {"match": "unavailable"}, "evidence": "no"}}}
    v = scorer.score_candidate(thin["results"], agree=2)
    check("insufficient verdict", v["verdict"] == "insufficient", v)

    # agree=1 is permissive: a single fire hits.
    one = {"id": "y", "results": {
        "hue": S.hue_consistency(pink, swatch(330)),
        "vlm": {"score": 0.0, "raw": {"match": "no"}, "evidence": "no"}}}
    check("agree=1 single-fire hits",
          scorer.score_candidate(one["results"], agree=1)["verdict"] == "hit")

    # score_all sorts by score desc.
    ranked = scorer.score_all([{"id": "fp", "results": fp["results"]},
                               {"id": "hit", "results": hit["results"]}], agree=2)
    check("score_all sorts hit first", ranked[0]["id"] == "hit", [r["id"] for r in ranked])


def test_ground():
    print("[ground]")
    reply = ('{"events":[{"start_time":"00:09:40","end_time":"00:09:50","confidence":0.9,'
             '"event":"穿粉色外套女性"},{"start_time":"00:12:38","end_time":"00:12:40",'
             '"confidence":0.5,"event":"疑似"}]}')
    segs = G._parse_segments(reply, "粉色")
    check("parsed 2 segments", len(segs) == 2, segs)
    check("HH:MM:SS -> seconds", segs[0]["start_s"] == 580.0 and segs[0]["end_s"] == 590.0, segs[0])
    check("confidence -> score", segs[0]["score"] == 0.9)
    check("bare-list parse", len(G._parse_segments('[{"start_time":"00:00:10","end_time":"00:00:12"}]', "")) == 1)
    check("garbage -> []", G._parse_segments("the target is not present", "") == [])
    rng = G.windows_to_frames([{"start_s": 580, "end_s": 600}], fps=25, total_frames=22566, pad_s=1.0)
    check("window padded+clamped", rng == [(14475, 15025)], rng)
    full = G.windows_to_frames([], fps=25, total_frames=22566)
    check("empty -> full scan", full == [(0, 22565)], full)
    url, err = G._video_url("https://example.com/a.mp4", 64)
    check("http url passthrough", url == "https://example.com/a.mp4" and err is None, (url, err))
    url, err = G._video_url("/no/such/file.mp4", 64)
    check("missing local file -> None", url is None and err, (url, err))


def test_provider():
    print("[provider]")
    # Keys are unset in this test env -> everything must degrade to None/[].
    st = provider.provider_status()
    check("no glm key reported", st["glm"] is False, st)
    check("no qwen key reported", st["qwen"] is False, st)
    check("no oxalpha key reported", st["oxalpha"] is False, st)
    check("refine_query None w/o key", provider.refine_query("a red car") is None)
    check("chat None w/o key", provider.chat([{"role": "user", "content": "hi"}]) is None)
    check("verify_target None w/o key",
          provider.verify_target("a plate", swatch(0)) is None)
    check("match_image None w/o key",
          provider.match_image(swatch(0), swatch(0)) is None)
    check("qwen_chat None w/o key", provider.qwen_chat([{"role": "user", "content": "hi"}]) is None)
    check("ox_chat None w/o key", provider.ox_chat([{"role": "user", "content": "hi"}]) is None)
    check("ground returns [] w/o key", G.ground_candidates("https://x/y.mp4", "a person") == [])


def test_provider_ox_routing():
    """With a fake oxalpha+glm key and a scripted _post: active-first
    try-order, cross-provider fallback on failure, model substitution."""
    print("[provider-ox]")
    calls = []

    def fake_post(base_url, key, model, messages, max_tokens, timeout, temperature):
        calls.append((base_url, model))
        if "openrouter" in base_url:
            return None                     # ox endpoint fails (e.g. trial over)
        return "ok"

    real_post = provider._post
    provider._post = fake_post
    os.environ["OXALPHA_API_KEY"] = "fake-ox-key"
    os.environ["ZHIPUAI_API_KEY"] = "fake-glm-key"
    try:
        provider.VLM_PROVIDER = "oxalpha"
        order = provider._order()
        check("ox first when active", order[0] == "oxalpha" and len(order) == 3, order)
        out = provider.chat([{"role": "user", "content": "hi"}])
        check("fallback answers after ox fails", out == "ok", out)
        check("tried openrouter then bigmodel",
              "openrouter" in calls[0][0] and "bigmodel" in calls[1][0], calls[:2])
        check("last_provider glm after fallback",
              provider.last_provider() == "glm", provider.last_provider())
        # Explicit glm model name on the ox channel must be substituted.
        calls.clear()
        provider._post = lambda b, k, m, *a, **kw: (calls.append((b, m)), "ok")[1]
        provider.refine_query("a red car")
        check("glm model name substituted on ox",
              calls and "stealth/ox-alpha" in calls[0][1], calls[:1])
        # Model-family guard.
        check("stealth/ known to ox",
              provider._model_known("stealth/ox-alpha", "oxalpha"), "")
        check("stealth/ foreign to glm/qwen",
              not provider._model_known("stealth/ox-alpha", "glm")
              and not provider._model_known("stealth/ox-alpha", "qwen"), "")
        check("glm name native to glm, foreign to ox",
              provider._model_known("glm-4v-flash", "glm")
              and not provider._model_known("glm-5.1", "oxalpha"), "")
    finally:
        provider._post = real_post
        os.environ.pop("OXALPHA_API_KEY", None)
        os.environ.pop("ZHIPUAI_API_KEY", None)
        provider.VLM_PROVIDER = (os.environ.get("VLM_PROVIDER") or "glm").lower()


def test_ground_ox_frames():
    """The ox grounding path builds a timestamped multi-image message; any
    failure degrades to []. Frames and the ox client are faked, no network."""
    print("[ground-ox]")
    # Pin the channel decision independent of the developer's ambient env.
    saved = {k: os.environ.pop(k, None)
             for k in ("DASHSCOPE_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY",
                       "VLM_PROVIDER")}
    real_vlm_provider = provider.VLM_PROVIDER
    os.environ["OXALPHA_API_KEY"] = "fake-ox-key"
    try:
        provider.VLM_PROVIDER = "oxalpha"
        # Frames faked; the ox-only routing clause (no DashScope key + ox key)
        # selects _ground_via_ox even though VLM_PROVIDER stays at its default.
        real_sample = G._sample_frames_b64
        G._sample_frames_b64 = lambda v, fps, mx, max_side=448: [
            (i * 2.0, f"data:image/jpeg;base64,f{i}") for i in range(3)]
        sent = {}

        def fake_ox_chat(messages, max_tokens=4096, timeout=120, model=None,
                         temperature=0.1):
            sent["content"] = messages[0]["content"]
            sent["model"] = model
            return ('{"events":[{"start_time":"00:00:02","end_time":"00:00:04",'
                    '"confidence":0.8,"event":"目标出现"}]}')

        import provider as P
        real_ox_chat = P.ox_chat
        P.ox_chat = fake_ox_chat
        real_dur = G._video_duration_s
        G._video_duration_s = lambda v: 6.0
        try:
            segs = G.ground_candidates("/fake/v.mp4", "红衣行人", model=None)
            content = sent["content"]
            texts = [c.get("text") or "" for c in content if c.get("type") == "text"]
            imgs = [c for c in content if c.get("type") == "image_url"]
            check("3 timestamped frame items sent", len(imgs) == 3, len(imgs))
            check("frame stamps in text items",
                  any("[t=" in t for t in texts), texts)
            check("prompt mentions relative seconds",
                  any("相对秒数" in t for t in texts), "")
            check("parsed window from reply",
                  len(segs) == 1 and segs[0]["start_s"] == 2.0
                  and segs[0]["end_s"] == 4.0, segs)
            # Failure of the ox call -> [] (degrade contract).
            P.ox_chat = lambda *a, **kw: None
            check("no ox reply -> []", G.ground_candidates("/fake/v.mp4", "x") == [])
            G._sample_frames_b64 = lambda *a, **kw: []
            check("no sampled frames -> []", G.ground_candidates("/fake/v.mp4", "x") == [])
        finally:
            P.ox_chat = real_ox_chat
            G._video_duration_s = real_dur
            G._sample_frames_b64 = real_sample
    finally:
        os.environ.pop("OXALPHA_API_KEY", None)
        provider.VLM_PROVIDER = real_vlm_provider
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    test_router()
    test_signals()
    test_scorer()
    test_ground()
    test_provider()
    test_provider_ox_routing()
    test_ground_ox_frames()
    print(f"\n{'='*40}\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
