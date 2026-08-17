#!/usr/bin/env python3
"""The §11.2 multi-signal disambiguation — productized.

The first eval report (人物检索) found that *neither* a single channel is
trustworthy on its own at low resolution:

  * OSNet @0.55 over-matches: 16 "exact" candidates, **15 false positives**.
  * The VLM confidently hallucinates ("is this the same person?" → all yes).

§11.2 of the research plan resolves this by demanding **three independent
signals agree** before a candidate is judged a hit. Each signal here returns

    {"score": float[0,1], "evidence": str, "raw": {...}}

``score`` is normalized so *higher = more consistent with the target*; ``raw``
carries the signal-specific measurements. The *firing thresholds* and the
agreement count live in ``scorer.py`` (they are the Phase-2 Optuna targets —
§11.2 cites Δhue≈35°, REID 0.55→0.65+; those defaults are encoded there, not
here, so a sweep can move them without touching the signal math).

Signals
-------
hue_consistency  Clothing-color hue agreement (HSV Δhue). Same person ≲35°;
                 false positives 60–110°.
temporal_curve   Shape of the similarity-over-time curve. A real subject shows
                 a smooth 0.35→0.83→0.35 rise-and-fall; a single-frame spike is
                 noise.
vlm_arbiter      Image-to-image VLM final review (provider.match_image) — the
                 decisive signal that confirmed the single true hit at t=438s
                 (conf=1.0) among 36 candidates.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Clothing-color hue consistency
# --------------------------------------------------------------------------- #

def _dominant_hue(bgr) -> tuple[Optional[float], float]:
    """Dominant hue (degrees, 0–360) over *colorful* pixels only, plus the
    fraction of pixels that were colorful. Low-saturation / near-black /
    near-white pixels are excluded because their hue is ill-defined (a gray
    jacket must not be judged "consistent" with a pink one by chance)."""
    import cv2
    if bgr is None or bgr.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s >= 40) & (v >= 40) & (v <= 235)
    frac = float(mask.mean())
    if frac < 0.05:                       # <5% colorful -> hue unreliable
        return None, frac
    hue_deg = h[mask].astype(np.float32) * 2.0     # OpenCV H is [0,180)
    # 36 bins of 10°, smoothed, take the peak bin's circular center.
    hist, _ = np.histogram(hue_deg, bins=36, range=(0.0, 360.0))
    hist = hist.astype(np.float32)
    hist = np.convolve(hist, np.ones(3) / 3.0, mode="same")  # 3-bin smooth
    peak = int(np.argmax(hist))
    return float(peak * 10.0 + 5.0), frac


def _circ_delta(a: float, b: float) -> float:
    """Circular hue difference in degrees, in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def hue_consistency(ref_bgr, crop_bgr) -> dict:
    """HSV hue agreement of the reference vs a candidate crop.

    score: 1.0 at Δhue=0°, 0.5 at ~35°, 0.0 at ≥110° (linear ramp, clamped).
    If either side is low-color (gray/dark), hue can't discriminate → score
    0.5 (neutral), ``raw.colorful=False`` so the scorer knows not to count it
    as an independent confirmation."""
    rh, rf = _dominant_hue(ref_bgr)
    ch, cf = _dominant_hue(crop_bgr)
    if rh is None or ch is None:
        return {"score": 0.5, "raw": {"dhue": None, "ref_hue": rh, "crop_hue": ch,
                                      "ref_colorful_frac": rf, "crop_colorful_frac": cf,
                                      "colorful": False},
                "evidence": "low-color crop; hue not discriminative"}
    dh = _circ_delta(rh, ch)
    # Linear ramp: 1.0@0, 0.5@35, 0.0@110.
    score = float(np.clip(1.0 - (dh - 0.0) / 110.0, 0.0, 1.0))
    verdict = "same" if dh <= 35.0 else ("plausible" if dh <= 60.0 else "different")
    return {"score": round(score, 3),
            "raw": {"dhue": round(dh, 1), "ref_hue": round(rh, 1),
                    "crop_hue": round(ch, 1), "ref_colorful_frac": round(rf, 3),
                    "crop_colorful_frac": round(cf, 3), "colorful": True},
            "evidence": f"Δhue={dh:.0f}° ({verdict}); ref_h≈{rh:.0f}° crop_h≈{ch:.0f}°"}


# --------------------------------------------------------------------------- #
# 2. Temporal similarity-curve shape
# --------------------------------------------------------------------------- #

def temporal_curve(samples) -> dict:
    """Score the *shape* of a candidate's similarity-over-time curve.

    ``samples`` is an iterable of ``(t_s, similarity)`` around the candidate
    (e.g. the OSNet cosine of the reference vs the same track across nearby
    frames). A genuine subject produces a smooth rise-and-fall
    (0.35→0.83→0.35); a stray detection is a single-frame spike.

    score = 0.45·peak_norm + 0.55·bellness, where bellness is the correlation
    of the curve with a triangular template centered on the peak (penalizes
    spikes). ``raw`` exposes peak, FWHM-like span, and bellness so the scorer's
    fire-rule (peak≥X AND bellness≥Y) can move independently."""
    pts = sorted([(float(t), float(s)) for t, s in samples], key=lambda p: p[0])
    ts = np.array([p[0] for p in pts], dtype=np.float64)
    sims = np.array([p[1] for p in pts], dtype=np.float64)
    if len(pts) < 3:
        return {"score": 0.3, "raw": {"peak": float(sims.max()) if sims.size else 0.0,
                                      "span_s": 0.0, "bellness": 0.0, "n": len(pts)},
                "evidence": f"too few samples ({len(pts)}); curve shape unjudgable"}
    peak = float(sims.max())
    pi = int(np.argmax(sims))
    baseline = float(np.percentile(sims, 30))      # the ~0.35 floor in the report
    rise = peak - baseline
    # Width (seconds) where sim ≥ floor(0.45) — proxy for FWHM given the
    # report's 0.35 baseline. Only the contiguous span around the peak counts.
    floor = 0.45
    above = np.where(sims >= floor)[0]
    if above.size:
        lo, hi = above[above <= pi].min(initial=pi), above[above >= pi].max(initial=pi)
        span_s = float(ts[hi] - ts[lo])
    else:
        span_s = 0.0
    # bellness = how much of the window is *sustainedly* elevated above the
    # noise floor (rewards a gradual climb over several frames — the report's
    # 0.35→0.83→0.35 with elevated shoulders; penalizes a lone single-frame
    # spike, which is exactly the report's noise pattern). 0.3·rise is chosen
    # so the 0.55 shoulders of a real climb clear it while a one-frame 0.83
    # spike (baseline≈0.31) does not.
    mid_level = baseline + 0.3 * rise
    n_above = int(np.sum(sims >= mid_level))
    bellness = (n_above - 1) / max(1, len(sims) - 1)
    bellness = max(0.0, min(1.0, bellness))
    peak_norm = min(1.0, peak / 0.8)
    score = round(0.45 * peak_norm + 0.55 * bellness, 3)
    note = "smooth rise/fall" if bellness >= 0.4 and span_s >= 0.5 else (
           "single-frame spike" if span_s < 0.5 else "weak curve")
    return {"score": score,
            "raw": {"peak": round(peak, 3), "span_s": round(span_s, 2),
                    "bellness": round(bellness, 3), "n": len(pts)},
            "evidence": f"peak={peak:.2f} bellness={bellness:.2f} span={span_s:.1f}s ({note})"}


# --------------------------------------------------------------------------- #
# 3. VLM image-to-image final review (the decisive arbiter)
# --------------------------------------------------------------------------- #

_VLM_SCORE = {"exact": 1.0, "partial": 0.6, "unclear": 0.3, "no": 0.0}


def vlm_arbiter(ref_bgr, crop_bgr, max_tokens: int = 2048) -> dict:
    """Image-to-image identity check via the provider (provider.match_image).

    This is the §11.2 终审 signal: in the field it was the *only* channel that
    could lift the single true hit (t≈438s, conf=1.0) out of 36 embedding
    candidates. score maps exact/partial/unclear/no → 1.0/0.6/0.3/0.0; when no
    VLM provider is available it returns score 0.0 and ``raw.match="unavailable"``
    so the scorer knows it abstains (does not count toward agreement)."""
    from provider import match_image            # local import: keeps cv2 lazy
    r = match_image(ref_bgr, crop_bgr, max_tokens=max_tokens)
    if not r:
        return {"score": 0.0, "raw": {"match": "unavailable", "conf": 0.0},
                "evidence": "VLM provider unavailable (no key / failed) — abstains"}
    label = str(r.get("match", "unclear")).lower()
    if label not in _VLM_SCORE:
        label = "unclear"
    conf = float(r.get("conf", 0.0) or 0.0)
    score = round(_VLM_SCORE[label], 3)
    return {"score": score, "raw": {"match": label, "conf": round(conf, 3)},
            "evidence": f"VLM {label} conf={conf:.2f} | {str(r.get('note', ''))[:80]}"}


if __name__ == "__main__":
    # Synthetic demo: two same-hue swatches vs a different hue, with no VLM key.
    import numpy as np
    def swatch(h_deg, s=160, v=180):
        import cv2
        h = int(h_deg / 2)
        return cv2.cvtColor(np.full((40, 40, 3), [h, s, v], np.uint8), cv2.COLOR_HSV2BGR)
    pink, magenta, cyan = swatch(330), swatch(300), swatch(180)
    print("hue pink~pink :", hue_consistency(pink, pink))
    print("hue pink~mag  :", hue_consistency(pink, magenta))
    print("hue pink~cyan :", hue_consistency(pink, cyan))
    print("temporal bell :", temporal_curve([(0, .35), (1, .55), (2, .83), (3, .55), (4, .35)]))
    print("temporal spike:", temporal_curve([(0, .30), (1, .83), (2, .32)]))
    print("vlm (no key)  :", vlm_arbiter(pink, pink))
