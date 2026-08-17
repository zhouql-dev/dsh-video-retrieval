#!/usr/bin/env python3
"""Pre-flight check for a surveillance video BEFORE running the locator.

Catches the #1 time-sink: a source file the engine cannot actually decode.
Many DVR/NVR exports (.mp4 with magic IMKH, .dav, raw .h264, fragmented mp4)
OPEN in cv2 and return "frames" with plausible brightness but are actually
corrupt (e.g. vertical-stripe decode garbage) or yield zero frames — and the
locator then reports coverage 0, which looks like "target absent" when the real
problem is the file.

Run this first. It:
  1. Inspects magic bytes + ffprobe metadata.
  2. Decodes a spread of frames via cv2 and checks they are real, distinct
     images (not zero, not all-identical, not near-uniform).
  3. Dumps a sample JPG for the operator to eyeball (stripes/garbage are easy
     to see but hard to detect automatically).
  4. If the file is unreadable, runs an escalation ladder and writes a clean,
     cv2-readable mp4 next to the input.

Exit code 0 + prints OK <path>; non-zero + prints a recovery attempt.

Usage (venv python):
  preflight.py <video> [--sample <out.jpg>] [--recover <out.mp4>]
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from common import FFMPEG_HINT
except Exception:  # keep preflight usable standalone
    FFMPEG_HINT = "ffmpeg/ffprobe not found on PATH — install ffmpeg (winget install ffmpeg / brew install ffmpeg)"


def ffprobe(path):
    if not shutil.which("ffprobe"):
        return {"error": FFMPEG_HINT}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate,duration",
             "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", path],
            capture_output=True, text=True).stdout
    except FileNotFoundError:
        return {"error": FFMPEG_HINT}
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); d[k] = v
    return d


def cv2_decode_check(path, sample_path=None, n=7):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "reason": "cv2 cannot open"}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idxs = [int(total * p) for p in (0, .1, .25, .5, .75, .9, .98)] if total > 0 else list(range(n))
    frames = []
    for fi in idxs[:n]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        frames.append((fi, fr))
    cap.release()
    if not frames:
        return {"ok": False, "reason": "cv2 decoded 0 frames", "n_total": total, "fps": fps}
    h, w = frames[0][1].shape[:2]
    if h == 0 or w == 0:
        return {"ok": False, "reason": "zero-size frames"}
    # distinctness: mean absolute diff between sampled frames is far more
    # sensitive than std-of-means — a static-lit surveillance scene can have
    # nearly identical frame *means* (std_across < 1.0) while real motion
    # moves pixels by 10-25/255. Compute both and trust the pixel diff.
    g = [f.astype(np.float32) for _, f in frames]
    diffs = [float(np.abs(g[i] - g[i+1]).mean()) for i in range(len(g) - 1)]
    max_pair_diff = float(max(diffs)) if diffs else 0.0
    means = [float(f.mean()) for f in g]
    std_across = float(np.std(means))
    # per-frame texture (uniform color/flat stripes have low std)
    tex = [float(f.std()) for _, f in frames]
    verdict = "ok"
    flags = []
    if max_pair_diff < 1.5 and std_across < 1.0:
        flags.append("frames-all-identical"); verdict = "suspect"
    if max(tex) < 8.0:
        flags.append("low-texture"); verdict = "suspect"
    if sample_path:
        fi, fr = frames[len(frames) // 2]
        small = cv2.resize(fr, (w // 2, h // 2)) if min(w, h) > 540 else fr
        cv2.imwrite(sample_path, small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return {"ok": verdict == "ok", "verdict": verdict, "flags": flags,
            "n_sampled": len(frames), "n_total": total, "fps": fps,
            "size": f"{w}x{h}", "mean_std_across": round(std_across, 2),
            "max_per_frame_std": round(max(tex), 2), "sample": sample_path}


def try_recover(path, out_mp4):
    """Escalation ladder. Returns path to a readable mp4 or None."""
    base = os.path.splitext(out_mp4)[0]
    if not shutil.which("ffmpeg"):
        print(FFMPEG_HINT)
        return None
    # Stage 1: remux
    if subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                       "-i", path, "-an", "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", "21", "-pix_fmt", "yuv420p", out_mp4]).returncode == 0:
        chk = cv2_decode_check(out_mp4)
        if chk.get("ok"):
            return out_mp4
    # Stage 2: MPEG-PS / proprietary PES demux (IMKH, .dav, raw h264)
    here = os.path.dirname(os.path.abspath(__file__))
    h264 = base + ".h264"
    r = subprocess.run([sys.executable, os.path.join(here, "demux_mpegps.py"),
                        path, h264, "--remux", out_mp4])
    if r.returncode == 0 and os.path.exists(out_mp4):
        chk = cv2_decode_check(out_mp4)
        if chk.get("ok"):
            return out_mp4
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sample", default=None, help="write a sample frame JPG for visual check")
    ap.add_argument("--recover", default=None, help="if unreadable, write a clean mp4 here")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"file not found: {args.video}")

    head = open(args.video, "rb").read(16)
    magic = "mp4(ftyp)" if head[4:8] == b"ftyp" else head[:4].hex() + " " + repr(head[:4])
    probe = ffprobe(args.video)
    dec = cv2_decode_check(args.video, args.sample or None)

    print(f"magic      : {magic}")
    print(f"ffprobe    : {probe}")
    print(f"cv2 decode : {dec}")

    if dec.get("ok"):
        print(f"VERDICT: OK — file is decodable. Use as-is: {args.video}")
        return

    print(f"VERDICT: UNREADABLE/SUSPECT ({', '.join(dec.get('flags') or []) or dec.get('reason')}).")
    print("If --sample looks like real content, the file is fine (texture heuristic is conservative).")
    if dec.get("verdict") == "suspect" and dec.get("ok") is False and not dec.get("reason"):
        # suspect-but-decoded: don't force recover, just warn — UNLESS the flag is
        # frames-all-identical (proprietary DVR/IMKH containers often decode to one
        # frozen frame), in which case --recover is worth attempting.
        print("Treat as readable but verify the sample frame visually before trusting detections.")
        if "frames-all-identical" not in (dec.get("flags") or []):
            return
        if not args.recover:
            print("Hint: re-run with --recover <clean.mp4> to remux/re-encode this IMKH/DVR-style file.")
            return
    if args.recover:
        print(f"Attempting recovery -> {args.recover} ...")
        good = try_recover(args.video, args.recover)
        if good:
            chk = cv2_decode_check(good)
            print(f"RECOVERED -> {good}  ({chk})")
        else:
            sys.exit("Recovery failed. See references/unreadable-video-recovery.md for manual byte-level steps.")


if __name__ == "__main__":
    main()
