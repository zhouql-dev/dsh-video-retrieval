#!/usr/bin/env python3
"""Last-resort recovery: carve a clean Annex-B H.264 stream out of a proprietary
DVR/surveillance container that ffmpeg/cv2 cannot demux.

Symptom that calls for this: `ffprobe` reports width=0 / duration=N/A and cv2
returns zero frames (or striped garbage), yet the file is large and the magic
bytes are NOT `ftyp` (real MP4). Common proprietary wrappers seen in the wild:
IMKH (MPEG-Program-Stream), Dahua .dav, Hikvision exports, raw H.264 streams.

Approach: walk 3-byte start codes (00 00 01) and interpret MPEG-PS packetization:
  0xBA pack header, 0xBB system header, 0xBD/0xC0..0xEF PES packets.
Extract ONLY the video elementary stream (PES id 0xE0) payloads, strip each PES
header, concatenate the (already Annex-B) H.264 payload. This drops audio / pack
/ system data. Pipe the result through ffmpeg to a standard mp4.

A naive "keep every NAL with a valid type" byte-carve is NOT enough: it injects
audio/pack garbage as spurious slice NALs and decodes to striped garbage. Proper
PES-payload extraction is required.

Usage:
  demux_mpegps.py <input> <output_h264>            # just carve the .h264
  demux_mpegps.py <input> <output_h264> --remux <out.mp4>   # carve + ffmpeg -> mp4
"""
from __future__ import annotations
import argparse, os, shutil, struct, subprocess, sys


def carve(src: str, dst: str) -> dict:
    data = open(src, "rb").read()
    n = len(data)
    out = bytearray()
    n_pes = n_vid = vid_bytes = 0
    saw = {}

    def is_sc(i):
        return i + 2 < n and data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1

    i = 0
    while i < n - 3:
        if is_sc(i):
            sid = data[i + 3]
            saw[sid] = saw.get(sid, 0) + 1
            if sid == 0xBA:  # pack header: advance to next start code
                j = i + 4
                while j < n - 3 and not is_sc(j):
                    j += 1
                i = j; continue
            if sid == 0xBB:  # system header: 2-byte length
                if i + 5 < n:
                    i = i + 6 + struct.unpack(">H", data[i + 4:i + 6])[0]
                else:
                    i += 4
                continue
            if 0xC0 <= sid <= 0xEF or sid == 0xBD:  # PES packet
                n_pes += 1
                if i + 5 >= n:
                    i += 4; continue
                length = struct.unpack(">H", data[i + 4:i + 6])[0]
                pes_start = i + 6
                pes_end = pes_start + length if length else _next_sc(data, n, pes_start)
                pes_end = min(pes_end, n)
                if sid == 0xE0 and pes_end - pes_start >= 3:  # video ES
                    n_vid += 1
                    hl = data[pes_start + 2]  # PES_header_data_length
                    payload_start = pes_start + 3 + hl
                    out += data[payload_start:pes_end]
                    vid_bytes += pes_end - payload_start
                i = pes_end; continue
            j = i + 4
            while j < n - 3 and not is_sc(j):
                j += 1
            i = j; continue
        i += 1

    open(dst, "wb").write(bytes(out))
    return {"input_bytes": n, "h264_bytes": len(out), "pes_packets": n_pes,
            "video_pes": n_vid, "stream_ids": {f"0x{k:02X}": v for k, v in sorted(saw.items())}}


def _next_sc(data, n, j):
    while j < n - 3 and not (data[j] == 0 and data[j + 1] == 0 and data[j + 2] == 1):
        j += 1
    return j


def remux(h264: str, mp4: str, fps: float = 25.0) -> bool:
    if not shutil.which("ffmpeg"):
        print("ffmpeg 未找到 / not found on PATH — 安装/install: winget install ffmpeg (Windows) "
              "或/or brew install ffmpeg (Mac). 原始 .h264 已写出 / raw .h264 was still written.")
        return False
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-fflags", "+genpts+discardcorrupt", "-err_detect", "ignore_err",
           "-r", str(fps), "-i", h264, "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
    return subprocess.run(cmd).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output_h264")
    ap.add_argument("--remux", metavar="MP4", help="also transcode to a cv2-readable mp4")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    info = carve(args.input, args.output_h264)
    print(info)
    if info["video_pes"] == 0:
        print("WARNING: no video PES (0xE0) packets found — this may not be MPEG-PS; "
              "inspect stream_ids and magic bytes; see references/unreadable-video-recovery.md")
    if args.remux:
        ok = remux(args.output_h264, args.remux, args.fps)
        print(f"remux -> {args.remux}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
