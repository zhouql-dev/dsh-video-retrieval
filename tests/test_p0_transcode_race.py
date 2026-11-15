#!/usr/bin/env python3
"""P0-14 转码缓存竞态：同一 path 两个并发请求不能互踩 tmp。

修法：tmp 名带 pid+uuid；落盘前校验产物可读；失败清理。
判定：两个请求都成功、返回同一 url、目录里没有 *.part.mp4 残留。
"""
import os, shutil, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
os.environ.setdefault("FUSION_CASES", os.path.join(tempfile.mkdtemp(), "c.jsonl"))
os.environ["FUSION_JOBS"] = tempfile.mkdtemp()
os.environ["FUSION_UPLOADS"] = os.environ.get("FUSION_UPLOADS") or tempfile.mkdtemp()

FAILED = []
def check(name, cond, detail=""):
    if cond: print(f"  ok   {name}")
    else: print(f"  FAIL {name} {detail}"); FAILED.append(name)

def main():
    import server as S
    from fastapi.testclient import TestClient
    if shutil.which("ffmpeg") is None:
        print("  skip 没有 ffmpeg"); return 0
    src = os.path.join(S.UPLOAD_DIR, "race_src.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=duration=1:size=64x64:rate=5",
                        "-pix_fmt", "yuv420p", src], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(src):
        print(f"  skip 造测试视频失败: {r.stderr[-120:]}"); return 0

    c = TestClient(S.app)
    H = {"X-Requested-With": "vr-console"}
    out, err = [], []
    def run():
        try:
            resp = c.post("/video/transcode", params={"path": src}, headers=H)
            (out if resp.status_code == 200 else err).append(
                (resp.status_code, resp.json() if resp.status_code == 200 else resp.text[:80]))
        except Exception as e:                      # noqa: BLE001
            err.append(("exc", repr(e)))
    ts = [threading.Thread(target=run) for _ in range(2)]
    for t in ts: t.start()
    for t in ts: t.join()

    check("两个请求都 200", len(out) == 2 and not err, f"ok={len(out)} err={err[:2]}")
    urls = {d.get("url") for _, d in out}
    check("返回同一份缓存 url", len(urls) == 1, str(urls))
    parts = [f for f in os.listdir(S._TRANSCODE_DIR) if f.endswith(".part.mp4")]
    check("没有 .part.mp4 残留", not parts, str(parts[:3]))
    mp4 = [f for f in os.listdir(S._TRANSCODE_DIR) if f.endswith(".mp4")]
    check("产物存在且可读", mp4 and all(os.path.getsize(os.path.join(S._TRANSCODE_DIR, f)) > 0 for f in mp4), str(mp4))

    print()
    if FAILED: print(f"FAIL {len(FAILED)} 项：{FAILED}"); return 1
    print("PASS 全部通过"); return 0

if __name__ == "__main__":
    sys.exit(main())
