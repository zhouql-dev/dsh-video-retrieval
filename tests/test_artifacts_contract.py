#!/usr/bin/env python3
"""P1-3：引擎 ↔ 服务 ↔ 前端的产物契约测试。

历史上两次静默 bug 都落在这条链上：
  * `15b5fab`：matches.json 的框是 xyxy，画框时被当成 xywh → 框画歪；
  * lossless-JSON：工具返回值里带 `undefined` 字段 → harness 直接报
    "value is not lossless JSON"，前端拿不到结果。
两者当时都没有任何测试读这些文件。本套件把五个契约钉住：

  1. boxes 契约：每项 {frame,t_s,box:[x1,y1,x2,y2]}，x2>x1、框在帧内、t_s≈frame/fps
  2. intervals 契约：start_s<=end_s、升序、不重叠；两种 gap 规则都对
  3. crop 契约：文件名 crop_f{idx:05d}_k{k}.jpg 与 manifest 双向一一对应、内容尺寸=框+8px 内边距
  4. tracked-clip：同一份 boxes.json → drawbox xywh，反推回 xyxy 与 GT 的 IoU>0.9
  5. 严格 JSON：产物不得含 NaN/Infinity；JS 侧可选字段必须"有值才挂"，不许挂 undefined

跑法：``python3 tests/test_artifacts_contract.py``（无模型、无网络、无 ffmpeg）。
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures", "frames3")

sys.path[:0] = [os.path.join(ROOT, "server"), os.path.join(ROOT, "engine")]

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


# --------------------------------------------------------------------------- #
# 桩模型：不加载 YOLO 权重，按 fixture 的期望框回框
# --------------------------------------------------------------------------- #
class _Tensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.xyxy.arr)


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


class StubModel:
    """predict() 依次返回 fixture 里每帧的框；额外带一个 area<min_area 的小框，
    用来验证 min_area 过滤（小框不该出现在 manifest 里）。"""

    def __init__(self, per_frame):
        self.per_frame = per_frame
        self.calls = 0

    def predict(self, frame, conf=0.2, imgsz=640, device="mps", verbose=False):
        boxes = self.per_frame[self.calls % len(self.per_frame)]
        self.calls += 1
        return [_Result(boxes, [0.9] * len(boxes))]


def load_fixture():
    meta = json.load(open(os.path.join(FIX, "boxes.json"), encoding="utf-8"))
    frames = []
    import cv2
    for f in meta["frames"]:
        img = cv2.imread(os.path.join(FIX, f"frame_{f['idx']:03d}.png"))
        assert img is not None, f"fixture frame {f['idx']} unreadable"
        frames.append(img)
    return meta, frames


# --------------------------------------------------------------------------- #
# 1 + 3：detect_and_crop 的 manifest 与 crop 文件
# --------------------------------------------------------------------------- #
def test_boxes_and_crops(common, meta, frames, tmpdir):
    print("[1] boxes manifest 契约")
    fps = float(meta["fps"])
    W, H = meta["width"], meta["height"]
    gt = {f["idx"]: f["box"] for f in meta["frames"]}
    # 每帧：GT 框 + 一个 10x10 小框（area=100 < min_area=7200，应被丢弃）
    per_frame = [[list(gt[i]), [5.0, 5.0, 15.0, 15.0]] for i in sorted(gt)]
    model = StubModel(per_frame)
    crops = os.path.join(tmpdir, "crops")
    os.makedirs(crops, exist_ok=True)

    manifest = []
    for idx, frame in enumerate(frames):
        manifest += common.detect_and_crop(model, frame, idx, fps, crops,
                                          min_area=7200, conf=0.2, imgsz=640,
                                          device="cpu")

    check("min_area 过滤：小框不入选", len(manifest) == len(frames),
          f"got {len(manifest)} entries, want {len(frames)}")
    for e in manifest:
        idx = e["frame"]
        x1, y1, x2, y2 = e["box"]
        w = f"f{idx}"
        check(f"{w} 字段齐全", set(e) >= {"frame", "t_s", "box", "area", "conf", "crop"}, str(sorted(e)))
        check(f"{w} box 长度 4", len(e["box"]) == 4, str(e["box"]))
        check(f"{w} x2>x1 且 y2>y1", x2 > x1 and y2 > y1, str(e["box"]))
        check(f"{w} 框在帧内", 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, str(e["box"]))
        check(f"{w} t_s≈frame/fps", abs(e["t_s"] - idx / fps) <= 5e-4,
              f"{e['t_s']} vs {idx / fps}")
        check(f"{w} area=宽*高", abs(e["area"] - (x2 - x1) * (y2 - y1)) <= 1.0,
              f"{e['area']} vs {(x2 - x1) * (y2 - y1)}")
        check(f"{w} 与 GT 同框", [round(v, 1) for v in e["box"]]
              == [round(float(v), 1) for v in gt[idx]], f"{e['box']} vs {gt[idx]}")

    print("[3] crop 文件契约")
    pat = re.compile(r"^crop_f(\d{5})_k(\d+)\.jpg$")
    on_disk = sorted(os.listdir(crops))
    manifest_names = [e["crop"] for e in manifest]
    check("manifest 与磁盘双向一一对应", sorted(manifest_names) == on_disk,
          f"manifest={sorted(manifest_names)} disk={on_disk}")
    check("每个 crop 名匹配 crop_f{idx:05d}_k{k}.jpg",
          all(pat.match(n) for n in on_disk), str(on_disk))
    for e in manifest:
        m = pat.match(e["crop"])
        check(f"{e['crop']} 序号与 frame 一致", m and int(m.group(1)) == e["frame"],
              f"name={e['crop']} frame={e['frame']}")
    # 内容契约：crop 尺寸 = 框 + 8px 内边距（clamp 到帧边界）
    import cv2
    for e, raw in zip(manifest, [gt[i] for i in sorted(gt)]):
        x1, y1, x2, y2 = [float(v) for v in raw]
        want_h = min(H, int(y2 + 8)) - max(0, int(y1 - 8))
        want_w = min(W, int(x2 + 8)) - max(0, int(x1 - 8))
        img = cv2.imread(os.path.join(crops, e["crop"]))
        check(f"{e['crop']} 尺寸 = 框+8px", img is not None and img.shape[:2] == (want_h, want_w),
              f"{None if img is None else img.shape[:2]} vs {(want_h, want_w)}")
        if img is not None:
            # 框中心在 fixture 里是实心矩形色 (30,60,200) BGR
            c = img[img.shape[0] // 2, img.shape[1] // 2]
            check(f"{e['crop']} 内容落在目标上",
                  abs(int(c[0]) - 30) <= 12 and abs(int(c[1]) - 60) <= 12
                  and abs(int(c[2]) - 200) <= 12, f"BGR={list(c)}")
    # 严格 JSON：产物里不许有 NaN/Infinity（JS 侧 JSON.parse 会直接炸）
    try:
        json.dumps(manifest, allow_nan=False)
        ok = True
    except ValueError as exc:
        ok = False
        print(f"        {exc}")
    check("manifest 是严格 JSON（无 NaN/Inf）", ok)


# --------------------------------------------------------------------------- #
# 2：intervals 契约
# --------------------------------------------------------------------------- #
def test_intervals(common, fps=25.0):
    print("[2] intervals 契约")
    check("空输入 → 空列表", common.build_intervals([], fps, gap_s=0) == [])

    cases = [
        ("连续帧规则 gap_s=0", [0, 1, 2, 25, 26, 50], 0, [(0, 2), (25, 26), (50, 50)]),
        ("容忍缺口 gap_s=0.2", [0, 1, 2, 25, 26, 50], 0.2, [(0, 2), (25, 26), (50, 50)]),
        ("同区间内小幅跳帧", [0, 2, 4], 0.2, [(0, 4)]),
        # 边界：gap_s=0.2 @25fps → gap=5 帧，相差 5 帧要合并、6 帧要断开（差一帧就红）
        ("缺口恰好 5 帧仍合并", [0, 5], 0.2, [(0, 5)]),
        ("缺口 6 帧必须断开", [0, 6], 0.2, [(0, 0), (6, 6)]),
        ("单帧", [7], 0, [(7, 7)]),
        ("乱序输入按帧号合并", [26, 0, 25, 1, 2], 0.2, [(0, 2), (25, 26)]),
    ]
    for name, frames, gap_s, want in cases:
        got = common.build_intervals(frames, fps, gap_s=gap_s)
        pairs = [(iv["frames"][0], iv["frames"][1]) for iv in got]
        check(f"{name} · 分组正确", pairs == want, f"{pairs} != {want}")
        check(f"{name} · 字段齐全", all(set(iv) == {"start_s", "end_s", "frames"} for iv in got),
              str(got[:1]))

    # 结构不变量：start<=end、升序、不重叠、时间与帧号一致
    for gap_s in (0, 0.2):
        frames = [0, 1, 2, 5, 9, 10, 40, 41]
        ivs = common.build_intervals(frames, fps, gap_s=gap_s)
        tag = f"gap_s={gap_s}"
        check(f"{tag} · start<=end", all(iv["start_s"] <= iv["end_s"] for iv in ivs), str(ivs))
        check(f"{tag} · 按时间升序",
              all(ivs[i]["start_s"] <= ivs[i + 1]["start_s"] for i in range(len(ivs) - 1)), str(ivs))
        check(f"{tag} · 互不重叠",
              all(ivs[i]["end_s"] <= ivs[i + 1]["start_s"] for i in range(len(ivs) - 1)), str(ivs))
        check(f"{tag} · start_s = frames[0]/fps",
              all(abs(iv["start_s"] - iv["frames"][0] / fps) <= 5e-3 for iv in ivs), str(ivs))
        check(f"{tag} · end_s = frames[1]/fps",
              all(abs(iv["end_s"] - iv["frames"][1] / fps) <= 5e-3 for iv in ivs), str(ivs))


# --------------------------------------------------------------------------- #
# 4：tracked-clip 的画框转换
# --------------------------------------------------------------------------- #
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def test_tracked_clip(server, meta):
    print("[4] tracked-clip 画框契约")
    gt = {f["idx"]: f["box"] for f in meta["frames"]}
    fps = float(meta["fps"])
    with tempfile.TemporaryDirectory() as td:
        # 两个引擎目录各写一份产物：boxes.json 与 matches.json 都要被读到
        os.makedirs(os.path.join(td, "run1"))
        os.makedirs(os.path.join(td, "run2"))
        rows = [{"frame": i, "t_s": round(i / fps, 3), "box": gt[i]} for i in sorted(gt)]
        json.dump(rows[:2], open(os.path.join(td, "run1", "boxes.json"), "w"))
        # run2 里：一条重复（应去重）、一条窗口外（应过滤）
        json.dump(rows[1:] + [{"frame": 90, "t_s": 3.6, "box": [1, 1, 9, 9]}],
                  open(os.path.join(td, "run2", "matches.json"), "w"))
        open(os.path.join(td, "run1", "matches.json"), "w").write("{ not json")   # 半截文件

        boxes = server._collect_boxes(td, 0.0, 0.2)
        check("跨目录收集 + 去重 + 窗口过滤", [(f, [round(v, 1) for v in b]) for f, b in boxes]
              == [(i, [round(float(v), 1) for v in gt[i]]) for i in sorted(gt)], str(boxes))
        check("坏 JSON 不致命", len(boxes) == 3, f"got {len(boxes)}")

        for floor_f in (0, 6):
            filters = server._drawbox_filters(boxes, floor_f)
            check(f"floor_f={floor_f} 每个框一条 filter", len(filters) == len(boxes), str(filters))
            for (fr, gtb), flt in zip(boxes, filters):
                m = re.match(r"drawbox=x=(-?\d+):y=(-?\d+):w=(\d+):h=(\d+):.*"
                             r"between\(n,(-?\d+),(-?\d+)\)", flt)
                check(f"f{fr} filter 可解析", m is not None, flt)
                if not m:
                    continue
                x, y, w, h = (int(m.group(i)) for i in (1, 2, 3, 4))
                drawn = (x, y, x + w, y + h)
                iou = _iou(drawn, [float(v) for v in gtb])
                check(f"f{fr} 画出的框与 GT IoU>0.9", iou > 0.9, f"IoU={iou:.3f} drawn={drawn} gt={gtb}")
                check(f"f{fr} drawbox 宽高 = x2-x1/y2-y1",
                      w == max(0, int(gtb[2] - gtb[0])) and h == max(0, int(gtb[3] - gtb[1])),
                      f"w={w} h={h} vs {gtb}")
                n_lo, n_hi = int(m.group(5)), int(m.group(6))
                check(f"f{fr} 帧号按 seek 平移", n_lo == max(0, fr - floor_f), f"n_lo={n_lo}")

        check("空 boxes → vf=null（不画框也能出片）",
              ",".join(server._drawbox_filters([], 0)) == "")


# --------------------------------------------------------------------------- #
# 5：undefined / 非严格 JSON 的输出边界
# --------------------------------------------------------------------------- #
def test_json_boundary(server, meta, manifest):
    print("[5] 输出边界（undefined / 严格 JSON）")
    # (a) JS 语义：对象里的 undefined 字段被 JSON.stringify 丢掉，
    #     但 harness 的 lossless-JSON 边界会报错 —— 所以只能"有值才挂"。
    js = ("const a={verdict:'OK',ok:true,recovered:undefined};"
          "const b={verdict:'OK',ok:true};"
          "if(a.recovered)b.recovered=a.recovered;"
          "process.stdout.write(JSON.stringify(a)+'|'+JSON.stringify(b))")
    try:
        out = subprocess.run(["node", "--input-type=module", "-e", js],
                             capture_output=True, text=True, timeout=30)
        got = (out.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        got = ""
    check("node: undefined 字段不进 JSON 字符串",
          got == '{"verdict":"OK","ok":true}|{"verdict":"OK","ok":true}', got or (out.stderr or "")[:200])

    # (b) 源码守卫：可选字段必须是条件挂载（不是先挂 undefined 再删）
    src_path = os.path.join(ROOT, "src", "index.js")
    src = open(src_path, encoding="utf-8").read() if os.path.exists(src_path) else ""
    check("插件源码：recovered 条件挂载",
          "if (recovered) result.recovered = recovered" in src,
          "guard line missing")
    check("插件源码：结果对象里没有 `key: undefined` 字面量",
          not re.search(r"[,{]\s*\w+\s*:\s*undefined\s*[,}]", src),
          "found `key: undefined` in a returned literal")

    # (c) 后端产物必须是严格 JSON：NaN/Infinity 会让前端 JSON.parse 直接抛
    bad = json.dumps(manifest, allow_nan=False)
    check("服务端产物可被 allow_nan=False 序列化", "NaN" not in bad and "Infinity" not in bad)

    # (d) 写盘路径用 json.dump 默认 allow_nan=True —— 钉住产物里确实没有非有限值
    def walk(o, path="root"):
        if isinstance(o, float):
            check(f"{path} 有限", np.isfinite(o), str(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(manifest)


def main():
    import common
    import server as S

    meta, frames = load_fixture()
    with tempfile.TemporaryDirectory() as td:
        test_boxes_and_crops(common, meta, frames, td)
    test_intervals(common)
    test_tracked_clip(S, meta)
    with tempfile.TemporaryDirectory() as td:
        crops = os.path.join(td, "crops")
        os.makedirs(crops, exist_ok=True)
        gt = {f["idx"]: f["box"] for f in meta["frames"]}
        model = StubModel([[list(gt[i])] for i in sorted(gt)])
        manifest = []
        for idx, frame in enumerate(frames):
            manifest += common.detect_and_crop(model, frame, idx, float(meta["fps"]),
                                               crops, min_area=7200, device="cpu")
    test_json_boundary(S, meta, manifest)

    print()
    if FAILED:
        print(f"FAIL {len(FAILED)} 项：{FAILED[:6]}")
        return 1
    print("PASS 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
