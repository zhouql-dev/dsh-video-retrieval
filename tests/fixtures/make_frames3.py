#!/usr/bin/env python3
"""重新生成 P1-3 的 3 帧 fixture（确定性，无随机数）。

产物 `tests/fixtures/frames3/frame_00{0,1,2}.png` 入库；本脚本只在需要重建时跑。
每帧 320x240：固定渐变背景 + 一个实心矩形（模拟"目标"），矩形位置就是
`fixtures/frames3/boxes.json` 里记录的期望框 —— 契约测试用桩模型直接回这些框，
于是"检测结果 → manifest → crop 文件"三段可以脱开 YOLO 权重做逐字段断言。
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "frames3")
W, H = 320, 240
FPS = 25.0
# 期望框（xyxy，浮点）；三帧轻微右移，模拟连续帧里的同一个目标
BOXES = {
    0: (40.0, 30.0, 120.0, 190.0),
    1: (44.5, 32.0, 124.5, 192.0),
    2: (48.0, 34.0, 128.0, 194.0),
}
COLOR = (30, 60, 200)          # BGR 实心矩形


def main():
    os.makedirs(OUT, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    for idx, (x1, y1, x2, y2) in BOXES.items():
        frame = np.zeros((H, W, 3), np.uint8)
        # 固定渐变：越往右下越亮，纯确定性
        frame[..., 0] = (xx * 255 // W).astype(np.uint8)
        frame[..., 1] = (yy * 255 // H).astype(np.uint8)
        frame[..., 2] = 40
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR, -1)
        path = os.path.join(OUT, f"frame_{idx:03d}.png")
        cv2.imwrite(path, frame)
        print(f"  wrote {path}")
    meta = {"fps": FPS, "width": W, "height": H,
            "frames": [{"idx": i, "t_s": round(i / FPS, 3),
                        "box": list(BOXES[i])} for i in sorted(BOXES)]}
    with open(os.path.join(OUT, "boxes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"  wrote {os.path.join(OUT, 'boxes.json')}")


if __name__ == "__main__":
    main()
