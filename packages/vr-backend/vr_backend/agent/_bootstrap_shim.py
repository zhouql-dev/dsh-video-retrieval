#!/usr/bin/env python3
"""脚本模式垫片（P1-8）：agent/ 在 sys.path[0] 时，先让上层可导入，再装上它。

全包只有三处 sys.path.insert —— 这里两处 + 上层的 _bootstrap.py 一处。
之所以不是零：``python bench/run_prw.py`` 的 sys.path[0] 是 bench/，
没有任何办法在不改 sys.path 的前提下跳到上一层。（改 ``-m`` 跑法也可以，
但入口是 JS/服务按文件路径拉起的，那才是真正的改动面。）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
