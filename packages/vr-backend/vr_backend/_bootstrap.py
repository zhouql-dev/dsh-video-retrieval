#!/usr/bin/env python3
"""P1-8：全包唯一的 sys.path 处理点。

为什么需要它：后端的入口不是一个，而是四类 ——

  * 脚本模式：DSH 插件/独立启动都按**文件路径**拉起 ``server.py``；这时
    ``sys.path[0]`` 是脚本所在目录，包内平铺导入（``import config``）能工作；
  * ``-m`` 模式：bench / agent 的部分入口用模块方式跑；
  * 被导入模式：测试与 server 直接 import 包内模块；
  * 引擎子进程：``engines/*.py`` 是独立脚本，自己有 ``sys.path[0]``。

四类入口各自 ``sys.path.insert`` 必然漂移 —— 历史上的
``engine/provider.py`` vs ``server/provider.py`` 就是"装成哪个版本看路径顺序"。
这里一次性定好顺序，其它文件只写 ``import _bootstrap``：

  1. 包目录（``vr_backend/``）**追加**；
  2. ``engines/`` 追加在**最后** —— 包内的同名模块永远优先，
     引擎目录只提供包内没有的模块（common/matcher/glmv）。

``append`` 而不是 ``insert(0)`` 是刻意的：谁先谁后由这个文件决定，不由调用者的
sys.path 决定。TODO(P1-8 后续)：全量改相对导入后删除本文件。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


def _append(path: str) -> None:
    if path and path not in sys.path:
        sys.path.append(path)


_append(_HERE)                                    # 包内平铺导入
_append(os.path.join(_HERE, "agent"))             # case_log / loop / evolve
_append(os.path.join(_HERE, "bench"))             # metrics / datasets / cycle
try:                                              # 引擎目录（SKILL_SCRIPTS 可覆盖）
    import paths as _paths                        # noqa: E402  （同目录，已在 path 上）

    _append(_paths.engines_dir())
    _ENGINES = _paths.engines_dir()
except Exception:                                 # noqa: BLE001 —— 引擎目录不存在也照常起
    _append(os.path.join(_HERE, "engines"))
    _ENGINES = os.path.join(_HERE, "engines")

__all__ = ["_HERE", "_ENGINES"]
