#!/usr/bin/env python3
"""E4 CCPD-GT benchmark logic tests — loader / stats / Layer-1 reflection.

Pure logic only (no OCR, no images): the GT loader is checked against the
real labels.txt (fast, 98,459 lines), accuracy statistics and the
deterministic confusion-pair reflection are exercised on fabricated
(pred, true) pairs. Run: ``python3 test_ccpd_gt.py``.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT != os.sep and not os.path.isdir(os.path.join(_ROOT, "server")):
    _ROOT = os.path.dirname(_ROOT)
_CODE = os.path.join(_ROOT, "server")
_FUSION = _CODE
for _p in (_CODE, os.path.join(_CODE, "agent"), os.path.join(_CODE, "bench"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
