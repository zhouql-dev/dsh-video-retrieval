#!/usr/bin/env python3
"""Tool contract layer — the deterministic core wrapped as agent-callable tools.

Every engine and judge in the system is registered here with a declarative
contract: what it does, when to use it, its inputs/outputs, its **cost tags**
(so an agent can budget: cheap local first, cloud/long scans later), what it
requires (keys/models/tesseract), and what to fall back to. An agent reads
``agent_context()`` to plan, then calls ``dispatch(tool, params, workdir)`` to
execute — always getting a uniform envelope back:

    {"tool", "status": ok|failed|skipped, "summary", "outputs", "error", "cost"}

Design rules (mirror the whole system's philosophy):
  * **Uniform envelope** — the agent never parses raw stdout; it gets structured
    verdicts + pointers to the engines' JSON outputs (metrics.json, hits.json…).
  * **Never claim "absent"** — a failed/insufficient run reports what was tried
    and what is missing, not "target not found".
  * **Cheap in-process judges** — ``router`` (classify a query) and ``rescoring``
    (re-judge cached candidates under new thresholds) run in microseconds, so
    the agent can deliberate by re-scoring instead of re-running engines.

Heavy engines run as subprocesses (same invocation an operator would use) and
are imported nowhere — this module stays importable without torch/ultralytics.
"""
from __future__ import annotations
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPTS = os.environ.get(
    "SKILL_SCRIPTS",
    "/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts")
FUSION_RUN = os.path.join(_HERE, "run.py")

# --------------------------------------------------------------------------- #
# Manifest — the declarative contracts an agent plans against
# --------------------------------------------------------------------------- #
# fields: name | kind (engine|judge) | impl (subprocess|inproc) |
#   desc (agent-facing: 用途+时机) | inputs (key: 说明) | outputs |
#   cost (time 估时, cloud 云调用, requires 前置) | fallback (工具名或None)

TOOL_MANIFEST = [
    {
        "name": "preflight", "kind": "engine", "impl": "subprocess",
        "desc": "检查视频可读性; DVR/NVR 私有封装(IMKH/.dav)cv2 读不了会伪装成'目标不存在'。"
                "任何视频检索之前必须跑; SUSPECT 时人眼看 sample.jpg, UNREADABLE 时跑恢复阶梯出干净 mp4。",
        "inputs": {"video": "视频路径", "sample": "抽帧样例输出路径", "recover": "恢复输出路径(可选)"},
        "outputs": ["verdict(OK|SUSPECT|UNREADABLE)", "sample.jpg", "recover.mp4"],
        "cost": {"time": "~秒-分钟", "cloud": 0, "requires": ["ffmpeg(仅恢复时需要)"]},
        "fallback": None,
    },
    {
        "name": "locate", "kind": "engine", "impl": "subprocess",
        "desc": "通用目标(描述型)单遍检测+跟踪: YOLO-World 开放词汇 + BoT-SORT,一次 VLM 种子消歧。"
                "适合'白车''骑车的人'这类无特定标识的目标。",
        "inputs": {"video": "视频路径", "query": "自然语言描述", "out": "输出目录", "use_vlm": "布尔(可选)"},
        "outputs": ["boxes.json", "intervals.json", "metrics.json", "annotated.mp4"],
        "cost": {"time": "分钟级", "cloud": "0-2 次(仅 use_vlm 种子消歧)", "requires": ["yolov8s-worldv2.pt"]},
        "fallback": "fusion_run",
    },
    {
        "name": "verify_target", "kind": "engine", "impl": "subprocess",
        "desc": "特定标识目标(车牌/独特衣着)两阶段: 全片检类 + 逐 crop VLM 核验读标识。"
                "定位相似对象群里的'那一个'。VLM 不可用时自动降级本地 OCR(fast_plate_scan)。",
        "inputs": {"video": "视频路径", "query": "含标识的完整描述", "out": "输出目录",
                   "noun": "检测类名词(可选, 缺省从 query 细化)", "step": "帧间隔(默认3)"},
        "outputs": ["manifest.json", "verify.json", "matches.json", "intervals.json", "metrics.json"],
        "cost": {"time": "分钟级", "cloud": "每去重候选 1 次 VLM", "requires": ["yolov8s-worldv2.pt", "VLM key(否则降级 OCR)"]},
        "fallback": "fast_plate_scan",
    },
    {
        "name": "person_search", "kind": "engine", "impl": "subprocess",
        "desc": "图像找人(人脸/人体参考图): 自动选主体 → 本地 embedding 匹配"
                "(insightface ArcFace 脸 / OSNet 行人, CLIP 兜底)→ VLM 图对图兜底。",
        "inputs": {"video": "视频路径", "ref": "参考图路径", "out": "输出目录",
                   "mode": "auto|face|person", "step": "帧间隔(默认3)",
                   "reid_thresh": "OSNet 余弦阈值(无 VLM 核验时建议 ≥0.75,默认 0.55 太宽)",
                   "face_thresh": "ArcFace 阈值(默认 0.42)"},
        "outputs": ["ref_subject.json", "manifest.json", "verify.json", "matches.json", "intervals.json", "metrics.json"],
        "cost": {"time": "分钟级", "cloud": "仅 embedding 不可用时的逐 crop VLM", "requires": ["yolov8s-worldv2.pt", "OSNet 权重(可选)"]},
        "fallback": "fusion_run",
    },
    {
        "name": "fast_plate_scan", "kind": "engine", "impl": "subprocess",
        "desc": "车牌全片重扫: 复用 car boxes 或全帧, 找车牌色块(蓝/黄/绿)放大均衡后 tesseract OCR,"
                "按字符混淆表(Q↔9/1, G↔6, B↔8…)变体匹配, 可疑时段 step=2 细扫。"
                "精确文本检索的主力与 VLM 降级路径; 15 分钟 1080p 约 9 分钟。",
        "inputs": {"video": "视频路径", "target": "车牌字母数字(无省份字)", "out": "输出目录",
                   "manifest": "car boxes 清单(可选, 缺省 --everywhere 全帧)", "step": "帧间隔(默认10)"},
        "outputs": ["hits.json", "weak_hits.json", "all_ocr.json", "suspicious.json", "metrics.json", "hit_f*.jpg"],
        "cost": {"time": "~9 分钟(15min 1080p)", "cloud": 0, "requires": ["tesseract"]},
        "fallback": None,
    },
    {
        "name": "fusion_run", "kind": "engine", "impl": "subprocess",
        "desc": "云边融合端到端(本包 Phase 1): router → 云粗筛窗口(qwen-vl-max, 失败返空回落全片)"
                "→ 本地检测/裁剪 → 三信号投票(色相/时间曲线/VLM 图对图, agree≥2)。"
                "疑难首选入口: 图像输入用 --ref 走完整三信号; 车牌查询自动走 OCR 分支。",
        "inputs": {"video": "视频路径", "query": "描述(车牌/人物/物体)", "out": "输出目录",
                   "ref": "参考图(可选, 开启三信号)", "agree": "命中所需信号数(默认2)", "no_ground": "布尔, 跳过云粗筛"},
        "outputs": ["route.json", "ground.json", "scored.json", "matches.json", "intervals.json", "metrics.json"],
        "cost": {"time": "分钟级(按分支)", "cloud": "1 次 grounding + 每候选 VLM 终审", "requires": ["yolov8s-worldv2.pt", "可选: DASHSCOPE/ZHIPUAI key"]},
        "fallback": "verify_target",
    },
    {
        "name": "router", "kind": "judge", "impl": "inproc",
        "desc": "任务分类(纯正则, 零成本): semantic(语义描述)|precise_text(车牌/证件号), 并提取标识串。"
                "agent 的第一反应, 但不是唯一判断——复合查询要自己拆步骤。",
        "inputs": {"query": "自然语言查询"},
        "outputs": ["branch", "target", "query"],
        "cost": {"time": "<1ms", "cloud": 0, "requires": []},
        "fallback": None,
    },
    {
        "name": "ground", "kind": "engine", "impl": "inproc",
        "desc": "云粗筛: qwen-vl-max 原生视频一次调用返回候选时间段[{start_s,end_s,score,reason}]。"
                "任何失败(无 key/文件超限/解析失败)返空 [] → 调用方全片扫描, 绝不锁死。",
        "inputs": {"video": "视频路径或 http(s) URL", "query": "描述", "fps": "抽帧率(默认1)"},
        "outputs": ["候选窗口列表"],
        "cost": {"time": "~16s", "cloud": "1 次 qwen-vl-max", "requires": ["DASHSCOPE_API_KEY"]},
        "fallback": None,
    },
    {
        "name": "rescoring", "kind": "judge", "impl": "inproc",
        "desc": "重评分: 对已算好信号的候选集换阈值/agree 重新投票(微秒级)。"
                "verdict=insufficient/0 命中时的第一招: 换阈值重判, 而不是重跑引擎。",
        "inputs": {"candidates": "scored.json 路径或候选列表", "thresholds": "阈值覆盖 dict(可选)",
                   "agree": "命中所需信号数(默认2)"},
        "outputs": ["verdicts(hit/miss/insufficient)", "按分排序的候选"],
        "cost": {"time": "<1ms", "cloud": 0, "requires": []},
        "fallback": None,
    },
]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def get_tool(name: str) -> dict | None:
    return next((t for t in TOOL_MANIFEST if t["name"] == name), None)


def check_requirements(tool: str) -> dict:
    """Cheap availability probe (no heavy imports): which keys/binaries a tool
    needs and which are missing. Helps the agent pick a viable plan up front."""
    t = get_tool(tool)
    if not t:
        return {"ok": False, "missing": [f"unknown tool {tool}"]}
    req = t["cost"].get("requires", [])
    missing = []
    for r in req:
        if "KEY" in r:
            env = r.replace("(可选)", "").strip()
            env = env.split(" ")[0].split("/")[0]
            if not (os.environ.get(env) or (env == "VLM key" and (
                    os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
                    or os.environ.get("DASHSCOPE_API_KEY")))):
                missing.append(r)
        elif r == "tesseract" and not shutil.which("tesseract"):
            missing.append(r)
        elif r == "ffmpeg(仅恢复时需要)" and not shutil.which("ffmpeg"):
            missing.append(r)
        elif r.endswith(".pt") or "权重" in r or r == "OSNet":
            pass  # weights resolve lazily; engines raise their own clear error
    return {"ok": not missing, "missing": missing, "requires": req}


def _script_for(tool: str) -> str | None:
    if tool in ("locate", "verify_target", "person_search", "fast_plate_scan", "preflight"):
        return os.path.join(SKILL_SCRIPTS, tool + ".py")
    if tool == "fusion_run":
        return FUSION_RUN
    return None


_REPO_ROOT = os.path.dirname(_HERE)


def _resolve_model(name: str) -> str:
    """Absolutize a model filename so engine subprocesses (cwd=workdir) never
    trigger an ultralytics re-download: repo root first, then the skill's
    bundled weights dir; falls back to the literal name unchanged."""
    for cand in (os.path.join(_REPO_ROOT, name),
                 os.path.join(SKILL_SCRIPTS, "..", "weights", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return name


def _build_argv(tool: str, params: dict) -> list[str]:
    """Map the manifest's input keys onto each engine's CLI (positional/flag
    names differ across engines — normalized here, one place)."""
    p = dict(params)
    out = p.get("out")
    common = {"locate": ["locate.py"], "verify_target": ["verify_target.py"],
              "person_search": ["person_search.py"],
              "fast_plate_scan": ["fast_plate_scan.py"], "preflight": ["preflight.py"],
              "fusion_run": ["run.py"]}[tool]
    argv = [sys.executable, _script_for(tool)]
    def add(flag, key):
        if key in p and p[key] is not None:
            argv.append(flag); argv.append(str(p[key]))
    def add_flag(flag, key):
        if p.get(key):
            argv.append(flag)
    if tool == "preflight":
        if "video" in p and p["video"] is not None:
            argv.append(str(p["video"]))     # positional: preflight.py VIDEO
        add("--sample", "sample")
        if p.get("recover"):
            argv.append("--recover"); argv.append(str(p["recover"]))
    elif tool == "locate":
        add("--video", "video"); add("--query", "query"); add("--out", "out")
        add_flag("--use-vlm", "use_vlm"); add("--device", "device")
        argv += ["--world-model",
                 p.get("world_model") or _resolve_model("yolov8s-worldv2.pt")]
    elif tool == "verify_target":
        add("--video", "video"); add("--query", "query"); add("--out", "out")
        add("--noun", "noun"); add("--step", "step"); add("--device", "device")
        argv += ["--model", p.get("model") or _resolve_model("yolov8s-worldv2.pt")]
    elif tool == "person_search":
        add("--video", "video"); add("--ref", "ref"); add("--out", "out")
        add("--mode", "mode"); add("--step", "step"); add("--device", "device")
        add("--reid-thresh", "reid_thresh"); add("--face-thresh", "face_thresh")
        argv += ["--model", p.get("model") or _resolve_model("yolov8s-worldv2.pt")]
    elif tool == "fast_plate_scan":
        add("--video", "video"); add("--target", "target"); add("--out", "out")
        if p.get("manifest"):
            argv += ["--manifest", str(p["manifest"])]
        else:
            argv.append("--everywhere")
        add("--step", "step")
    elif tool == "fusion_run":
        add("--video", "video"); add("--query", "query"); add("--out", "out")
        add("--ref", "ref"); add("--agree", "agree"); add("--step", "step")
        add_flag("--no-ground", "no_ground"); add("--device", "device")
        argv += ["--model", p.get("model") or _resolve_model("yolov8s-worldv2.pt")]
    return argv


def _summary_from_outdir(outdir: str) -> dict:
    """Harvest the engines' own metrics.json/intervals.json into the envelope,
    so the agent gets structured numbers without reading raw stdout."""
    s = {}
    for f, keys in (("metrics.json", ("branch", "n_hits", "n_matches", "n_intervals",
                                     "n_ocr_calls", "elapsed_s", "intervals", "degraded_to_ocr")),
                    ("intervals.json", None)):
        path = os.path.join(outdir, f)
        if os.path.exists(path):
            try:
                d = json.load(open(path))
                s[f] = (d if keys is None else {k: d[k] for k in keys if k in d})
            except Exception:
                s[f] = "(unreadable)"
    return s


def dispatch(tool: str, params: dict, workdir: str | None = None,
             on_line=None) -> dict:
    """Execute a tool by name with a params dict; always returns the uniform
    envelope. Engines run as subprocesses with the same interpreter; judges run
    in-process. Never raises for tool failures (status="failed" instead).

    ``on_line(str)``: called with each stdout line as it arrives (live progress
    for the agent/GUI — engines print ``[stage1] scanned x/y`` etc.)."""
    t = get_tool(tool)
    if not t:
        return {"tool": tool, "status": "failed", "summary": {},
                "outputs": {}, "error": f"unknown tool: {tool}", "cost": {}}
    if workdir:
        os.makedirs(workdir, exist_ok=True)

    if t["impl"] == "inproc":
        return _dispatch_inproc(tool, params)

    script = _script_for(tool)
    if not script or not os.path.exists(script):
        return {"tool": tool, "status": "skipped", "summary": {},
                "outputs": {}, "error": f"script not found: {script} (set SKILL_SCRIPTS)",
                "cost": t["cost"]}
    argv = _build_argv(tool, params)
    timeout = int(params.get("timeout", 3600))
    # Engines resolve weights (ultralytics weights_dir) and default config
    # paths RELATIVE TO CWD. Run them from the repo root (where weights/ and
    # fusion/config live) — out dirs in params are absolute, so nothing leaks.
    run_cwd = _REPO_ROOT if os.path.isdir(_REPO_ROOT) else None
    try:
        if on_line is None:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout, cwd=run_cwd)
            stdout, stderr, rc = r.stdout, r.stderr, r.returncode
        else:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    cwd=run_cwd)
            # reader thread: readline() blocks while the engine computes, so
            # pump lines through a queue and enforce the timeout from the poller
            q: queue.Queue = queue.Queue()

            def _pump():
                for line in proc.stdout:
                    q.put(line)
                q.put(None)                     # EOF sentinel

            threading.Thread(target=_pump, daemon=True).start()
            t0 = time.time()
            while True:
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - t0 > timeout:
                        proc.kill()
                        raise subprocess.TimeoutExpired(argv, timeout)
                    continue
                if item is None:
                    break
                on_line(item.rstrip())
            proc.wait(timeout=max(1.0, timeout - (time.time() - t0)))
            stdout, stderr, rc = "", proc.stderr.read(), proc.returncode
    except subprocess.TimeoutExpired:
        return {"tool": tool, "status": "failed", "summary": {},
                "outputs": {}, "error": f"timeout after {timeout}s", "cost": t["cost"]}
    out = params.get("out")
    outputs = {}
    if out and os.path.isdir(out):
        outputs["dir"] = out
        outputs.update({f: os.path.join(out, f) for f in os.listdir(out)
                        if f.endswith((".json", ".jpg", ".png", ".mp4"))})
    summary = _summary_from_outdir(out) if out else {}
    status = "ok" if rc == 0 else "failed"
    tail = (stderr or stdout or "")[-800:]
    return {"tool": tool, "status": status, "summary": summary,
            "outputs": outputs,
            "error": None if status == "ok" else (tail or f"exit {rc}"),
            "cost": t["cost"]}


def _dispatch_inproc(tool: str, params: dict) -> dict:
    if tool == "router":
        import router
        verdict = router.classify_query(params.get("query", ""))
        return {"tool": tool, "status": "ok", "summary": verdict, "outputs": {},
                "error": None, "cost": {"time": "<1ms", "cloud": 0, "requires": []}}
    if tool == "ground":
        import ground as G
        windows = G.ground_candidates(params.get("video", ""), params.get("query", ""),
                                      fps=float(params.get("fps", G.DEFAULT_FPS)))
        return {"tool": tool, "status": "ok", "summary": {"windows": windows,
                                                          "n": len(windows),
                                                          "note": "空列表 = 云不可用/无目标时段, 调用方须全片扫描"},
                "outputs": {}, "error": None,
                "cost": {"time": "~16s", "cloud": 1, "requires": ["DASHSCOPE_API_KEY"]}}
    if tool == "rescoring":
        import json as _json
        import scorer
        cands = params.get("candidates")
        if isinstance(cands, str) and os.path.exists(cands):
            cands = _json.load(open(cands))
        if not isinstance(cands, list) or not cands:
            return {"tool": tool, "status": "failed", "summary": {}, "outputs": {},
                    "error": "candidates: 需要 scored.json 路径或候选列表(非空)", "cost": {}}
        try:
            agree = int(params.get("agree", 2))
        except (TypeError, ValueError):
            agree = 2                      # LLM prose -> default bar, never crash
        thresholds = params.get("thresholds") or {}
        # an LLM may hand a non-dict (e.g. a prose string) — coerce, never crash
        if not isinstance(thresholds, dict):
            thresholds = {}
        # scored.json rows already carry per-signal results; normalize both
        # shapes ({results: {...}} from run.py, or rows pre-scored by scorer).
        for c in cands:
            if "results" not in c and "signals" in c:
                c["results"] = {s["name"]: s for s in c["signals"]}
        scored = scorer.score_all(cands, thresholds, agree)
        summary = {"n": len(scored), "verdicts": {v: sum(1 for c in scored if c["verdict"] == v)
                                                  for v in ("hit", "miss", "insufficient")},
                   "hits": [{"id": c.get("id"), "frame": c.get("frame"), "t_s": c.get("t_s"),
                             "score": c["score"], "fired": c["fired"]}
                            for c in scored if c["verdict"] == "hit"]}
        return {"tool": tool, "status": "ok", "summary": summary,
                "outputs": {"scored": scored}, "error": None,
                "cost": {"time": "<1ms", "cloud": 0, "requires": []}}
    return {"tool": tool, "status": "failed", "summary": {}, "outputs": {},
            "error": f"inproc impl missing for {tool}", "cost": {}}


# --------------------------------------------------------------------------- #
# Agent context — the manifest rendered for prompt injection
# --------------------------------------------------------------------------- #

def agent_context() -> str:
    """Compact agent-readable inventory: one block per tool with cost tags and
    requirements, plus the availability probe for the current environment."""
    lines = ["# 可用工具 (确定性核心)", ""]
    for t in TOOL_MANIFEST:
        c = t["cost"]
        avail = check_requirements(t["name"])
        status = "可用" if avail["ok"] else f"缺: {', '.join(avail['missing'])}"
        lines.append(f"## {t['name']}  [{t['kind']}]  {status}")
        lines.append(f"用途: {t['desc']}")
        lines.append(f"输入: {', '.join(t['inputs'])}")
        lines.append(f"产出: {', '.join(t['outputs'])}")
        lines.append(f"成本: 时间 {c['time']} | 云调用 {c['cloud']}"
                     + (f" | 回退: {t['fallback']}" if t.get("fallback") else ""))
        lines.append("")
    lines.append("# 调用约定")
    lines.append("dispatch(tool, params, workdir) 返回 {tool,status:ok|failed|skipped,"
                 "summary,outputs,error,cost}。status!=ok 时读 error 决定下一步,"
                 "绝不把'失败'说成'目标不存在'。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(agent_context())
