#!/usr/bin/env python3
"""Agent loop — the operation layer on top of the deterministic core.

The LLM is a *retrieval operator*, not a from-scratch solver: the system
prompt is ``toolbox.agent_context()`` (tool manifest + cost tags) + the full
AGENT-INSTRUCTIONS playbook + the current runtime-config summary (E0 seam).
Each step is 模型选工具 → ``toolbox.dispatch`` → 统一信封回灌 → 判定.

Channels (all env-driven, no hard dependency):
  * tool-calling via litellm when available (``LLM_BASE_URL/LLM_API_KEY/
    LLM_MODEL``, default GLM OpenAI-compatible glm-4-flash → DashScope
    qwen-plus fallback); tool-calling failing → pure-text ReAct lines
    ``dispatch(tool, {...})`` / ``final({...})``;
  * no key at all → the deterministic playbook (§3 定式路径 + §4 疑难剧本的
    固定版) — still honest, still records hard cases, never fabricates.

Guardrails enforced HERE (not trusted to the model):
  * preflight always runs first — an unreadable video masquerades as
    "目标不存在" otherwise;
  * a hit requires non-empty intervals from an engine's own output; an
    unsubstantiated LLM "hit" is downgraded, never echoed;
  * failure/insufficient is reported as 尝试了X / 覆盖Y / 原因Z / 建议 —
    the string "目标不存在" is never produced;
  * max_steps caps the loop; every §4 (疑难) disposition ends in
    ``case_log.record_case`` so confirmed cases feed the evolution layer.

Controlled autonomous coding: ``run_glue`` executes agent-generated glue
scripts in a subprocess with a whitelisted-import scan and a timeout —
no network except through the provider module.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import toolbox as TB                    # noqa: E402
import case_log as CL                   # noqa: E402
try:
    import config as CFG                # noqa: E402
    _HAS_CONFIG = True
except Exception:                       # noqa: BLE001
    CFG = None
    _HAS_CONFIG = False

MAX_STEPS = 12

# last LLM failure (surfaced in degrade notes — a silent swallow cost a whole
# debugging round once: a clobbered .env key read as generic "unavailable")
LITELM_LAST_ERROR: str | None = None

# --------------------------------------------------------------------------- #
# text ReAct fallback (tool-calling unavailable)
# --------------------------------------------------------------------------- #

_RE_DISPATCH = re.compile(
    r"^\s*dispatch\s*\(\s*([A-Za-z_][\w]*)\s*,\s*(\{.*\})\s*\)\s*$", re.S)
_RE_FINAL = re.compile(r"^\s*final\s*\(\s*(\{.*\})\s*\)\s*$", re.S)


def parse_react_line(line: str):
    """Parse one ReAct instruction line.

    ``dispatch(tool, {...}) -> ("dispatch", tool, params)``,
    ``final({...}) -> ("final", None, obj)``, anything else ``None``."""
    m = _RE_DISPATCH.match((line or "").strip())
    if m:
        try:
            return ("dispatch", m.group(1), json.loads(m.group(2)))
        except json.JSONDecodeError:
            return None
    m = _RE_FINAL.match((line or "").strip())
    if m:
        try:
            return ("final", None, json.loads(m.group(1)))
        except json.JSONDecodeError:
            return None
    return None


def _extract_json(text: str):
    """Pull the first {...} JSON object out of a free-text reply, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# LLM channel (litellm, GLM default → DashScope fallback)
# --------------------------------------------------------------------------- #

def llm_env() -> tuple[str | None, str | None, str]:
    base = (os.environ.get("LLM_BASE_URL") or
            "https://open.bigmodel.cn/api/paas/v4")
    key = (os.environ.get("LLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
           or os.environ.get("GLM_API_KEY"))
    model = os.environ.get("LLM_MODEL") or "glm-4-flash"
    return base, key, model


def llm_key_present() -> bool:
    return bool(llm_env()[1] or os.environ.get("DASHSCOPE_API_KEY"))


def _vlm_key_present() -> bool:
    """Any VLM key (the image-search arbiter)."""
    return bool(os.environ.get("ZHIPUAI_API_KEY")
                or os.environ.get("GLM_API_KEY")
                or os.environ.get("DASHSCOPE_API_KEY"))


def _litellm_chat(messages: list[dict], tools: list[dict] = None):
    """One LLM call. Returns {text:...} | {tool_calls:[...]} | None (all
    providers failed). Tries tool-calling first, plain text second."""
    # Skip litellm's remote model-cost-map fetch: on CN networks the GitHub
    # request times out and adds ~10s of dead wait to the first call.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    base, key, model = llm_env()
    providers = []
    if key:
        providers.append((f"openai/{model}", base, key))
    dash = os.environ.get("DASHSCOPE_API_KEY")
    if dash:
        providers.append(("openai/qwen-plus",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1", dash))
    if not providers:
        return None
    try:
        import litellm
    except Exception:                       # noqa: BLE001
        return None
    for m, b, k in providers:
        kw = dict(model=m, messages=messages, api_base=b, api_key=k,
                  temperature=0.2, timeout=120, num_retries=0)
        attempts = [("tools", tools)] if tools else []
        attempts.append(("text", None))
        for kind, payload in attempts:
            if kind == "tools":
                kw["tools"] = payload
                kw["tool_choice"] = "auto"
            else:
                kw.pop("tools", None)
                kw.pop("tool_choice", None)
            try:
                resp = litellm.completion(**kw)
                msg = resp.choices[0].message
                # thinking models (deepseek-v4-pro 等) require echoing
                # reasoning_content back on the next request — without it the
                # API rejects the round-trip with a 400.
                rc = getattr(msg, "reasoning_content", None) or ""
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    return {"tool_calls": [
                                # "type" is required by strict OpenAI-compatible
                                # deserializers (deepseek rejects without it)
                                {"id": getattr(t, "id", f"c{i}"),
                                 "type": "function",
                                 "name": t.function.name,
                                 "arguments": t.function.arguments}
                                for i, t in enumerate(tc)],
                            "reasoning_content": rc}
                if getattr(msg, "content", None):
                    return {"text": msg.content, "reasoning_content": rc}
            except Exception as e:               # noqa: BLE001 — next provider/attempt
                global LITELM_LAST_ERROR
                LITELM_LAST_ERROR = f"{type(e).__name__}: {str(e)[:180]}"
                continue
    return None


# --------------------------------------------------------------------------- #
# controlled autonomous coding (glue subprocess)
# --------------------------------------------------------------------------- #

GLUE_ALLOWED_IMPORTS = {
    "json", "os", "sys", "re", "math", "statistics", "pathlib", "collections",
    "itertools", "functools", "datetime", "time", "glob", "shutil", "argparse",
    "csv", "cv2", "numpy", "PIL", "provider", "fusion.provider",
}
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))",
    re.M)


def run_glue(code: str, workdir: str, timeout: int = 300) -> dict:
    """Execute an agent-generated glue script in a controlled subprocess.

    Import whitelist + timeout; ``requests`` is NOT whitelisted — network only
    via the provider module. Returns {status: ok|failed|timeout|rejected,
    stdout, stderr, returncode, error}."""
    for m in _IMPORT_RE.finditer(code or ""):
        mod = (m.group(1) or m.group(2)).split(".")[0]
        if mod not in GLUE_ALLOWED_IMPORTS:
            return {"status": "rejected",
                    "error": f"import not whitelisted: {mod}", "stdout": "",
                    "stderr": "", "returncode": None}
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "glue.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=workdir)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout}s",
                "stdout": "", "stderr": "", "returncode": None}
    return {"status": "ok" if r.returncode == 0 else "failed",
            "stdout": (r.stdout or "")[-4000:],
            "stderr": (r.stderr or "")[-2000:],
            "returncode": r.returncode, "error": None}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    """The retrieval operator. ``run()`` is the only public entry point.

    ``dispatch_fn``/``llm_fn`` are test seams (default to the real toolbox and
    litellm channel); ``on_event`` receives the SSE-able step events."""

    def __init__(self, workdir: str = None, max_steps: int = MAX_STEPS,
                 on_event=None, cases_path: str = None,
                 dispatch_fn=None, llm_fn=None):
        self.workdir = workdir or tempfile.mkdtemp(prefix="agent-run-")
        self.max_steps = max_steps
        self.on_event = on_event or (lambda e: None)
        self.cases_path = cases_path or CL.DEFAULT_CASES_PATH
        self.llm_fn = llm_fn
        # wrap the real dispatcher to stream engine stdout as live progress
        # (throttled to ~1 line/s so per-frame prints don't flood the GUI)
        base = dispatch_fn or TB.dispatch
        self._last_engine_emit: dict[str, float] = {}

        def _wrapped(tool, params, workdir=None):
            # no VLM keys -> image search has NO verifier, so the engine's
            # loose 0.55 reid threshold floods false matches (report-1:
            # true ≈0.83, FPs 0.55-0.72). Raise it to 0.75 and say so.
            if (tool == "person_search"
                    and "reid_thresh" not in params
                    and not _vlm_key_present()):
                params = {**params, "reid_thresh": 0.75}
                self._emit({"type": "note",
                            "msg": "无 VLM 核验 → 以图搜人采用 reid 0.75 "
                                   "高阈值预筛,结果需人工复核"})
            if base is TB.dispatch:

                def _on_line(ln):
                    now = time.time()
                    last = self._last_engine_emit.get(tool, 0.0)
                    if (now - last >= 1.0 or ln.startswith("[done]")
                            or ln.startswith("[stage")):
                        self._last_engine_emit[tool] = now
                        self._emit({"type": "engine", "tool": tool,
                                    "line": ln[:200]})
                return base(tool, params, workdir, on_line=_on_line)
            return base(tool, params, workdir)
        self.dispatch_fn = _wrapped
        self._steps: list[int] = []
        self._events: list[dict] = []
        self._envelopes: list[dict] = []
        self._attempts: list[dict] = []
        self._tried: list[str] = []
        self._case_id: str | None = None
        self._hard_case: bool = False

    # -- plumbing ------------------------------------------------------------

    def _emit(self, event: dict):
        event.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._events.append(event)
        try:
            self.on_event(event)
        except Exception:                   # noqa: BLE001 — events are advisory
            pass

    def _dispatch(self, tool: str, params: dict, workdir: str = None) -> dict:
        t0 = time.time()
        try:
            env = self.dispatch_fn(tool, params, workdir or self.workdir)
        except Exception as e:               # noqa: BLE001 — a bad tool input must
            env = {"tool": tool, "status": "failed", "summary": {},   # never kill
                   "outputs": {}, "error": f"{type(e).__name__}: {e}",  # the loop
                   "cost": {}}
        env.setdefault("tool", tool)
        self._envelopes.append(env)
        if tool not in self._tried:
            self._tried.append(tool)
        self._attempts.append({"tool": tool, "params": params,
                               "note": "", "envelope": env})
        self._emit({"type": "step", "tool": tool, "status": env.get("status"),
                    "summary": env.get("summary"), "error": env.get("error"),
                    "duration_s": round(time.time() - t0, 1)})
        return env

    def _llm_available(self) -> bool:
        return bool(self.llm_fn) or llm_key_present()

    def _call_llm(self, messages, tools=None):
        if self.llm_fn:
            return self.llm_fn(messages, tools=tools)
        return _litellm_chat(messages, tools=tools)

    def _system_prompt(self) -> str:
        parts = [TB.agent_context()]
        instr = os.path.join(_HERE, "AGENT-INSTRUCTIONS.md")
        if os.path.exists(instr):
            with open(instr, encoding="utf-8") as f:
                parts.append(f.read())
        if _HAS_CONFIG:
            parts.append("# 当前运行时配置\n"
                         + json.dumps(CFG.summary(), ensure_ascii=False, indent=2))
        return "\n\n".join(parts)

    def _tool_schemas(self) -> list[dict]:
        schemas = []
        for t in TB.TOOL_MANIFEST:
            props = {k: {"type": "string", "description": v}
                     for k, v in t["inputs"].items()}
            required = [k for k in ("video", "query", "ref", "target")
                        if k in t["inputs"]]
            schemas.append({"type": "function", "function": {
                "name": t["name"], "description": t["desc"],
                "parameters": {"type": "object", "properties": props,
                               "required": required}}})
        return schemas

    # -- judging -------------------------------------------------------------

    def _judge(self, env: dict) -> tuple[str, list, dict]:
        """Map an envelope to (verdict, intervals, evidence). A hit needs the
        engine's own intervals — never synthesized."""
        s = env.get("summary") or {}
        m = s.get("metrics.json") or {}
        intervals = m.get("intervals") or []
        n_hits = int(m.get("n_hits", 0) or m.get("n_matches", 0) or len(intervals))
        hits = s.get("hits") or []
        if env.get("status") == "ok" and (n_hits > 0 or intervals or hits):
            if not intervals and hits:
                intervals = [{"start_s": float(h["t_s"]), "end_s": float(h["t_s"]) + 0.5,
                              "evidence": {"fired": h.get("fired"), "score": h.get("score")}}
                             for h in hits]
            return "hit", intervals, {"n_hits": n_hits or len(intervals),
                                      "summary": s}
        verdict = "insufficient" if env.get("status") == "ok" else "failed"
        return verdict, [], {"summary": s, "error": env.get("error")}

    def _best_hit_evidence(self) -> tuple[list, dict]:
        for env in reversed(self._envelopes):
            verdict, intervals, evidence = self._judge(env)
            if verdict == "hit":
                return intervals, evidence
        return [], {}

    # -- case recording ------------------------------------------------------

    def _record_case(self, query, video, ref, qtype, outcome, reasoning, plan):
        provider_status = {"glm": bool(llm_env()[1]),
                           "qwen": bool(os.environ.get("DASHSCOPE_API_KEY"))}
        case = {
            "id": CL.new_case_id(), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query": query, "query_type": qtype,
            "inputs": {"video": video, "ref": ref, "provider_status": provider_status},
            "plan": plan,
            "attempts": [{"tool": a["tool"], "params": a["params"],
                          "note": a["note"], "envelope": a["envelope"]}
                         for a in self._attempts],
            "outcome": outcome, "agent_reasoning": reasoning,
            "human_confirmed": None, "gt": None,
        }
        ok, msg = CL.record_case(case, self.cases_path)
        if ok:
            self._case_id = case["id"]
        self._emit({"type": "case", "case_id": case["id"], "msg": msg})

    # -- result assembly -----------------------------------------------------

    def _finalize(self, verdict, intervals, evidence, **kw) -> dict:
        res = {"verdict": verdict, "intervals": intervals, "evidence": evidence,
               "case_id": self._case_id,
               "tried": list(self._tried),
               "coverage": kw.get("coverage", ""),
               "reason": kw.get("reason", ""),
               "suggestion": kw.get("suggestion", ""),
               "note": kw.get("note", ""),
               "log": list(self._events)}
        self._emit({"type": "final", "verdict": verdict,
                    "n_intervals": len(intervals), "case_id": self._case_id})
        return res

    # -- main entry ----------------------------------------------------------

    def run(self, query: str, video: str, ref: str = None,
            mode: str = "auto") -> dict:
        """Run one retrieval task. Returns the AGENT-INSTRUCTIONS §7 shape:
        {verdict: hit|no_hit|inconclusive|user_input, intervals, evidence,
        case_id, tried, coverage, reason, suggestion, log}."""
        # Engine subprocesses run with cwd=workdir — absolutize user paths so
        # relative video/ref arguments survive the context switch.
        video = os.path.abspath(video or "")
        if ref:
            ref = os.path.abspath(ref)
        # ref-only search: no text query — default to image-based person
        # search and skip router (an empty string would misclassify).
        ref_only = not (query or "").strip() and bool(ref)
        if ref_only:
            query = "参考图中的人物(以图搜人)"
        self._emit({"type": "start", "query": query, "video": video,
                    "ref": ref, "mode": mode})

        # ① preflight — never skipped (护栏 #1).
        pre = self._dispatch("preflight",
                             {"video": video,
                              "sample": os.path.join(self.workdir, "sample.jpg")})
        pre_verdict = str((pre.get("summary") or {}).get("verdict", "")).upper()
        if pre.get("status") != "ok" or pre_verdict == "UNREADABLE":
            self._record_case(query, video, ref, "environment", "unresolved",
                              f"preflight: {pre_verdict or pre.get('error')}",
                              ["preflight 检查视频可读性", "视频不可读 → 如实报告"])
            return self._finalize(
                "inconclusive", [], {},
                coverage="未扫描(视频不可读)",
                reason=f"视频预检未通过: {pre_verdict or pre.get('error')}。"
                       "视频不可读会伪装成'目标不存在',必须先恢复视频。",
                suggestion="检查文件是否 DVR/NVR 私有封装(IMKH/.dav);若是,用 "
                           "preflight --recover 走 ffmpeg 恢复阶梯出干净 mp4 后重试。")

        # ref-only: image-based person search, router intentionally skipped.
        if ref_only:
            route = {"branch": "semantic", "target": None, "query": query}
            self._emit({"type": "note", "msg": "仅参考图 → 以图搜人(语义分支)"})
            if mode == "deterministic" or not self._llm_available():
                if not self._llm_available():
                    self._emit({"type": "note",
                                "msg": "无 LLM key → 走确定性剧本(定式路径+固定疑难剧本)"})
                return self._playbook(query, video, ref, route)
            return self._llm_loop(query, video, ref, route, pre, None)

        # ② router — the plan anchor.
        rt_env = self._dispatch("router", {"query": query})
        route = rt_env.get("summary") or {}
        branch = route.get("branch", "semantic")

        if mode == "deterministic" or not self._llm_available():
            if not self._llm_available():
                self._emit({"type": "note",
                            "msg": "无 LLM key → 走确定性剧本(定式路径+固定疑难剧本)"})
            return self._playbook(query, video, ref, route)

        return self._llm_loop(query, video, ref, route, pre, rt_env)

    # -- deterministic playbook (§3 定式 + §4 固定剧本) ------------------------

    def _playbook(self, query, video, ref, route) -> dict:
        branch = route.get("branch", "semantic")
        plan = ["preflight 检查视频", "router 分类",
                "选引擎跑定式", "不足 → rescoring 重判 → 换阈值 → 换引擎 → 如实报告"]
        if branch == "precise_text":
            if not route.get("target"):
                self._record_case(query, video, ref, "insufficient", "unresolved",
                                  "无法从查询提取车牌/证件号", plan)
                return self._finalize(
                    "no_hit", [], {},
                    coverage="未扫描(无检索标识)",
                    reason="未能从查询中提取车牌/证件号等精确标识串。",
                    suggestion="补充车牌号(如 京Q1G728)或改述为语义描述。")
            env = self._dispatch("fast_plate_scan", {
                "video": video, "target": route["target"],
                "out": os.path.join(self.workdir, "plate")})
        elif ref:
            env = self._dispatch("fusion_run", {
                "video": video, "query": query, "ref": ref,
                "out": os.path.join(self.workdir, "fusion")})
        else:
            env = self._dispatch("fusion_run", {
                "video": video, "query": query,
                "out": os.path.join(self.workdir, "fusion")})

        verdict, intervals, evidence = self._judge(env)
        if verdict == "hit":
            # record the case so the GUI 确认/否认 buttons can turn every
            # hit into evolution feed (image search is 预筛+复核 by design)
            self._record_case(query, video, ref, "routine", "resolved",
                              "以图搜人命中(待人工复核确认)" if ref
                              else "检索命中(可确认/否认)", plan)
            return self._finalize("hit", intervals, evidence,
                                  coverage=self._coverage(branch),
                                  note=self._interval_flood_note(intervals))

        qtype = "environment" if verdict == "failed" else "insufficient"
        for a in self._attempts:
            a["note"] = a["note"] or "确定性剧本执行"

        # ③ insufficient → rescoring (微秒级) → permissive → 换引擎.
        if env.get("status") == "ok" and (env.get("outputs") or {}).get("scored.json"):
            env = self._dispatch("rescoring", {
                "candidates": env["outputs"]["scored.json"], "agree": 2})
            verdict, intervals, evidence = self._judge(env)
            if verdict == "hit":
                self._record_case(query, video, ref, "insufficient", "resolved",
                                  "换阈值重判后命中", plan)
                return self._finalize("hit", intervals, evidence,
                                      coverage=self._coverage(branch))
            env = self._dispatch("rescoring", {
                "candidates": env.get("outputs", {}).get("scored", []),
                "thresholds": {"hue_dhue": 80.0, "temporal_peak": 0.4,
                               "temporal_bellness": 0.1, "temporal_span_s": 0.0},
                "agree": 1})
            verdict, intervals, evidence = self._judge(env)
            if verdict == "hit":
                self._record_case(query, video, ref, "insufficient", "resolved",
                                  "放宽阈值重判后命中(注:候选需人工复核)", plan)
                return self._finalize("hit", intervals, evidence,
                                      coverage=self._coverage(branch),
                                      note="permissive rescoring hit — 建议人工复核")

        if ref:
            env = self._dispatch("person_search", {
                "video": video, "ref": ref, "out": os.path.join(self.workdir, "ps")})
        else:
            env = self._dispatch("verify_target", {
                "video": video, "query": query,
                "out": os.path.join(self.workdir, "vt")})
        verdict, intervals, evidence = self._judge(env)
        if verdict == "hit":
            self._record_case(query, video, ref, qtype, "resolved",
                              "换引擎后命中", plan)
            return self._finalize("hit", intervals, evidence,
                                  coverage=self._coverage(branch))

        self._record_case(query, video, ref, qtype, "unresolved",
                          "定式+rescoring+换引擎均未命中;如实报告",
                          plan)
        return self._finalize(
            "no_hit", [], evidence,
            coverage=self._coverage(branch),
            reason=f"已尝试 {', '.join(self._tried)};覆盖范围: {self._coverage(branch)};"
                   "引擎返回 insufficient/失败(信号不足或环境缺失),未观察到满足"
                   "阈值的时间段。这仅表示本系统当前未找到,不代表目标一定不在视频中。",
            suggestion="建议: 换更清晰的参考图 / 补充标识性描述 / 人工复核候选帧。")

    def _coverage(self, branch) -> str:
        return ("全片 OCR 扫描" if branch == "precise_text" else
                "云粗筛窗口 + 本地检测(全片兜底)")

    @staticmethod
    def _interval_flood_note(intervals) -> str:
        """Honest warning when a hit report looks like a threshold flood."""
        if len(intervals) > 20:
            return (f"命中区间偏多({len(intervals)} 个)——疑似阈值过宽或目标特征"
                    "不具区分度;请人工复核,或提供更清晰的参考图/换检索方式。")
        return ""

    # -- LLM loop ------------------------------------------------------------

    def _llm_loop(self, query, video, ref, route, pre_env, rt_env) -> dict:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content":
                f"任务: 在视频中检索目标。\nvideo: {video}\nquery: {query}\n"
                f"ref: {ref or '无'}\n"
                "已自动完成: preflight 与 router(结果见下)。继续: 每次只输出一个 "
                "dispatch(工具名, {参数}) 指令; 判定结束时输出 "
                "final({\"verdict\": \"hit|no_hit|user_input\", \"note\": \"依据\"})。"},
            {"role": "user", "content": json.dumps(
                {"preflight": pre_env, "router": rt_env}, ensure_ascii=False)[:4000]},
        ]
        for step in range(1, self.max_steps + 1):
            self._steps.append(step)
            reply = self._call_llm(messages, tools=self._tool_schemas())
            if reply is None:
                reason = LITELM_LAST_ERROR or "无响应"
                self._emit({"type": "note",
                            "msg": f"LLM 通道不可用({reason[:120]})→ 降级确定性剧本"})
                return self._playbook(query, video, ref, route)
            tool_calls = reply.get("tool_calls")
            if tool_calls:
                # Some models emit the "final" verdict as a pseudo tool call;
                # treat it as the verdict instead of dispatching an unknown tool.
                for tc in tool_calls:
                    if str(tc.get("name", "")).lower() in ("final", "finish",
                                                           "stop"):
                        try:
                            obj = json.loads(tc.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            obj = {}
                        if isinstance(obj, dict):
                            return self._finalize_from_llm(obj, query, video,
                                                           ref, route)
                # wire format: strict OpenAI-compatible APIs (deepseek) require
                # the nested {"id","type","function":{name,arguments}} shape.
                wire = [{"id": tc.get("id", f"c{step}"), "type": "function",
                         "function": {"name": tc.get("name", ""),
                                      "arguments": tc.get("arguments", "{}")}}
                        for tc in tool_calls]
                assistant = {"role": "assistant", "content": "",
                             "tool_calls": wire}
                if reply.get("reasoning_content"):
                    assistant["reasoning_content"] = reply["reasoning_content"]
                messages.append(assistant)
                for tc in tool_calls:
                    name = tc.get("name", "")
                    try:
                        params = json.loads(tc.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        params = {}
                    if not isinstance(params, dict):
                        params = {}
                    # the task owns its inputs and outputs — the model may
                    # NOT redirect them elsewhere (a model-invented out dir
                    # once stranded all annotated frames outside the job)
                    params["video"] = video
                    params["out"] = os.path.join(self.workdir,
                                                 f"step{step}_{name or 'tool'}")
                    if ref:
                        params.setdefault("ref", ref)
                    env = self._dispatch(name, params)
                    messages.append({"role": "tool",
                                     "tool_call_id": tc.get("id", f"c{step}"),
                                     "content": json.dumps(env, ensure_ascii=False)[:6000]})
                continue
            text = (reply.get("text") or "").strip()
            parsed = parse_react_line(text)
            if parsed and parsed[0] == "dispatch":
                _, tool, params = parsed
                params["video"] = video
                params["out"] = os.path.join(self.workdir,
                                             f"step{step}_{tool}")
                if ref:
                    params.setdefault("ref", ref)
                env = self._dispatch(tool, params)
                asst = {"role": "assistant", "content": text}
                if reply.get("reasoning_content"):
                    asst["reasoning_content"] = reply["reasoning_content"]
                messages.append(asst)
                messages.append({"role": "user",
                                 "content": json.dumps(env, ensure_ascii=False)[:6000]})
                continue
            final_obj = parsed[2] if parsed and parsed[0] == "final" \
                else _extract_json(text)
            if final_obj is not None:
                return self._finalize_from_llm(final_obj, query, video, ref, route)
            messages.append({"role": "user", "content":
                "请只输出 dispatch(tool, {...}) 或 final({...}) 两种指令之一。"})
        # max_steps exhausted
        self._record_case(query, video, ref, "insufficient", "unresolved",
                          f"超过 max_steps={self.max_steps} 仍未收敛", [])
        return self._finalize(
            "no_hit", [], {},
            coverage="部分扫描(未完成)", 
            reason=f"agent 循环超过 max_steps={self.max_steps} 未收敛;"
                   f"已尝试 {', '.join(self._tried) or '无'}。"
                   "结果不足仅表示本系统当前未找到。",
            suggestion="提高 max_steps 或改用 mode=deterministic 走定式路径。")

    def _finalize_from_llm(self, obj, query, video, ref, route) -> dict:
        verdict = str(obj.get("verdict", "")).lower()
        note = obj.get("note", "")
        if verdict == "hit":
            intervals, evidence = self._best_hit_evidence()
            if not intervals:
                # 护栏: hit 必须带引擎产出的时间段.
                self._record_case(query, video, ref, "insufficient", "unresolved",
                                  "LLM 声称命中但无引擎时间段依据,已降级", [])
                return self._finalize(
                    "no_hit", [], evidence,
                    coverage=self._coverage(route.get("branch", "semantic")),
                    reason="模型声称命中,但没有任何引擎输出支撑命中时间段,"
                           "按护栏降级为未命中(断言必须基于证据)。",
                    suggestion="复核最近引擎输出的 scored.json/intervals.json。")
            # 命中即记案例:确认/否认按钮是演化层养料
            self._record_case(query, video, ref, "routine", "resolved",
                              "LLM 判定命中(待人工复核确认)", [])
            return self._finalize("hit", intervals, evidence,
                                  coverage=self._coverage(route.get("branch", "semantic")),
                                  note=self._interval_flood_note(intervals)
                                       or (note or ""))
        if verdict == "user_input":
            self._record_case(query, video, ref, "insufficient", "user_input",
                              note or "需要用户补充信息", [])
            return self._finalize("user_input", [], {},
                                  coverage="", reason=note or "需要用户补充信息",
                                  suggestion="请提供更清晰的参考图或补充标识性描述。")
        self._record_case(query, video, ref, "insufficient",
                          "resolved" if verdict == "no_hit" else "unresolved",
                          note or "LLM 判定未命中(如实报告)", [])
        return self._finalize(
            "no_hit", [], {},
            coverage=self._coverage(route.get("branch", "semantic")),
            reason=(note or "模型判定未命中") + ";已尝试 "
                   + ", ".join(self._tried) + "。结果不足仅表示本系统当前未找到。",
            suggestion="建议换参考图/补充标识/人工复核候选帧。")
