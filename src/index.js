// @ts-check
/**
 * dsh-video-retrieval — native DSH mode plugin (P1–P3).
 *
 * Host half: registers the vr_* toolset. Retrieval jobs run in the vendored
 * fusion backend (dsh/server/server.py, spawned lazily); preflight and the
 * bench tools spawn the repo venv directly. Plain ESM (TypeScript + esbuild
 * arrive with a dev-dependency pass later).
 */
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'dsh-video-retrieval'
export const inject = ['tools']

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// The plugin's own package root (dist/index.js → ../ = package root). This is
// the only reliable location anchor: it is independent of the DSH loader's
// `!!js` scope (which has no `require`) and of the profile/preset baseUrl.
const PKG_ROOT = fileURLToPath(new URL('..', import.meta.url))

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {Record<string, unknown>} config row config from agent.cordis.yml
 */
export function apply(ctx, config = {}) {
  const str = (v) => (typeof v === 'string' && v !== '' ? v : undefined)
  const cfg = {
    python: str(config.python) || process.env.VTL_PY || 'python3',
    engineDir: str(config.engineDir) || path.join(PKG_ROOT, 'engine'),
    serverDir: str(config.serverDir) || path.join(PKG_ROOT, 'server'),
    configDir: str(config.configDir) || path.join(PKG_ROOT, 'config'),
    dataDir: str(config.dataDir) || path.join(os.homedir(), '.dsh', 'video-retrieval'),
    datasetDir: str(config.datasetDir) || process.env.BENCH_DATA || '',
    backendUrl: str(config.backendUrl) || 'http://127.0.0.1:8788',
  }

  const BACKEND_PORT = (() => {
    try { return new URL(cfg.backendUrl).port || '8787' } catch { return '8787' }
  })()

  // ── subprocess helper ─────────────────────────────────────────────────────

  /**
   * Spawn the repo venv python, collect stdout/stderr, honor the tool call's
   * AbortSignal, enforce a timeout.
   * @param {string[]} args
   * @param {{ signal?: AbortSignal | null, timeoutMs?: number, cwd?: string, env?: Record<string,string> }} opts
   */
  function runPython(args, { signal, timeoutMs = 120_000, cwd, env } = {}) {
    return new Promise((resolve) => {
      /** @type {import('node:child_process').ChildProcess | undefined} */
      let child
      let stdout = ''
      let stderr = ''
      let timedOut = false
      let spawnError = ''
      const timer = setTimeout(() => { timedOut = true; child?.kill('SIGKILL') }, timeoutMs)
      const onAbort = () => child?.kill('SIGKILL')
      signal?.addEventListener('abort', onAbort, { once: true })
      try {
        child = spawn(cfg.python, args, {
          cwd: cwd || cfg.engineDir || undefined,
          env: env ? { ...process.env, ...env } : undefined,
          stdio: ['ignore', 'pipe', 'pipe'],
        })
      } catch (err) {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onAbort)
        resolve({ code: -1, signal: null, stdout: '', stderr: String(err), timedOut: false })
        return
      }
      child.stdout?.on('data', (d) => { stdout += d.toString() })
      child.stderr?.on('data', (d) => { stderr += d.toString() })
      child.on('error', (err) => { spawnError = String(err) })
      child.on('close', (code, sig) => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onAbort)
        resolve({ code: code ?? -1, signal: sig ?? null, stdout, stderr: spawnError || stderr, timedOut })
      })
    })
  }

  // ── backend helpers ───────────────────────────────────────────────────────

  /** @returns {Record<string,string>} env the fusion backend runs under. */
  function backendEnv() {
    return {
      ...process.env,
      SKILL_SCRIPTS: cfg.engineDir,
      FUSION_CONFIG: cfg.configDir,
      FUSION_CASES: path.join(cfg.dataDir, 'cases.jsonl'),
      FUSION_JOBS: path.join(cfg.dataDir, 'jobs'),
      FUSION_UPLOADS: path.join(cfg.dataDir, 'uploads'),
      FUSION_WEB: path.join(cfg.serverDir, 'web'),
      BENCH_DATA: cfg.datasetDir,
    }
  }

  /**
   * @param {string} method
   * @param {string} urlPath
   * @param {unknown} body
   * @param {{ signal?: AbortSignal | null, timeoutMs?: number }} opts
   */
  async function http(method, urlPath, body, { signal, timeoutMs = 60_000 } = {}) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    const onAbort = () => controller.abort()
    signal?.addEventListener('abort', onAbort, { once: true })
    try {
      const res = await fetch(cfg.backendUrl + urlPath, {
        method,
        headers: body !== undefined ? { 'content-type': 'application/json' } : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      })
      const text = await res.text()
      let data
      try { data = JSON.parse(text) } catch { data = { raw: text.slice(0, 800) } }
      return { status: res.status, data }
    } catch (err) {
      if (controller.signal.aborted && !signal?.aborted) {
        return { status: 0, data: { error: `backend timeout after ${timeoutMs}ms` } }
      }
      return { status: 0, data: { error: String(err) } }
    } finally {
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
    }
  }

  let backendReady = false

  /**
   * Lazily start the vendored fusion backend and wait for /health.
   * Health is only trusted when it reports OUR cases_path (a foreign server
   * may occupy the port — never talk to it).
   */
  async function ensureBackend(signal) {
    if (backendReady) return true
    const expectedCases = path.join(cfg.dataDir, 'cases.jsonl')
    const trust = (d) => d?.status === 'ok' && String(d?.cases_path ?? '').includes(cfg.dataDir)
    const probe = await http('GET', '/health', undefined, { signal, timeoutMs: 2000 })
    if (probe.status === 200 && trust(probe.data)) { backendReady = true; return true }
    if (probe.status === 200 && probe.data?.status === 'ok') {
      throw new Error(`a foreign server occupies ${cfg.backendUrl} (cases_path=${probe.data?.cases_path}); expected ${expectedCases}. Stop it or change backendUrl in the preset config.`)
    }
    fs.mkdirSync(cfg.dataDir, { recursive: true })
    const logFd = fs.openSync(path.join(cfg.dataDir, 'server.log'), 'a')
    const child = spawn(cfg.python, [path.join(cfg.serverDir, 'server.py'), '--host', '127.0.0.1', '--port', BACKEND_PORT], {
      cwd: cfg.serverDir,
      env: backendEnv(),
      stdio: ['ignore', logFd, logFd],
      detached: true,
    })
    child.on('error', () => { try { fs.closeSync(logFd) } catch {} })
    child.unref()
    for (let i = 0; i < 90; i++) {
      await sleep(1000)
      const h = await http('GET', '/health', undefined, { signal, timeoutMs: 2000 })
      if (h.status === 200 && trust(h.data)) { backendReady = true; return true }
    }
    return false
  }

  /** Submit a search job to the backend. Throws on failure (tool error). */
  async function submitSearch(payload, exec, hint) {
    const up = await ensureBackend(exec.signal)
    if (!up) {
      throw new Error(`video-retrieval backend unreachable at ${cfg.backendUrl} (see ${path.join(cfg.dataDir, 'server.log')})`)
    }
    const r = await http('POST', '/search', payload, { signal: exec.signal, timeoutMs: 60_000 })
    if (r.status !== 200) throw new Error(`vr search rejected (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
    return { ...r.data, hint: hint ?? 'poll with vr_job_status / collect with vr_job_result' }
  }

  // ── tools ─────────────────────────────────────────────────────────────────

  // Client-only host row (profile insert for the web console half): the
  // client bundle is discovered from the package manifest and served to the
  // browser; this host plane must not register the tools a second time —
  // instead it makes sure the console backend is up so the iframe has
  // something to show.
  if (config.clientOnly === true) {
    void (async () => {
      try {
        const probe = await http('GET', '/health', undefined, { timeoutMs: 1500 })
        if (probe.status === 200 && probe.data?.status === 'ok') return
        fs.mkdirSync(cfg.dataDir, { recursive: true })
        const logFd = fs.openSync(path.join(cfg.dataDir, 'server.log'), 'a')
        const child = spawn(cfg.python, [path.join(cfg.serverDir, 'server.py'), '--host', '127.0.0.1', '--port', BACKEND_PORT], {
          cwd: cfg.serverDir,
          env: backendEnv(),
          stdio: ['ignore', logFd, logFd],
          detached: true,
        })
        child.on('error', () => { try { fs.closeSync(logFd) } catch {} })
        child.unref()
      } catch { /* console backend is best-effort at boot */ }
    })()
    return
  }

  ctx.tools.register(defineTool({
    name: 'vr_preflight',
    description:
      'Pre-flight a surveillance video BEFORE any retrieval: decodes a spread of frames and checks they are real and distinct (catches IMKH/MPEG-PS DVR exports, frozen frames, zero-frame files), dumps a sample JPEG, and — when the file is unreadable — runs the ffmpeg recovery ladder (remux → MPEG-PS PES demux) to a clean mp4. ALWAYS run first: "target absent" after a skipped preflight is meaningless.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video file.' },
      recover: { type: 'string', description: 'Optional absolute path for the recovered, engine-readable mp4 (needs ffmpeg).' },
      sample: { type: 'string', description: 'Optional absolute path for the sample-frame JPEG; defaults to a temp file.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          verdict: { type: 'string', required: true, enum: ['OK', 'SUSPECT', 'UNREADABLE', 'RECOVERED', 'ERROR'] },
          ok: { type: 'boolean', required: true },
          sample: { type: 'string' },
          recovered: { type: 'string' },
          details: { type: 'string' },
          stderr: { type: 'string' },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `vr_preflight → ${value.verdict}${value.recovered ? `; recovered: ${value.recovered}` : ''}${value.ok ? '' : ' — verify the sample frame before trusting detections'}`,
      }],
    },
    async execute(args, exec) {
      const script = path.join(cfg.engineDir, 'preflight.py')
      const sample = args.sample ?? path.join(os.tmpdir(), `vr-preflight-${Date.now()}.jpg`)
      const argv = [script, args.video, '--sample', sample]
      if (args.recover) argv.push('--recover', args.recover)
      const r = await runPython(argv, { signal: exec.signal })
      const out = r.stdout
      const verdictLine = out.split('\n').find((l) => l.startsWith('VERDICT:')) ?? ''
      const decodeLine = out.split('\n').find((l) => l.startsWith('cv2 decode')) ?? ''
      let verdict = 'UNREADABLE'
      let ok = false
      let recovered
      if (r.timedOut) verdict = 'ERROR'
      else if (out.includes('VERDICT: OK')) { verdict = 'OK'; ok = true }
      else if (r.code === 0 && out.includes('Treat as readable')) { verdict = 'SUSPECT'; ok = true }
      else if (out.includes('RECOVERED ->')) {
        const m = out.match(/RECOVERED -> (.+?)\s*\(\{/)
        verdict = 'RECOVERED'; ok = true; recovered = m ? m[1].trim() : undefined
      } else if (out.includes('file not found')) verdict = 'ERROR'
      else if (r.code === -1 || r.signal) verdict = 'ERROR'
      const result = {
        verdict, ok, sample,
        details: [verdictLine, decodeLine].filter(Boolean).join(' | '),
        stderr: (r.stderr ?? '').slice(-1500),
      }
      // Only attach `recovered` when present: an `undefined` field breaks the
      // harness lossless-JSON output boundary ("value is not lossless JSON").
      if (recovered) result.recovered = recovered
      return result
    },
  }))

  const jobOutput = {
    schema: { type: 'json' },
    render: (_args, value) => [{
      type: 'text',
      text: typeof value?.status === 'string'
        ? `vr job ${value.job_id ?? ''} → ${value.status}${value.status === 'running' ? ' (poll vr_job_status, collect vr_job_result)' : ''}`
        : `vr job ${JSON.stringify(value).slice(0, 300)}`,
    }],
  }

  ctx.tools.register(defineTool({
    name: 'vr_search',
    description:
      'Submit a retrieval job to the video-retrieval backend: surveillance video + natural-language query (and/or reference image) → temporal presence intervals, per-frame boxes, and annotated artifacts. The backend routes precise-text (plate/ID) to local OCR and semantic queries to the cloud-edge funnel with multi-signal voting. Returns a job_id immediately; poll vr_job_status and collect vr_job_result.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video.' },
      query: { type: 'string', required: true, description: 'Natural-language target description, plate, or identifier.' },
      ref: { type: 'string', description: 'Optional absolute path to a reference image (face/person crop) — enables person search + three-signal verification.' },
      mode: { type: 'string', enum: ['auto', 'deterministic'], description: 'Backend run mode; auto = LLM-assisted playbook, deterministic = fixed playbook without an LLM. Defaults to auto.' },
    },
    output: jobOutput,
    async execute(args, exec) {
      return submitSearch({ video: args.video, query: args.query, ref: args.ref, mode: args.mode ?? 'auto' }, exec)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_locate',
    description: 'Submit a general-target localization job (YOLO-World grounding + tracking + VLM seed disambiguation) for a described object ("the white car", "person riding a bicycle"). Job-based; poll vr_job_status / vr_job_result.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video.' },
      query: { type: 'string', required: true, description: 'Target description.' },
    },
    output: jobOutput,
    async execute(args, exec) {
      return submitSearch({ video: args.video, query: args.query, mode: 'auto' }, exec)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_verify_target',
    description: 'Submit a specific-identifier verification job (two-stage detect-everywhere + per-crop VLM verification, with local OCR fallback for plates). Use for a particular plate or distinctive clothing among lookalikes. Job-based; poll vr_job_status / vr_job_result.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video.' },
      query: { type: 'string', required: true, description: 'The identifier ("车牌号 京Q1G728 的小轿车", "person with a red backpack").' },
    },
    output: jobOutput,
    async execute(args, exec) {
      return submitSearch({ video: args.video, query: args.query, mode: 'auto' }, exec)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_person_search',
    description: 'Submit a person/face search job from a reference image (auto subject selection → embedding match: ArcFace/OSNet/CLIP → multi-signal verification). Job-based; poll vr_job_status / vr_job_result.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video.' },
      ref: { type: 'string', required: true, description: 'Absolute path to the reference image (face or person crop).' },
      query: { type: 'string', description: 'Optional clothing/context description to help grounding.' },
    },
    output: jobOutput,
    async execute(args, exec) {
      return submitSearch({ video: args.video, query: args.query ?? '', ref: args.ref, mode: 'auto' }, exec)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_fast_plate_scan',
    description: 'Submit a whole-clip plate scan job: local OCR + character-confusion table over candidate plate regions (works without any cloud key). Job-based; poll vr_job_status / vr_job_result.',
    parameters: {
      video: { type: 'string', required: true, description: 'Absolute path to the surveillance video.' },
      plate: { type: 'string', required: true, description: 'The plate to find (e.g. 京Q1G728 or Q1G728).' },
    },
    output: jobOutput,
    async execute(args, exec) {
      return submitSearch({ video: args.video, query: `车牌号 ${args.plate} 的车辆`, mode: 'auto' }, exec)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_job_status',
    description: 'Poll a video-retrieval job: {status: running|done|failed, result}.',
    parameters: {
      job_id: { type: 'string', required: true, description: 'Job id returned by vr_search/vr_locate/vr_verify_target/vr_person_search/vr_fast_plate_scan.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr job ${v?.job_id} → ${v?.status ?? 'unknown'}` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('GET', `/search/${args.job_id}`, undefined, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status === 404) throw new Error(`unknown job ${args.job_id}`)
      if (r.status !== 200) throw new Error(`vr job status failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_job_result',
    description: 'Collect a finished video-retrieval job result (intervals, boxes/matches, metrics, artifact paths). Returns status + a pointer when the job is still running.',
    parameters: {
      job_id: { type: 'string', required: true, description: 'Job id.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr job ${v?.job_id} → ${v?.status ?? 'unknown'}` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('GET', `/search/${args.job_id}`, undefined, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status === 404) throw new Error(`unknown job ${args.job_id}`)
      if (r.status !== 200) throw new Error(`vr job result failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      if (r.data?.status !== 'done' && r.data?.status !== 'failed') {
        return { ...r.data, done: false, message: `job still ${r.data?.status ?? 'unknown'} — poll vr_job_status, then call vr_job_result again.` }
      }
      return { ...r.data, done: true }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_job_cancel',
    description: 'Cancel a running retrieval job (terminates its engine subprocesses and marks it cancelled). Use when a search is stuck or no longer needed.',
    parameters: {
      job_id: { type: 'string', required: true, description: 'Job id.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr cancel ${v?.job_id} → ${v?.status ?? 'unknown'}` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('POST', `/search/${args.job_id}/cancel`, {}, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status === 404) throw new Error(`unknown job ${args.job_id}`)
      if (r.status !== 200) throw new Error(`vr cancel failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_cases',
    description: 'List the case library (recorded difficult/confirmed cases) — the self-improvement feed. Pass confirmed=true to see only human-confirmed cases.',
    parameters: {
      confirmed: { type: 'boolean', description: 'Only confirmed cases.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr cases → ${Array.isArray(v?.cases) ? v.cases.length : '?'} entries` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('GET', '/cases', undefined, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status !== 200) throw new Error(`vr cases failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      const data = r.data
      if (args.confirmed && Array.isArray(data.cases)) {
        data.cases = data.cases.filter((c) => c?.confirmed)
      }
      return data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_case_confirm',
    description: 'Record a human confirmation for a case (hit intervals / positive reads) — the one human action that feeds the evolution layer.',
    parameters: {
      case_id: { type: 'string', required: true, description: 'Case id from vr_cases.' },
      hit_intervals: { type: 'json', description: 'GT hit intervals, e.g. [["580","582"],["758","760"]] or [{t_s,t_e}] — kept as the backend expects.' },
      positive_reads: { type: 'json', description: 'Optional positive reads list.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr case ${v?.case_id ?? ''} confirmed → ${JSON.stringify(v).slice(0, 200)}` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const gt = {}
      if (args.hit_intervals !== undefined) gt.hit_intervals = args.hit_intervals
      if (args.positive_reads !== undefined) gt.positive_reads = args.positive_reads
      const r = await http('POST', `/cases/${args.case_id}/confirm`, { gt }, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status !== 200) throw new Error(`vr case confirm failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_config',
    description: 'Read the active runtime config (thresholds / confusables / prompts) — what the self-evolution layer last adopted, plus veto state.',
    parameters: {},
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr config → ${JSON.stringify(v).slice(0, 300)}` }] },
    async execute(_args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('GET', '/config', undefined, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status !== 200) throw new Error(`vr config failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_veto',
    description: 'Freeze or unfreeze the self-evolution loop (.veto). Freeze = all evolution runs skip until unfrozen.',
    parameters: {
      freeze: { type: 'boolean', required: true, description: 'true = freeze evolution, false = unfreeze.' },
      reason: { type: 'string', description: 'Short reason recorded with the veto.' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr veto → ${JSON.stringify(v)}` }] },
    async execute(args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('POST', '/config/veto', { freeze: args.freeze, reason: args.reason ?? '' }, { signal: exec.signal, timeoutMs: 30_000 })
      if (r.status !== 200) throw new Error(`vr veto failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_evolve',
    description:
      'Trigger ONE self-evolution cycle: merge confirmed cases + benchmark candidates → Optuna threshold optimization + Layer1 confusables/prompts reflection → holdout gate (adopt only if ≥ old − 0.005) → rollback snapshot → hot reload into dsh/config/. Honors .veto (skips when frozen). Can take minutes.',
    parameters: {},
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr evolve → ${JSON.stringify(v).slice(0, 400)}` }] },
    async execute(_args, exec) {
      await ensureBackend(exec.signal)
      const r = await http('GET', '/evolve', undefined, { signal: exec.signal, timeoutMs: 900_000 })
      if (r.status !== 200) throw new Error(`vr evolve failed (HTTP ${r.status}): ${JSON.stringify(r.data)}`)
      return r.data
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_report',
    description: 'Run the benchmark report generator (4-dataset harness with SOTA comparison) into dsh/data/report. Long-running (minutes); results are files, not model context.',
    parameters: {},
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr report → ${JSON.stringify(v).slice(0, 300)}` }] },
    async execute(_args, exec) {
      const out = path.join(cfg.dataDir, 'report')
      const r = await runPython([path.join(cfg.serverDir, 'bench', 'make_report.py'), '--out', out], {
        signal: exec.signal, timeoutMs: 1800_000, cwd: cfg.serverDir, env: backendEnv(),
      })
      if (r.code !== 0) throw new Error(`vr report failed (exit ${r.code}): ${(r.stderr || r.stdout || '').slice(-2000)}`)
      return { status: 'ok', out, stdout: r.stdout.slice(-1500) }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'vr_calibrate',
    description: 'Calibrate multi-signal thresholds for one scene: given a scored.json and GT intervals, sweep thresholds and write the suggestion (feeds the evolution layer).',
    parameters: {
      scored: { type: 'string', required: true, description: 'Absolute path to a scored.json produced by a search run.' },
      gt: { type: 'string', required: true, description: 'Ground truth intervals as "start,end;start,end" in seconds.' },
      fps: { type: 'number', description: 'Video fps (default 25).' },
      agree: { type: 'integer', description: 'Minimum agreeing signals (default 2).' },
    },
    output: { schema: { type: 'json' }, render: (_a, v) => [{ type: 'text', text: `vr calibrate → ${JSON.stringify(v).slice(0, 300)}` }] },
    async execute(args, exec) {
      const out = path.join(cfg.dataDir, `calibration-${Date.now()}.json`)
      const argv = [path.join(cfg.serverDir, 'bench', 'calibrate.py'), '--scored', args.scored, '--gt', args.gt, '--out', out]
      if (args.fps !== undefined) argv.push('--fps', String(args.fps))
      if (args.agree !== undefined) argv.push('--agree', String(args.agree))
      const r = await runPython(argv, { signal: exec.signal, timeoutMs: 600_000, cwd: cfg.serverDir, env: backendEnv() })
      if (r.code !== 0) throw new Error(`vr calibrate failed (exit ${r.code}): ${(r.stderr || r.stdout || '').slice(-2000)}`)
      return { status: 'ok', out, stdout: r.stdout.slice(-1500) }
    },
  }))
}
