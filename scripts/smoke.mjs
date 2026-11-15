#!/usr/bin/env node
// Pure-Node smoke test (no Python, no weights) — runnable in CI.
// 1. Host half: imports dist/index.js, registers the vr_* tools against a
//    fake tool registry, and verifies the clientOnly guard registers nothing.
// 2. Client half: loads dist-client/index.js through a fake
//    window.__ModuleLoader__ and verifies both slot registrations fire.
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(import.meta.url)

// ── 1. host half ────────────────────────────────────────────────────────────
// dist/index.js imports @deepseek-ai/dsh-tools (a peer only present inside the
// DSH profile). In CI it is absent, so the registration check degrades to a
// warning; build.mjs already syntax-checks the host half.
let host = null
try {
  host = await import(pathToFileURL(join(root, 'packages', 'vr-plugin', 'dist', 'index.js')).href)
} catch (err) {
  console.error(`[fail] host half import: ${err.code ?? err.message}`)
  process.exit(1)
}

const cfg = {
  python: 'python3',
  engineDir: join(root, 'packages', 'vr-backend', 'vr_backend', 'engines'),
  serverDir: join(root, 'packages', 'vr-backend', 'vr_backend'),
  configDir: join(root, 'packages', 'vr-backend', 'vr_backend', 'config'),
  dataDir: join(root, 'packages', 'vr-backend', 'vr_backend', 'data'),
  datasetDir: '',
  backendUrl: 'http://127.0.0.1:8788',
}

// apply() is async: awaiting it is the whole point — an un-awaited call returns
// before a single tool is registered (the old smoke test did exactly that and
// reported 0 tools instead of the real count).
const registered = []
await host.apply({ tools: { register: (t) => registered.push(t) } }, cfg)

// @deepseek-ai/dsh-tools is a peer that only exists inside a DSH profile.
// Missing peer = exit 3 ("skipped"), an explicit, visible state — never a
// warn-and-pass: a green smoke with zero tools registered proves nothing.
if (registered.length === 0) {
  console.error('[skip] host half: no tools registered — @deepseek-ai/dsh-tools peer not resolvable '
    + '(run scripts/install.sh, which links the profile peer into node_modules)')
  process.exit(3)
}
const names = registered.map((t) => t.name)
for (const name of ['vr_preflight', 'vr_search', 'vr_job_cancel', 'vr_evolve']) {
  if (!names.includes(name)) throw new Error(`host half missing tool: ${name}`)
}
console.log(`host half: ${names.length} tools registered (${names.join(', ')})`)

const guardReg = []
await host.apply({ tools: { register: (t) => guardReg.push(t) } }, { ...cfg, clientOnly: true })
if (guardReg.length !== 0) throw new Error('clientOnly guard failed')
console.log('host half: clientOnly guard OK')

// ── 2. client half ──────────────────────────────────────────────────────────
let def
global.window = { __ModuleLoader__: { load: (d) => { def = d } } }
const clientPath = join(root, 'packages', 'vr-plugin', 'dist-client', 'index.js')
// The bundle is a classic script (references `window`); execute it directly.
const code = readFileSync(clientPath, 'utf8')
new Function(code)()
if (!def) throw new Error('client bundle did not register with __ModuleLoader__')
const fakeReact = {
  createElement: (t) => ({ t }),
  useSyncExternalStore: (s, g) => g(),
  useEffect: (f) => { f() },
  useRef: (v) => ({ current: v }),
  useState: (i) => [i, () => {}],
}
const regs = []
const api = def.factory((id) => (id === 'react' ? fakeReact : undefined))
api.apply({ get: (k) => (k === 'slots' ? { inject: (n, cb) => { cb(); return () => {} }, register: (o) => { regs.push(o.name) } } : undefined) })
for (const s of ['shell.overlay', 'sidebar.footer.action']) {
  if (!regs.includes(s)) throw new Error(`client bundle missing slot: ${s}`)
}
console.log(`client half: slots registered (${regs.join(', ')})`)

console.log('✅ smoke OK')
