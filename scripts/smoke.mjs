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
const host = await import(pathToFileURL(join(root, 'dist', 'index.js')).href)
const registered = []
const cfg = {
  python: 'python3',
  engineDir: join(root, 'engine'),
  serverDir: join(root, 'server'),
  configDir: join(root, 'config'),
  dataDir: join(root, 'data'),
  datasetDir: '',
  backendUrl: 'http://127.0.0.1:8788',
}
host.apply({ tools: { register: (t) => registered.push(t) } }, cfg)
const names = registered.map((t) => t.name)
const expected = ['vr_preflight', 'vr_search', 'vr_job_cancel', 'vr_evolve']
for (const name of expected) {
  if (!names.includes(name)) throw new Error(`host half missing tool: ${name}`)
}
console.log(`host half: ${names.length} tools registered (${names.join(', ')})`)

const guardReg = []
host.apply({ tools: { register: (t) => guardReg.push(t) } }, { ...cfg, clientOnly: true })
if (guardReg.length !== 0) throw new Error('clientOnly guard failed')
console.log('host half: clientOnly guard OK')

// ── 2. client half ──────────────────────────────────────────────────────────
let def
global.window = { __ModuleLoader__: { load: (d) => { def = d } } }
const clientPath = join(root, 'dist-client', 'index.js')
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
