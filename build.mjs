// Build the host half + validate the client half:
//   host:  syntax-check src/index.js, then write TWO artifacts:
//          dist/index.js            — latest copy (used by local smoke tests)
//          dist/index.<hash8>.js    — content-hashed; the preset row points here so
//                                     each deploy is a NEW module URL (the Cordis
//                                     loader caches plugins by file URL per process —
//                                     re-deploying to the same path would keep
//                                     serving stale code until a dsh restart).
//   client: dist-client/index.js is hand-maintained (__ModuleLoader__ bundle);
//          only syntax-checked here.
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = dirname(fileURLToPath(import.meta.url))
const src = join(root, 'src', 'index.js')
const clientSrc = join(root, 'dist-client', 'index.js')
const distDir = join(root, 'dist')

for (const file of [src, clientSrc]) {
  const check = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' })
  if (check.status !== 0) {
    console.error(`${file} failed syntax check:\n${check.stderr}`)
    process.exit(1)
  }
}

const code = readFileSync(src, 'utf8')
const hash = createHash('sha256').update(code).digest('hex').slice(0, 8)

mkdirSync(distDir, { recursive: true })
writeFileSync(join(distDir, 'index.js'), code)
writeFileSync(join(distDir, `index.${hash}.js`), code)
console.log(`built: dist/index.js + dist/index.${hash}.js  (client bundle syntax OK)`)
