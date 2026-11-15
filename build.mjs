// Build the host half + validate the client half:
//   host:  syntax-check src/index.js, then write dist/index.js.
//          (P2-1：不再生成 dist/index.<hash8>.js —— preset 早已改用包名解析，
//           全仓没有任何地方引用哈希产物，它们只是构建垃圾。)
//   client: dist-client/index.js is hand-maintained (__ModuleLoader__ bundle);
//          only syntax-checked here.
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
mkdirSync(distDir, { recursive: true })
writeFileSync(join(distDir, 'index.js'), code)
console.log('built: dist/index.js  (client bundle syntax OK)')
