// P1-16: path resolution, mirrored 1:1 with vr_backend/paths.py.
//
// Python is the authority (it resolves the state root the backend actually
// writes to). This module exists so the plugin never invents its own default:
// `tests/test_paths_contract.py` runs BOTH resolvers in the same environment
// and fails if any of state/engines/backend/client/bench differs.
//
// Priority order is identical to paths.py — keep them in step:
//   state    VTL_STATE_DIR > $DSH_HOME/video-retrieval > ~/.dsh/video-retrieval
//   weights  VTL_WEIGHTS_DIR > <repo>/models > ~/.cache/vr/weights
//   bench    BENCH_DATA > <repo>/dataset > <repo>/../dataset
//   engines  SKILL_SCRIPTS > <backend>/engines
//   client   FUSION_WEB > <repo>/packages/vr-plugin/client
import os from 'node:os'
import path from 'node:path'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// src/paths.js → ../ = packages/vr-plugin → ../.. = packages → ../../.. = repo
export const PKG_ROOT = fileURLToPath(new URL('..', import.meta.url))
export const REPO_ROOT = fileURLToPath(new URL('../../../', import.meta.url)).replace(/\/$/, '')

const home = () => os.homedir()

export function repoRoot() {
  let cur = PKG_ROOT
  for (let i = 0; i < 6; i += 1) {
    if (existsSync(path.join(cur, 'package.json')) || existsSync(path.join(cur, '.git'))) return cur
    const parent = path.dirname(cur)
    if (parent === cur) break
    cur = parent
  }
  return REPO_ROOT
}

export function backendDir() {
  return path.join(repoRoot(), 'packages', 'vr-backend', 'vr_backend')
}

export function enginesDir() {
  return process.env.SKILL_SCRIPTS || path.join(backendDir(), 'engines')
}

export function dshHome() {
  return process.env.DSH_HOME || path.join(home(), '.dsh')
}

export function stateRoot() {
  const env = process.env.VTL_STATE_DIR
  if (env) return path.resolve(env.replace(/^~(?=$|\/)/, home()))
  return path.join(dshHome(), 'video-retrieval')
}

export function weightsRoot() {
  const env = process.env.VTL_WEIGHTS_DIR
  if (env) return path.resolve(env.replace(/^~(?=$|\/)/, home()))
  const cand = path.join(repoRoot(), 'models')
  if (existsSync(cand)) return cand
  return path.join(home(), '.cache', 'vr', 'weights')
}

export function benchRoot() {
  const env = process.env.BENCH_DATA
  if (env) return path.resolve(env.replace(/^~(?=$|\/)/, home()))
  for (const cand of [path.join(repoRoot(), 'dataset'), path.join(path.dirname(repoRoot()), 'dataset')]) {
    if (existsSync(cand)) return cand
  }
  return path.join(repoRoot(), 'dataset')
}

export function clientRoot() {
  return process.env.FUSION_WEB || path.join(repoRoot(), 'packages', 'vr-plugin', 'client')
}

export function allPaths() {
  return {
    repo_root: repoRoot(),
    backend_dir: backendDir(),
    engines_dir: enginesDir(),
    state_root: stateRoot(),
    weights_root: weightsRoot(),
    bench_root: benchRoot(),
    client_root: clientRoot(),
    dsh_home: dshHome(),
  }
}

// CLI: `node packages/vr-plugin/src/paths.js --print` (JSON) or `… state_root`.
// Kept identical in shape to `python -m vr_backend.paths` so the two can be diffed.
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const args = process.argv.slice(2)
  const all = allPaths()
  if (args.length === 0 || args.includes('--print') || args.includes('--json')) {
    process.stdout.write(`${JSON.stringify(all, null, 2)}\n`)
  } else if (args[0] in all) {
    process.stdout.write(`${all[args[0]]}\n`)
  } else {
    process.stderr.write(`unknown key: ${args[0]} (one of: ${Object.keys(all).join(', ')})\n`)
    process.exit(2)
  }
}
