#!/usr/bin/env node
// CDP helper for usability testing (dev tooling): drive Chrome DevTools to
// navigate, evaluate JS, click by text/selector, and capture screenshots.
//
// Usage:
//   node cdp.mjs shot <url> <out.png> [--wait ms] [--click "<text or selector>"] [--eval "<js>"] [--width W] [--height H]
//   node cdp.mjs eval <url> "<js>" [--wait ms]
//
// Requires Chrome with remote debugging: default http://127.0.0.1:9222
// (launch: Chrome --headless=new --remote-debugging-port=9222 --user-data-dir=/tmp/vr-cdp ...)
const [mode, url, arg, ...rest] = process.argv.slice(2)
const opts = {}
for (let i = 0; i < rest.length; i++) {
  if (rest[i] === '--wait') opts.wait = Number(rest[++i])
  else if (rest[i] === '--click') opts.click = rest[++i]
  else if (rest[i] === '--eval') opts.eval = rest[++i]
  else if (rest[i] === '--width') opts.width = Number(rest[++i])
  else if (rest[i] === '--height') opts.height = Number(rest[++i])
  else if (rest[i] === '--nonav') opts.nonav = true
}
const CDP_HTTP = process.env.CDP_HTTP || 'http://127.0.0.1:9222'
const waitMs = opts.wait ?? 2500

async function main() {
  const targets = await (await fetch(`${CDP_HTTP}/json/list`)).json()
  const page = targets.find((t) => t.type === 'page')
  if (!page) throw new Error('no page target; is Chrome running with --remote-debugging-port?')
  const ws = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })
  let seq = 0
  const pending = new Map()
  const events = []
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id) }
    else if (msg.method) events.push(msg)
  }
  const send = (method, params = {}) => new Promise((resolve) => {
    const id = ++seq
    pending.set(id, resolve)
    ws.send(JSON.stringify({ id, method, params }))
  })
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
  const waitEvent = async (method, timeoutMs = 20000) => {
    const t0 = Date.now()
    while (Date.now() - t0 < timeoutMs) {
      const i = events.findIndex((e) => e.method === method)
      if (i >= 0) return events.splice(i, 1)[0]
      await sleep(100)
    }
    return null
  }

  await send('Page.enable')
  await send('Runtime.enable')
  if (opts.width && opts.height) {
    await send('Emulation.setDeviceMetricsOverride', {
      width: opts.width, height: opts.height, deviceScaleFactor: 1, mobile: false,
    })
  }
  if (!opts.nonav) {
    await send('Page.navigate', { url })
    await waitEvent('Page.loadEventFired')
    await sleep(waitMs)
  }

  if (opts.click) {
    const js = typeof opts.click === 'string' && opts.click.startsWith('(')
      ? opts.click
      : `(() => {
        const t = ${JSON.stringify(opts.click)}
        const el = [...document.querySelectorAll('button, a, [role=button], input, label')]
          .find(e => (e.textContent || '').trim().includes(t))
        if (!el) return 'NOT FOUND: ' + t
        el.click(); return 'CLICKED: ' + t
      })()`
    const r = await send('Runtime.evaluate', { expression: js, returnByValue: true })
    console.log('click:', r.result?.result?.value)
    await sleep(2000)
  }
  if (mode === 'eval' || opts.eval) {
    const expr = mode === 'eval' ? arg : opts.eval
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
    console.log('eval:', JSON.stringify(r.result?.result?.value ?? r.result?.exceptionDetails?.text).slice(0, 3000))
  }
  if (mode === 'shot') {
    const shot = await send('Page.captureScreenshot', { format: 'png' })
    const fs = await import('node:fs')
    fs.writeFileSync(arg, Buffer.from(shot.result.data, 'base64'))
    console.log(`screenshot → ${arg}`)
  }
  ws.close()
}
main().catch((e) => { console.error(String(e)); process.exit(1) })
