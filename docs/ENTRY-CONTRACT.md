# 双入口契约（standalone 与 DSH 插件 = 同一套代码 + 同一份数据）

> 2026-11-15 定稿。两个入口**不是两套系统**：一份 Python 后端、一份状态根、一个进程。

## 1. 两个入口

| | 独立启动（standalone） | DSH 插件 |
|---|---|---|
| 触发 | `bash scripts/serve.sh` | DSH 加载 `vr-console` 行（clientOnly）+ `vr-plugin` 行 |
| 代码 | `server/` + `engine/`（本包） | 同一份 |
| 状态根 | `~/.dsh/video-retrieval`（`VTL_STATE_DIR` 可覆盖） | 同一份 |
| 端口 | 8788（`VTL_PORT` 可覆盖；`backendUrl` 单点） | 同一值 |
| 界面 | 浏览器直接打开控制台 `http://127.0.0.1:8788/` | 侧边栏按钮 + iframe 打开**同一个**控制台 |
| 工具 | 无 | 16 个 `vr_*` 工具（走同一后端 HTTP） |
| 典型场景 | 无 DSH / 只想用控制台 / 调试 | 在 DSH 会话里用工具 + 面板 |

## 2. 共享的不变量

1. **一份代码**：只有本包的 `server/`、`engine/`；`fusion/` 是上游源，通过 `scripts/sync.sh` 镜像进来，`scripts/sync.sh --check` 在 CI 里把关。
2. **一份状态**：唯一可写根由 `VTL_STATE_DIR` 决定，默认 `~/.dsh/video-retrieval`；包内目录只读。
3. **一个进程**：`server/main()` 启动时用 `flock` 锁 `<state>/server.pid`，第二个实例直接退出（exit 2），不会双写。
4. **身份握手**：`GET /health` 返回 `{version, pid, nonce, state_root, cases_path}`；同时把同样内容写到 `<state>/server.identity.json`。插件只认 pid+nonce 一致的进程（不再用路径子串）。
5. **URL 不硬编码**：客户端用 `window.__VR_BACKEND__`（host 注入）优先，默认 `127.0.0.1:8788` 仅作兜底。

## 3. 先起谁都可以

| 顺序 | 结果 |
|---|---|
| 先 standalone，后 DSH | DSH 探测到 pid+nonce 一致 → **复用**，不新起进程；16 个工具立刻可用 |
| 先 DSH，后 `serve.sh` | serve.sh 的 `/health` 探测命中同一状态根 → 打印 "backend already running"，只打开浏览器 |
| 两个都用不同 `VTL_STATE_DIR` | 被视为两套系统，各自独立（允许，但案例库不共享） |
| 端口被别的服务占着 | 插件报错并打印对方 pid/nonce 与期望值，提示 `kill <pid>` 或改 `backendUrl` |

## 4. 升级/卸载

- 状态根在包外，`npm`/Release 升级不动数据。
- 重启后端：`POST /shutdown`（带 `X-Requested-With: vr-console`），或用 pid 文件里的 pid。
- 客户端与控制台版本必须匹配后端 `/health` 的 `version`；不匹配时控制台顶部会提示刷新。
