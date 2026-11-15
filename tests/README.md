# tests/ — 一棵树、三种层（P1-1 / P1-17）

一条命令跑全部：

```bash
/Users/zhouql1978_1/dev/video-retrieval/video/bin/python tests/run_all.py
```

`run_all.py` 发现 `tests/`、`tests/agent/`、`tests/bench/` 下所有 `test_*.py`，
逐个跑子进程；任一非 0 退出 → 整体 exit 1（CI 同款口径）。不需要 pytest。

## 三层

| 层 | 文件 | 回答的问题 |
|---|---|---|
| **契约层**（跨模块/跨进程） | `test_artifacts_contract.py`、`test_paths_contract.py`、`test_p1_ui_guards.py`、`test_p0_api_guards.py` | 引擎↔服务↔前端的产物格式对不对；Python 与 JS 的路径解析是否一致；源码里有没有把用户数据直接塞进 innerHTML |
| **后端单测**（原 `fusion/` 迁入） | `test_smoke.py`、`test_agent.py`、`test_config.py`、`test_phase2.py`、`test_server.py`、`agent/`、`bench/` | 检索管线、评测指标、服务端点、agent 循环的行为 |
| **回归层**（P0/P1/P2 每条修复的钉子） | `test_p0_*.py`、`test_p1_*.py`、`test_p2_*.py`、`golden/` | 某个具体 bug 修完不许回来 |

## 约定

1. **不依赖 pytest**：普通脚本 + 自己数 `passed/failed` + `sys.exit(1)`。
2. **不写仓库**：所有产物写 `tempfile.mkdtemp()`；`test_server.py` 还会在跑前拍
   `config/` 与 `.env` 的快照，跑完比对，防止测试污染真实运行态。
3. **重活不进门禁**：需要模型权重/大视频的用例（`test_p1_scheduler.py` 等）
   只用桩或跳过；`bench/` 里跑真实推理的脚本不进 `run_all.py`。
4. **bench 数据只读**：`BENCH_DATA`（缺省 `<repo>/dataset` 或 `<repo>/../dataset`，
   见 `vr_backend/paths.py`）。测试不许写 `bench_out/`。
5. **先红后绿**：每条 P0/P1 修复的用例，提交前都要把修复 stash 掉跑一遍证明它变红
   （见 `../coding-review/video-retrieval/PLAN.md` 附录 A），否则不算修完。
