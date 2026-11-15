# config/ — 运行时演化配置

这个目录是**自进化层（evolution layer）的运行时状态**，由后端在运行中读写。它随包发布（本 README 保证目录在 git/tarball 里存在），但其中的运行时产物不入 git。

## 会在这里生成的文件

| 文件 | 内容 | 谁写 |
|---|---|---|
| `thresholds.json` | 多信号投票阈值（Optuna 优化结果） | `vr_evolve` / 演化控制器 |
| `confusables.json` | 车牌字符混淆表 | `vr_evolve`（Layer1 反射） |
| `prompts.json` | 提示词配置 | `vr_evolve` |
| `evolutions.jsonl` | 演化审计日志 | 后端 |
| `rollback/` | 回滚快照 | 后端 |
| `.veto` | 冻结演化开关（touch 即冻结） | 后端 / `vr_veto` |

## 说明

- 目录为空 = 使用内置默认阈值/混淆表/提示词，完全可用。
- 缺失的 JSON 会静默降级到默认值，不会崩。
- 用 `.veto` 可冻结自进化，防止自动改配置。
