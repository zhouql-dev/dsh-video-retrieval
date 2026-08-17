# fusion/ — 云边协同检索 harness(agent 操作层 + 自进化回环)

EXECUTION-GUIDE 的 Phase 1 落点:在 `mvp/`(M0 基线,**不动**,当库引用)之上新建
「**云边漏斗 + 多信号演化层**」端到端 pipeline;2026-08-14/15 在其上完成 **Agent Harness
(E0–E7)**:agent loop、运行时配置层、服务+三栏对话 GUI、演化控制器、CCPD 真 GT 基准与
三模式测评。测试 **416 项全绿**(`make test` 十套件 416)。

```
Tier1  video-target-localize (本地)  ──►  router ─► { 语义: 云 grounding 生窗口 + 本地 embed + 三信号投票
   (本 harness)                      │           |  精确文本: 本地 OCR + 字符混淆表 (全片扫描) }
                                     └─► provider(VLM_PROVIDER=glm|qwen) ◄── Q1/Q2/Q3 共用云通道
```

## 模块

| 文件 | 职责 | 契约 |
|---|---|---|
| `provider.py` | 统一 VLM 客户端。`VLM_PROVIDER=glm\|qwen` 路由，active 失败回退另一 provider，**全部 env-gated 静默降级**（无 key → None，本地基线永不硬依赖云端）。复刻 `mvp/glmv.py` 四函数签名/返回不变。 | `refine_query/disambiguate/verify_target/match_image` + `qwen_chat`（原生视频，ground 用） |
| `router.py` | `classify_query(q) → semantic \| precise_text`。车牌/身份证号正则 → precise_text，否则 semantic。纯逻辑、无网络。 | `{branch, target, query}` |
| `ground.py` | `ground_candidates(video, query)`：qwen-vl-max 原生视频接地，返粗候选窗口 `[{start_s,end_s,score,reason}]`。**任何失败/不存在 → 返空**，调用方回落全片。本地文件走 base64（≤`GROUND_VIDEO_MAX_MB`，超限返空），http(s) URL 直传。 | `list[seg]`；`windows_to_frames()` |
| `signals.py` | §11.2 三信号产品化：`hue_consistency`（HSV Δhue，同人≲35°、误报 60-110°）、`temporal_curve`（相似度形态：平滑爬升 vs 单帧尖峰）、`vlm_arbiter`（图对图终审，无 key 弃权）。 | 每个返 `{score, evidence, raw}` |
| `scorer.py` | 多信号投票，`--agree N`（≥N 个非弃权信号 fire 才判 hit）。阈值集中在此（Phase 2 Optuna 落点），**弃权不计入分母**（低色相/无 VLM 时仍可用剩余信号达成一致，单个信号永不 hit）。 | `{hit, verdict, score, fired, abstained, signals:[…]}` |
| `run.py` | 端到端 CLI：router → ground → 本地检测/裁剪（复用 skill `common`/`matcher`）→ signals → scorer。三条分支按路由自动选。 | 输出 `route/ground/matches/intervals/metrics.json` + 标注帧 |
| `toolbox.py` | **工具契约层**：9 个工具（6 引擎 + 3 进程内判定器）声明式 manifest（用途/输入输出/成本标签/回退），`dispatch(tool, params) → 统一信封`，`agent_context()` 供智能体规划。 | `dispatch("router", {...})` 等 |
| `agent/` | **智能体执行层**：`AGENT-INSTRUCTIONS.md`（定式+临场发挥+护栏三原则完整手册）、`case.schema.json` + `case_log.py`（疑难案例记录 → 人工确认 → 转数据集闭环）。 | `case_log.record_case()` / `cases_to_dataset()` |
| `bench/` | **基准测试 harness**(2026-08-13):四数据集 runner + 两条报告验收线 + 优化闭环。见下。 | `venv bench/run_*.py` |
| `config.py` + `config/` | **E0 运行时配置层**:thresholds/confusables/prompts 三文件热加载叠加到核心,rollback 快照 + `.veto` 冻结;演化回填零改代码。 | `load_runtime_config()` / `save_runtime_config()` |
| `agent/loop.py` + `agent/__main__.py` | **E1 智能体循环**:LLM(litellm 工具调用,思考模型 reasoning_content 已支持)→纯文本 ReAct→确定性剧本三级降级;护栏(不谎报/命中须引擎区间/out 归位);CLI `python -m fusion.agent`。 | `Agent.run(query, video, ref, mode)` |
| `server.py` + `web/index.html` | **E2 服务+三栏智能体 GUI**(:8787):SSE 实时进度(引擎 stdout 流式)、jobs 持久化、`/chat` 对话(含对话式配置工具)、`/video/{info,file,transcode,tracked-clip}`、案例确认;`--warmup` 预热、`--watch` 演化周期。 | FastAPI 端点 |
| `agent/evolve.py` | **E3 演化控制器**:确认案例+PRW+CCPD-GT 合并 → Optuna/混淆表反射 → 留存集门禁 → 回填 config → `evolutions.jsonl`。 | `run_once()` / CLI `--once\|--watch` |
| 测试 | **416 项全绿**(`make test`):smoke 42 / agent 125 / phase2 20 / config 32 / loop 44 / evolve 25 / server 69 / bench 28 / ccpd_gt 17 + harness 测评 14。 | `make test` |

## 设计依据（两份报告）

| 报告结论 | 本 harness 如何吸收 |
|---|---|
| **人物**：单语义模型自信幻觉（0.95 误报）、单 embedding 过度匹配（OSNet 16 exact 里 15 误报）→ 必须**多信号一致**。 | `signals` + `scorer`：三信号交叉、`--agree` 默认 2。`temporal_curve` 专门惩罚单帧尖峰。 |
| **车牌**：云端 grounding **返空**（读不了 70×27 小牌）、云端 OCR 误读（京→苏E7589）→ 精确文本必须**本地 OCR + 混淆表 + 全片扫描**，且**不能因 grounding 返空而锁死**。 | `router` → precise_text 分支直接调 `fast_plate_scan.py`（`--everywhere`）；`ground` 返空 → `windows_to_frames` 返全片范围。 |
| 云边接缝 = `glmv.py` 四函数。 | `provider.py` 原样保留契约，仅在 `_chat` 加 provider 路由。 |

## 智能体执行层（泛化性来源）

算法 = **确定性核心（工具）+ 智能体（操作者）**。常规查询走定式（可靠、可测试）；
算法覆盖不到的疑难（复合查询/结果不足/环境异常/参考图质量差/组合检索/新字符集）由智能体
**组合工具临场发挥**——覆盖面从"预设路径"变成"工具的所有组合"：

- `toolbox.py`：9 个工具的统一契约（成本标签 + `dispatch` 统一信封 + `agent_context()` 规划清单）；
- `agent/AGENT-INSTRUCTIONS.md`：完整操作手册（标准决策流 + 疑难处置剧本 + 护栏三原则）；
- `agent/case_log.py`：疑难案例记录 → 人工确认 → `cases_to_dataset()` 转成演化层数据集，
  闭环「疑难 → 记录 → 提炼 → 回填核心 → 疑难变常规」；
- 权威 skill 的 `SKILL.md` 已追加浓缩版 **Agent decision protocol**。

## 快速开始

```bash
cd /Users/zhouql1978_1/dev/video-retrieval

# 0) 一键 GUI(三栏智能体界面;预热+30 分钟演化周期)
scripts/serve.sh                           # → http://127.0.0.1:8787

# 0') 智能体 CLI(--query 与 --ref 至少一个;无 LLM key 自动降级剧本)
./video/bin/python -m fusion.agent --video clip.mp4 --query "车牌号 京Q1G728"
./video/bin/python -m fusion.agent --video clip.mp4 --ref 嫌疑人.png   # 以图搜人

# 1) 全量测试(九套件,~10s,无需 key/模型)
make test                                  # → 402 passed 全绿(+测评 14)

# 2) 单模块自测(无 key/无模型)
./video/bin/python fusion/router.py        # 路由判定
./video/bin/python fusion/signals.py       # 三信号 + 无 key 弃权
./video/bin/python fusion/scorer.py        # 命中/误报投票
./video/bin/python fusion/ground.py        # 时间戳解析 + 窗口→帧

# 3) provider 冒烟（需要 key；无 key 则全部静默返 None，不报错）
VLM_PROVIDER=qwen DASHSCOPE_API_KEY=sk-... python3 fusion/provider.py

# 2) 单模块自测
python3 router.py        # 路由判定
python3 signals.py       # 三信号 + 无 key 弃权
python3 scorer.py        # 命中/误报投票
python3 ground.py        # 时间戳解析 + 窗口→帧

# 3) provider 冒烟（需要 key；无 key 则全部静默返 None，不报错）
VLM_PROVIDER=qwen DASHSCOPE_API_KEY=sk-... python3 provider.py
```

### 端到端（需要视频 + 模型权重 + 对应 key）

```bash
export SKILL_SCRIPTS=/Users/zhouql1978_1/dev/AIupdating/skills/video-target-localize/scripts

# 车牌检索（精确文本分支：本地 OCR + 混淆表，全片扫描）
python3 run.py --video /path/clip.mp4 --query "车牌号 京Q1G728 的小轿车" --out out_q1g728

# 人物图对图检索（语义分支：云生窗口 + §11.2 三信号投票，需 --ref）
VLM_PROVIDER=qwen DASHSCOPE_API_KEY=sk-... \
  python3 run.py --video /path/clip.mp4 --ref /path/ref.jpg \
                 --query "穿粉色外套白色裤子的女性" --out out_person --agree 2

# 文本语义检索（无参考图：云生窗口 + verify_target 逐 crop 核验）
python3 run.py --video /path/clip.mp4 --query "a beige coat" --out out_coat
```

## 基准测试(数据集验证,2026-08-13 就绪)

`bench/` 用 `dataset/` 四个带 GT 数据集验证算法能力,并把真实标注喂给演化层(闭环):

```bash
VENV=~/dev/video-retrieval/video/bin/python   # torch MPS venv

# 四基准(全量跑, 结果在 bench_out/*/metrics.json)
"$VENV" bench/run_prw.py      --out bench_out/prw_full      # PRW: mAP/CMC (OSNet, oracle det)
"$VENV" bench/run_ccpd.py     --out bench_out/ccpd_full     # CCPD: 检测IoU (⚠️子集无文本GT)
"$VENV" bench/run_ccpd_gt.py --limit 2000 --out bench_out/ccpd_gt  # CCPD-GT: 真 GT 识别准确率(98,459 张,分难度子集)
"$VENV" fusion/bench/run_harness_eval.py                    # 三模式测评: agent/playbook/pipeline × 3 任务同 GT 对比
"$VENV" bench/run_rstpreid.py --out bench_out/rst_full      # RSTPReid: 零样本CLIP R@k
"$VENV" bench/run_ijba.py     --out bench_out/ijba_s1       # IJB-A: 1:1 TAR@FAR + 1:N Rank

# 两条报告验收线(完整云边, 需 key)
DASHSCOPE_API_KEY=… ZHIPUAI_API_KEY=… "$VENV" bench/run_repro.py --out bench_out/repro

# 优化闭环: PRW candidates -> Optuna 阈值; 混淆表 -> gepa/直连反射
"$VENV" bench/cycle.py --out bench_out/cycle

# 汇总报告(含 SOTA 对照, 诚实标注口径)
"$VENV" bench/make_report.py --out bench_out/report

# 单测(28 项, 需 scipy)
"$VENV" bench/test_bench.py
```

**协议口径(诚实标注)**:PRW 用 oracle detections(隔离匹配质量);CCPD balance 子集文件名
已匿名化(仅检测 IoU),**识别准确率走 `run_ccpd_gt.py`(dataset/CCPD-GT 98,459 张真 GT;
200 样本实测 base 子集 exact 0.83/容错 0.90,与 SOTA >0.99 的差距如实标注)**;RSTPReid 是
零样本 CLIP vs 训练版 SOTA;IJB-A 官方 10-split。SOTA 对照见 `dataset/SOTA-METRICS.md`。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLM_PROVIDER` | `glm` | active provider；active 失败自动回退另一个 |
| `ZHIPUAI_API_KEY` / `GLM_API_KEY` | — | GLM 通道（bigmodel.cn） |
| `DASHSCOPE_API_KEY` | — | Qwen 通道（grounding 原生视频必需） |
| `GLM_VISION_MODEL` | `glm-4v-flash` | GLM 视觉模型 |
| `QWEN_VL_MODEL` | `qwen-vl-max` | Qwen 视觉/grounding 模型 |
| `QWEN_TEXT_MODEL` | `qwen-plus` | refine_query 文本模型 |
| `GROUND_VIDEO_MAX_MB` | `64` | 本地视频 base64 上限，超限返空回落 |
| `QWEN_GROUND_FPS` | `1` | DashScope 视频抽帧率 |
| `SKILL_SCRIPTS` | `…/video-target-localize/scripts` | 本地引擎（common/matcher/fast_plate_scan）路径 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | deepseek 兼容端点 | agent loop 智能体通道(思考模型已支持;GUI 配置页/对话可直接改,写 `.env`) |
| `FUSION_CONFIG` / `FUSION_CONFUSABLES` | `fusion/config/` | E0 配置层目录 / 混淆表外部文件 |
| `OPTUNA_TRIALS` / `GEPA_MAX_CALLS` | 40 / 40 | 演化预算帽(安全阀) |

## 验证对照（Phase 1 验收线，需真实数据复跑）

1. **人物查询**：候选窗口覆盖 580s/760s；`scorer` 把 OSNet 的误报区间（report-1 的 91 个）压到 `verdict=miss`，仅多信号一致帧判 hit。
2. **车牌「京Q1G728」**：grounding 返空 → 自动全片 OCR；`matches.json` 锁 838-840s（对照 `vtl_out_q1g728/intervals.json`）。
3. **回归**：Ref-DAVIS17 `box_tube_iou/temporal_iou` 不劣于 0.550/0.841（§7.2，`--no-ground` 走纯本地基线）。

## Phase 2 演化层(已产品化,`agent/evolve.py` + `config.py`)

- **Layer1 文本（gepa/反射）**：`config/prompts.json`(provider 提示词)、`config/confusables.json`(混淆表,fast_plate_scan 热读)——演化回填后**下次运行自动生效,零改代码**;CCPD-GT 提供无 key 确定性反射数据源。
- **Layer2 数值（Optuna）**：`config/thresholds.json` 叠加 `scorer.DEFAULT_THRESHOLDS`;CLI > config > 默认。留存集 20% 门禁(新 ≥ 旧 − 0.005 才采纳)。
- **Layer3 结构**：`run.py` 分支拓扑仍由人/LLM 架构师定性重构(不入自动演化)。
- **触发**:`scripts/evolve.sh` / `GET /evolve` / server `--watch 30` / launchd;全程 `evolutions.jsonl` 可查,`.veto` 冻结,rollback 快照。

### 原始接口设计(历史保留)

- **Layer1 文本（gepa）**：`provider.py` 的 4 段提示词、`fast_plate_scan.py:43-49` 混淆表、`router` 规则、`SKILL.md` 指令——都是字符串，gepa `optimize_anything` 直接当"文本梯度"优化。
- **Layer2 数值（Optuna）**：`scorer.DEFAULT_THRESHOLDS`（Δhue、REID peak、bellness、span）+ `run.py` 的 `--agree/step/window/ground-fps` 全是 CLI 可扫参数，复用同一 evaluator。
- **Layer3 结构**：`run.py` 的分支拓扑（router→ground→detect→signals→score）可由 LLM 架构师定性重构。
