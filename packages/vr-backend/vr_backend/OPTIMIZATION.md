# 它如何优化 —— Phase 2 演化层机制说明

> **状态(2026-08-15)**:本文的三层机制已产品化为 `agent/evolve.py`(控制器)+ `config.py`(回填接缝)。
> 真实运行记录:HTTP `/evolve` 实测 Optuna 采纳(holdout macro-F1 0.4535→0.4586 过门禁回填 thresholds.json,
> scorer 子进程热读新值)与 CCPD 真 GT 反射被门禁诚实拒绝(容错 +0.05 但误判宽度超限)两路径;
> 数据源=确认案例 + PRW candidates + CCPD-GT(98,459 真 GT)。`.veto` 冻结 / rollback 快照 / evolutions.jsonl 全程可查。
>
> 回答你的问题：「它到底是怎么做优化的？」
> 一句话：**把 Phase 1 的 harness 变成一个目标函数 `f(配置) → 质量`，然后让两层优化器去搜使它最大的配置。** Layer 2 搜「数字」，Layer 1 搜「文字」，Layer 3 改「结构」。三层共用同一个 evaluator。

## 0. 统一的心智模型

```
                  ┌─────────────────────────────────────────────┐
   配置 (待优化) ──►│  evaluator (evaluator.py)                   │──► macro-F1 (质量)
   · 数字阈值      │  拿配置在【标注数据集】上重跑 harness 的判定  │
   · 提示词/混淆表  │  对比 GT，算 P/R/F1                          │
   · pipeline 结构 │  （纯计算，不调云、不跑 CV）                  │
                  └─────────────────────────────────────────────┘
       ▲                                                        │
       │ 优化器每轮提议一个新配置                                  │
   ┌───┴──────────────────────────────────┐                     │
   │ Layer2 Optuna  Layer1 gepa  Layer3 LLM │◄─── 把 (配置, F1) 历史喂回去当"梯度"
   └───────────────────────────────────────┘
```

关键洞察：**评估器极其便宜**。Phase 1 的信号（hue/temporal/vlm 的 `raw` 值）一旦算出来就缓存了；优化时只是反复「换阈值 → 重新投票 → 数对错」，**微秒级、无网络、无 key**。这让几百次试搜变得可行。

## 1. evaluator.py —— 整个优化的地基

```python
# Layer 2：阈值 → F1
make_threshold_objective(candidates) -> f(thresholds) -> macro_f1
#   把每个候选已缓存的信号 raw 用新阈值过一遍 scorer.score_candidate，
#   比对候选的 gt 标签，返回 macro-F1。

# Layer 1：文本(混淆表) → F1
make_text_objective(rows, target) -> f(table_text) -> macro_f1
#   把候选混淆表 JSON 解析回来，重跑 ocr_matches 对每条 OCR 读取，
#   比对 gt，返回 macro-F1。
```

**为什么是 macro-F1，不是 accuracy？** 两份报告都是极端类不平衡（车牌：1 条真读 vs 54 条噪声；人物：1 个真人 vs 15 个 OSNet"exact"误报）。accuracy 毫无用处——"全判否"就有 96%。macro-F1 把正类（稀有真命中）和负类（正确拒绝）等权：

| 配置 | 正类 P/R/F1 | 负类 F1 | macro-F1 |
|---|---|---|---|
| "全判否"（平凡拒绝器）| 0 / 0 / 0 | 0.87* | **0.43** ← 暴露它 |
| OSNet 阈值 0.55（report-1 基线，16 选 1）| 0.06 / 1 / 0.12 | 0 | **0.06** ← 暴露过度匹配 |
| §11.2 三信号 agree=2（1 TP, 0 FP）| 1 / 1 / 1 | 1 | **1.00** |

\*"全判否"也会把 3 个真目标拒掉，所以负类 F1 不是 1。macro-F1 精准地把三种情况拉开——这正是优化要的最大化目标。测试里已验证这个单调性（`test_phase2.py`）。

## 2. Layer 2 —— Optuna 搜数字（`optimize_thresholds.py`）

**搜什么**（`SEARCH_SPACE`，全是 Phase 1 的判定阈值，已 CLI 化）：
```
hue_dhue ∈ [15, 80]        # §11.2 同人线 35°；report: 误报在 60-110°
temporal_peak ∈ [0.4, 0.8]  # OSNet 相似度闸门（report 用 0.55）
temporal_bellness ∈ [0.1,0.7]
temporal_span_s ∈ [0, 2]
agree ∈ {1, 2, 3}           # 几个信号一致才判 hit
```

**怎么搜**（机制）：
1. Optuna 先随机撒几个点 warm-up；
2. 之后用 **TPE（贝叶斯）**：根据已有 `(阈值, F1)` 历史建一个概率模型，猜"哪种阈值区域 F1 高"，在那种区域多采样——5 维空间里比网格/随机快一个量级；
3. 每轮 = `suggest 一个点 → 调 evaluator → 记 F1`；
4. 跑完 N 轮，F1 最高的那组的阈值就是「优化后配置」，直接喂回 `run.py --hue-dhue … --agree …`。

**这就是 §11.2 "REID 0.55→?" 的数值化答案**：不再手拍 0.65，而是让搜在标注数据上找到使 macro-F1 最大的那个数。

> 状态：`run()` 已就绪但**未执行**。需要 `pip install optuna` + 真实标注数据集。当前 `test_phase2.py` 只验证"单点评估"和"参数映射"正确。

## 3. Layer 1 —— gepa 搜文字（`optimize_text.py`）

**搜什么**（文字工件，当"参数"）：
- **车牌字符混淆表** `fast_plate_scan.py:43-49`（`{"Q":"O0D9G16",...}`）—— report-2 锁定 838-840s 的决定性因素，序列化成 JSON 文本让 gepa 改。
- **VLM 提示词**（verify_target / match_image 的 prompt）。
- （后续）router 规则、SKILL.md agent 指令。

**怎么搜**（机制，与 Layer 2 本质不同）：
1. 种子 = 当前混淆表文本；
2. gepa 把历史 `(候选表, F1)` 日志喂给一个 **reflection_lm（glm-5.1）**，LLM 读日志后**提议一个变异**——比如"最近 3 个漏检都是 1 被读成 I，给 1 加上 I"——这相当于用自然语言当**"文本梯度"**；
3. evaluator 给新表打 F1；
4. 留最好的。重复 `max_metric_calls` 次（指引预算 ~120）。

**成本差异**：混淆表目标每轮是纯字符串匹配（无 VLM），预算能跑很远；提示词目标每轮是真 VLM 调用，预算就是真金白银（~4-5k 调用，免费档内）。

> 状态：`run()` 已就绪但**未执行**。需要 `pip install gepa` + LLM key。evaluator（`make_text_objective`）已接线并通过测试——验证方式：删掉表里的 `"1"` 条目，F1 的 recall 应下降（测试已覆盖）。

## 4. Layer 3 —— LLM 架构师改结构（定性，非数值）

不搜参数，而是让 LLM **重构 pipeline 拓扑/降级阶梯**本身——比如"把 VLM 终审从逐 crop 改成批审"、"grounding 返空时先按场景切换再全扫"。gepa/Optuna 不适用（没有可微/可搜的标量参数），靠架构师定性提案 + 人工采纳。`run.py` 的分支结构（router→ground→detect→signals→score）就是它的操作对象。

## 4.5 智能体疑难案例 → 数据集的闭环（case_log）

智能体在算法覆盖不到的疑难上的**临场发挥**，正是演化层最珍贵的数据源：

```
智能体疑难处置 ──► case_log.record_case(查询+计划+尝试+结果)
                     │  人工确认 + 填 GT (hit_intervals / positive_reads)
                     ▼
              cases_to_dataset() ──► 转成 dataset.py 契约 (candidates / rows)
                     │
                     ▼
        gepa/Optuna 提炼 ──► 新规则/阈值/提示词回填确定性核心
                     │
                     ▼
           该疑难变"常规" ──► 智能体腾出精力处理更难案例 (循环)
```

协议与 schema：`fusion/agent/case.schema.json`（JSONL 每行一条），`case_log.py` 提供
校验/写入/读取/转换（113 项测试覆盖，含未确认案例不入数据集的过滤）。

## 5. 现在能做 vs 待数据集

| 能做（已验证） | 待数据集到位后做 |
|---|---|
| evaluator 在脚手架上算 F1、单调性正确 | 在真实标注上跑 `optimize_thresholds.run()` 出最优阈值 |
| 混淆表目标的增删反应正确 | 跑 `optimize_text.run()` 出优化后的表/提示词 |
| 两驱动单点评估 + 参数映射正确 | 对照两报告复现+超越（838-840s、580/760s） |
| 驱动在缺库时优雅拦截（清晰 ImportError） | macro-F1 提升曲线、Pareto 对照 §7.2 回归线 |

**数据集契约**（`dataset.py` 待填的真实 loader，形状已固定）：
```python
candidates = [{"id","t_s","gt":bool, "results":{hue/temporal/vlm:{score,raw,evidence}}}]
rows       = [{"ocr":str, "gt":bool}]
```
当前 `dataset.py` 用报告公布的数字 + 现有 eval 产物（`vtl_fastscan_out/all_ocr.json`）提供**脚手架**，只用于接线验证——不读出任何"已优化"结论。
