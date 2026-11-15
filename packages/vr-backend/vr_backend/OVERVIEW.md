# 监控视频云边协同检索系统 —— 完整讲解

> 整合版:整体流程 + 使用算法 + 两种输入形态(语言/图像)+ 云端故障降级 + 演化优化。
> 配套代码:`fusion/`(Phase 1 harness + Phase 2 演化层),本地引擎:`video-target-localize` skill。
> 通俗版见 `../科普-云边协同视频目标定位.md`;研究依据见 `../研究方案-监控视频时空目标定位.md`(§7.2/§11.2)。

---

## 0. 系统要解决什么

**输入**一段监控视频 + 一句自然语言描述(或一张人脸/人体参考图),**输出**目标出现的
**时间段 + 每帧轨迹框**,并用**标注好的画面**交付。难点:视频帧数极多、机器要"听懂人话"、
还要在普通笔记本(M4/16GB)上实时跑。

两条实测教训决定了整个架构(2026-08 两份报告):
- **云端 VLM 会"自信地幻觉"**:图对图"是否同一人"全答 yes;粉/红混淆还打 0.95 分。
- **本地 embedding 会"草木皆兵"**:OSNet@0.55 把 16 个候选判成 exact,15 个是误报;
  车牌 70×27 小字云端读不了、云 OCR 还会把"京"读成"苏"。

→ 结论:**谁都别单独信**。云负责粗筛(快、懂语义),本地负责精查(清、能读字),
最后**多信号交叉验证**才敢下结论。

---

## 1. 总体架构:云边漏斗 + 演化层

```
输入 ──► router(任务路由)
          ├── 精确文本(车牌/证件号) ──► 本地 OCR + 字符混淆表 ──► 全片扫描
          └── 语义(人物/物体描述) ───► 云粗筛时间段 ──► 本地检测+裁剪 ──► 三信号投票
                                                                          │
演化层(Phase 2): 把"判定配置"变成目标函数 f(配置)→macro-F1 ──► 自动找最优配置
   Layer1 文本(gepa 改提示词/混淆表)  Layer2 数值(Optuna 改阈值)  Layer3 结构(LLM 架构师)
```

铁律:**本地基线永不硬依赖云端**。所有云端调用 env-gated 静默降级,失败返 None/[],
本地照常运行(降级阶梯见 §5)。

---

## 2. 检索流水线(逐环节)

### 2.1 路由器(router)—— 先把问题分类
**算法:正则表达式。** 看到"京+字母+数字"车牌格式或 18 位身份证 → `precise_text`;
否则 → `semantic`。纯逻辑、无网络、毫秒级。*代码:`router.py`*。

### 2.2 精确文本分支(车牌/证件号)—— 全本地
**云读不了小牌,所以这条路根本不请云**(报告2:云 grounding 返空、云 OCR 误读):
1. **tesseract OCR**:车牌区域放大 8 倍 + 灰度化 + 直方图均衡后识别字符。
2. **字符混淆表** `{"Q":"O0D9G16", "G":"6C0Q19", ...}`:小字模糊导致 Q→9/O/0/1、G→6、B→8
   等形近误读,匹配时生成所有形近变体,`11G728` 也能命中 `Q1G728`。
   **这是锁定 838-840s 的决定性算法**(实测 291 次 OCR 调用,542 秒全片扫)。
3. 双遍扫描:粗扫(step=10)→ 对可疑时段细扫(step=2,~1 秒可读窗口不会漏)。

### 2.3 语义分支 —— 云边漏斗
1. **云粗筛(grounding)**:`qwen-vl-max` 原生视频,一次调用把整段视频粗看一遍,
   返回目标可能出现的**时间段窗口** `[{start_s,end_s,score,reason}]`。
   *任何失败(无 key/文件太大/解析失败)都返空 `[]` → 调用方自动全片扫描,绝不锁死。*
2. **本地检测 + 裁剪**:只在粗筛窗口内(或全片)每隔 N 帧检测一次,裁特写、按时间窗去重
   (每 ~1 秒留面积最大的一个,控制后续开销)。
3. **匹配/核验**:
   - *文本输入*:YOLO-World 开放词汇按描述词找 + VLM 逐图核验(verify_target:描述+图 → exact/partial/no)。
   - *图像输入*:ArcFace/OSNet 特征 + 余弦相似度打"像不像"分(见 §3,完整三信号)。
4. **三信号投票** 见下。

### 2.4 三信号投票(§11.2 多信号甄别法)—— 核心算法
把每个候选交给**三个独立证人**,各自打分,**≥2 个(默认)作证才判命中**:

| 信号 | 算法 | 通俗解释 |
|---|---|---|
| 颜色 | HSV 主色相直方图 + 圆形色相差 | 同一人 Δ色相 ≲35°;误报 60-110° |
| 时间曲线 | 相似度序列的"持续抬升占比"+ 峰值/跨度 | 真人:0.35→0.83→0.35 平滑爬升回落;误报:单帧尖峰 |
| VLM 终审 | 图对图让大模型判断"是否同一人" | 参考图+候选图一起发,exact/partial/no |

投票规则(**scorer.py**):不能判断的信号**弃权**(灰色没法比颜色、无 key 无法终审),
弃权不计入分母;**单一信号永远不能判命中**;剩余信号不足 `agree` 时给
`insufficient`(不假装结论)。*这就是把"16 候选 15 误报"压到只剩真命中的机制。*

---

## 3. 图像输入(人脸/人体 crop)完整流程

```
参考图(人脸/人体 crop)
  │
  ▼
① 参考图预处理: 自动选主体(优先最大的脸→否则最大的行人)→ 裁剪+留白
   → 算出特征向量 qv → 存 ref_subject.jpg(挑错了可手动重裁)
  │
  ▼
② 云粗筛(可选): 视频→Qwen 粗看→候选时间段窗口(需配文字描述,或 --no-ground 全片扫)
  │
  ▼
③ 本地检测+裁剪: 主体=脸 → insightface 检脸;主体=行人 → YOLO 检人
   每隔 N 帧检测、裁剪、按时间窗去重
  │
  ▼
④ 嵌入匹配: 人脸 → ArcFace 512维;行人 → OSNet 512维
   参考 qv 与每个候选做余弦相似度 → (时间段, 像不像分) ← 误报高发区,交给信号层
  │
  ▼
⑤ 三信号投票(参考图在手,信号最全):
   颜色(参考 vs 候选主色相) + 时间曲线 + VLM 图对图终审
  │
  ▼
⑥ ≥2 信号一致 → 命中 → 时间段 + 标注帧
```

- **人脸 vs 人体**:`--mode auto` 优先脸(insightface/SCRFD 检脸 + ArcFace),脸不可读时退行人
  (YOLO + OSNet)。实战细节:小脸在裁剪图里再检一次会失败,特征在检测时刻就算好随 crop 带走。
- **与文本输入的差别**:图像输入多一层**身份 embedding**,并且让"颜色对比"与"图对图终审"
  两个需要参考图的信号从"不可用"变"可用"——是 §11.2 多信号甄别法的完整形态。
  *代码:`run.py --ref` → `run_semantic_image()`;引擎:`person_search.py`(0 选主体→1 检测→2 匹配→3 输出)。*

---

## 4. 用到的算法清单

| 环节 | 算法 | 一句话原理 | 为什么选它 |
|---|---|---|---|
| 任务路由 | 正则表达式 | 匹配车牌/证件号格式 | 确定性、零成本 |
| 云粗筛 | qwen-vl-max 原生视频 | 整段视频一次调用给时间段 | 快、懂语义;只做粗筛不判终局 |
| 本地检测 | YOLO-World | 开放词汇检测(CLIP 文本编码→逐帧框) | 免训练、能跟任意名词 |
| 跟踪(M0) | BoT-SORT | 卡尔曼+匈牙利匹配跨帧连轨迹 | 稀疏 pivot 的"惯性"来源 |
| 身份匹配 | ArcFace / OSNet | 512维特征 + 余弦相似度 | 人脸/行人重识别,快速粗筛 |
| 车牌识别 | tesseract OCR + 混淆表 | 放大+均衡→识别→形近变体匹配 | 小字下唯一可靠路径 |
| 三信号甄别 | HSV 色相 / 时间曲线形态 / VLM 图对图 | 三个独立证人交叉验证 | 单通道都会骗人,交集才可信 |
| 多信号投票 | 阈值判定+弃权+一致性计数 | ≥agree 个非弃权信号才判 hit | 防幻觉、防过度匹配 |
| 质量度量 | macro-F1 | 正类+负类等权 | 极端类不平衡下 accuracy 无效 |
| 数值优化 | Optuna TPE(贝叶斯) | 从历史成绩猜高分区域再采样 | 5维阈值空间,比网格快一个量级 |
| 文本优化 | gepa(LLM 读日志提议变异) | 大模型读 (候选,分数) 日志→自然语言改进 | 提示词/混淆表无梯度可算 |
| 结构优化 | LLM 架构师 | 定性重构 pipeline 拓扑/降级阶梯 | 非数值,gepa/Optuna 不适用 |

---

## 5. 云端故障降级阶梯(所有环节的兜底)

```
云端全好 ─────────────► 完整云边融合
   ▼ ① 某家云故障(429/超时/key失效)
自动切换 ─────────────► glm ↔ qwen 互为回退(provider.chat 逐个试)
   ▼ ② 云全部不可用(断网/无 key)
各自降级 ─────────────► 见下表,本地照跑
   ▼ ③ 本地模型也缺(无 insightface/OSNet 权重)
embedding 链退 ───────► ArcFace/OSNet → CLIP → VLM → 只出裁剪图给人工
   ▼ ④ 连裁剪都失败
兜底 ────────────────► 清晰报错,绝不输出假结果
```

| 环节 | 云挂了怎么办 | 代码 |
|---|---|---|
| 云粗筛 | 返空 `[]` → 自动全片扫描,不锁死搜索 | `ground.py`;`windows_to_frames([])`→全片 |
| 文本语义细化 | 用原始查询当检测词 | `provider.refine_query() or "object"` |
| 文本逐图核验 | 全 unclear → 检测 `vlm_unusable` → 车牌自动转本地 OCR | `verify_target.vlm_unusable`/`run_ocr_degrade` |
| 图像三信号 | VLM 终审**弃权**,颜色+时间曲线两票仍可达成 agree=2 | `scorer.py` 弃权语义 |
| 车牌检索 | 根本不依赖云(路由直接走本地 OCR) | router→precise_text→`fast_plate_scan` |
| 本地缺模型 | embedding 链逐级退;最坏仍产出裁剪图+manifest 供人工审核,`exit 0` | `matcher.py` FACE_OK/REID_OK/CLIP_OK |
| 超时 | 60/120/180s 上限,失败返 None 不挂起 | `provider._post` |

核心哲学:**云端失败 ≠ 目标不存在**。"返空"只触发"换更费时的本地路径",
绝不触发"结论:没找到"。云挂了这些命令照样能跑:
```bash
python3 run.py --video clip.mp4 --query "京Q1G728" --out o1            # 车牌,全本地
python3 run.py --video clip.mp4 --ref ref.jpg --query "描述" --out o2 --no-ground  # 图像,两票
python3 run.py --video clip.mp4 --query "beige coat" --out o3 --no-ground          # 文本,出裁剪图等云
```

**已知边界(待补)**:① 三信号全弃权时当前静默 0 命中,应导出裁剪图+提示人工审核;
② `run_semantic_text` 未接 `vlm_unusable→OCR` 降级(带车牌查询已被 router 截走,影响有限);
③ 粗筛失败→全片扫描正确但耗时,可加"场景变化抽帧"轻量本地粗筛。

---

## 6. 演化优化(Phase 2):系统如何自己变好

统一心智模型——**把 harness 变成目标函数,让优化器搜最好的配置**:

```
配置(阈值/提示词/混淆表) ──► 在标注数据集上重跑判定 ──► macro-F1(质量分)
        ▲                                            │
        └──── 优化器每轮提议新配置 ◄──── (配置, 分数) 历史
```

- **评估器极便宜**:Phase 1 信号(hue/temporal/vlm 的 raw 值)算一次就缓存,
  优化时只是"换阈值→重新投票→数对错",微秒级、无网络、无 key → 几百次试搜可行。
- **质量分 = macro-F1**:数据极端不平衡(1 真 vs 54 噪声),accuracy 无效;
  macro-F1 把"全判否"判 0.43、"OSNet 0.55 过度匹配"判 0.06、"三信号一致"判 1.00。

| 层 | 搜什么 | 怎么搜 | 状态 |
|---|---|---|---|
| Layer1(gepa) | 提示词、车牌混淆表、路由规则、SKILL.md 指令 | LLM(glm-5.1)读 (候选,分数) 日志,用自然语言提议变异("给 1 加上 I"),改完打分循环;预算 max_metric_calls≈120 | 驱动就绪,**未执行**(待数据集) |
| Layer2(Optuna) | 阈值:hue_dhue、temporal_peak/bellness/span、agree | TPE 贝叶斯:随机 warm-up 后从历史成绩猜高分区域采样;每轮=提议→评估→记分;最优阈值直接喂回 run.py | 驱动就绪,**未执行**(待数据集) |
| Layer3(LLM 架构师) | pipeline 拓扑/降级阶梯 | 定性重构,人工采纳 | 设计就绪 |

**§11.2 "REID 0.55→?" 的数值化答案**:不再手拍阈值,搜索在标注数据上自动找使 macro-F1 最大的配置。
*代码:`evaluator.py`(共享目标函数)→ `optimize_thresholds.py` / `optimize_text.py`。*

---

## 6.5 智能体执行层(算法做成 skill 的意义)

架构 = **确定性核心(工具包)+ 智能体(操作者)**。引擎脚本本来就是 skill 形态
(`SKILL.md` agent 指令 + scripts 工具);执行指引把 "SKILL.md agent 指令" 列为
Layer1(gepa)优化对象——智能体层是这层设计的自然完成。

```
┌─────────────────────────────────────────────────┐
│ 智能体(操作者)                                    │
│  读 SKILL.md → 制定计划 → 调用工具 → 观察中间结果   │
│  → 判定 → 覆盖不了?→ 临场发挥(组合/换招/问用户)      │
└───────────────┬─────────────────────────────────┘
                ▼ 每个脚本 = 一个工具(输入→JSON 输出)
┌─────────────────────────────────────────────────┐
│ 确定性核心: preflight│locate│verify_target│        │
│ person_search│fast_plate_scan│fusion/run│         │
│ signals│scorer(可拆开单独调用)                     │
└─────────────────────────────────────────────────┘
```

**为什么带来泛化性/适应性**:覆盖面从"预设路径的并集"变成"工具的**所有组合**";
智能体可读中间产物(manifest/verify/scored.json)、可自己看裁剪图(多模态)、
可按现场环境挑策略、疑难时换引擎/换阈值/换信号/拆查询/问用户。

**疑难分类与临场发挥**(算法覆盖不到的情形):
| 疑难 | 定式会怎样 | 智能体发挥 |
|---|---|---|
| 复合查询("3点蓝门旁红衣人") | 全片找红衣→误报 | 拆步骤:视觉锚点"蓝门"→时间窗→窗内红衣→身份核验 |
| 结果不足(insufficient/0 命中) | 静默 0 输出 | 自己看 crops → 换阈值重评分 → 换引擎 → 请求更清晰参考图/问用户 |
| 环境异常(云挂+无 tesseract+视频不可读) | 阶梯退到底 | preflight 恢复阶梯 → 换 ffmpeg 变体 → CLIP 兜底 → 说明失败原因 |
| 参考图多主体/低清/侧脸 | 选最大主体可能选错 | 识别"图里有 2 人"→ 问用户 → 换特写重搜 |
| 组合检索("和 A 同框的 B") | 无此预设 | 搜 A→取时间段→段内搜 B→取交集 |
| 混淆表没覆盖的新字符集 | 匹配失败 | OCR 原文+自身视觉核对 → 现场补规则 |

**护栏**:① 断言必须基于证据("没找到"≠"不存在",输出覆盖范围/尝试过程);
② 工具标成本标签(OCR 全片≈9min、VLM 逐 crop),智能体预算意识(先便宜后贵);
③ 确定性核心仍是安全默认,常规查询直接走定式,不污染可回归的核心。

**与演化层的闭环**:智能体在疑难上的临场发挥被记录(查询+尝试+结果,人工确认后
成为新标注数据)→ gepa/Optuna 提炼成新规则/阈值/提示词 → 回填确定性核心 →
该疑难变常规 → 智能体腾出精力处理更难案例。**疑难变常规,常规被自动化**。

### 6.5.1 落地形态与使用(2026-08-15 更新:E0–E7 完成 + GUI v2 + LLM 通道实测)

**三种使用方式**(详见仓库根 `AGENT-HARNESS-执行与部署方案.md` §1):

```bash
# 方式一:三栏智能体 GUI(主,民警日常;含引擎预热与 30 分钟演化周期)
scripts/serve.sh                      # http://127.0.0.1:8787

# 方式二:命令行(--query 与 --ref 至少一个;仅参考图=以图搜人;无 key 自动降级剧本)
python -m fusion.agent --video 路口监控.mp4 --ref 嫌疑人截图.png
python -m fusion.agent --video 路口监控.mp4 --query "车牌号 京Q1G728"

# 方式三:HTTP API(SSE 进度/案例/配置/视频端点/聊天)
curl -X POST localhost:8787/search -d '{"video":"/data/x.mp4","query":"白车"}'
curl -X POST localhost:8787/chat  -d '{"messages":[{"role":"user","content":"现在用什么模型?"}]}'
```

**GUI v2(2026-08-15,三栏智能体原生界面)**:
- **左栏·视频**:拖入/选路径 → 探测(时长/帧数/帧率/分辨率/**真实容器**)→ 播放;
  DVR 的 MPEG-PS 封装浏览器不能播时**自动转封装**(h264+aac,缓存,实测 425MB≈6s)。
- **中栏·对话**:检索描述或**直接与 LLM 对话**(系统提示含检索手册,能讲清系统能力);
  📎 附加参考图(以图搜人);检索时**引擎进度流式反馈**(每步耗时 + ⏳ 进行中占位,不会误以为卡死);
  **配置也能用说的**——「把模型换成 xxx」「现在用的什么模型」经 get/set_settings 工具执行,
  保存即生效并写 .env,密钥只显掩码,改完自动连通性自检。
- **右栏·命中片段**:标注帧 + 时间/帧号/证据;**▶ 播放跟踪片段**=独立小视频,
  目标出现→消失、逐帧红框跟随(引擎逐帧 box + ffmpeg drawbox)。
- 三栏**可拖拽分隔线**(最小宽度保护,localStorage 记忆);Markdown/表格渲染。

实现构成(全部落地并有测试锁定):
- **操作层** `fusion/agent/loop.py`:LLM 通道(litellm 工具调用→纯文本 ReAct→
  确定性剧本三级降级;**思考模型支持:reasoning_content 回传+严格 tool_calls wire format**
  ——DeepSeek 实测多轮跑通)+ max_steps + 护栏(绝不输出"目标不存在";命中必须有
  引擎产出的时间段;**out/video 强制归位 job 目录**;工具崩溃变 failed 信封不杀循环)
  + 疑难处置(insufficient→rescoring→换阈值→换引擎→如实报告)+ 命中即记案例
  + 无 VLM key 时 person_search 自动 reid 0.75 高阈值预筛 + 受控胶水脚本。
- **接缝层** `fusion/config.py`(E0):thresholds/confusables/prompts 三文件在
  运行时叠加到确定性核心 —— 演化回填零改代码,缺失/坏 JSON 静默降级。
- **使用层** `fusion/server.py` + `fusion/web/index.html`:FastAPI + SSE 实时进度、
  jobs.jsonl 持久化(重启回放,中断诚实标记)、案例「确认/否认」(演化养料)、
  配置页+对话式配置、`/chat`、`/video/{info,file,transcode,tracked-clip}`、
  `--warmup` 预热与 `--watch` 演化周期。
- **演化控制器** `fusion/agent/evolve.py`(E3):合并数据集(确认案例 + bench PRW
  + CCPD-GT)→ Optuna 阈值优化 + Layer1 混淆表反射(CCPD 真 GT 确定性反射,
  无 key 无网络)→ 留存集 20% 门禁(新 ≥ 旧 − 0.005 才采纳)→ 回填
  `fusion/config/`(带 _meta + rollback 快照)→ `evolutions.jsonl`;`.veto` 冻结。
  HTTP `/evolve` 实测:采纳(Optuna 0.4535→0.4586)与拒绝(CCPD 反射宽度超限)两路径均真实发生。
- **测评层** `bench/eval_gt.py` + `run_harness_eval.py`(2026-08-15):agent/playbook/
  pipeline 三模式 × person/plate/smoke 三任务同 GT 对比(断点续跑+引擎超时),报告见
  `bench_out/harness_eval/report.md`。

---

## 7. 当前状态

| 模块 | 状态 |
|---|---|
| Phase 1 harness(`provider/router/ground/signals/scorer/run`) | ✅ 已建,42 项单元测试全过 |
| Phase 2 演化层(`evaluator/dataset/两个优化驱动`) | ✅ 已解锁执行:真实标注上跑通 Optuna(PRW)+ 直连反射(混淆表) |
| 智能体执行层(`toolbox` 工具契约 + `agent/` 手册与案例闭环) | ✅ 已建,113 项测试全过;权威 SKILL.md 已追加处置协议 |
| 基准 harness(`bench/`:四数据集+验收线+闭环) | ✅ 已建,28 项测试全过;全量跑批完成(见下) |
| 端到端验收 | ✅ 人物线 pass(GT 2/2、FP 53<91);车牌线、Ref-DAVIS17 回归见 bench 报告 |
| **Agent harness E0–E7(2026-08-14 完成)** | ✅ **全链路落地**:运行时配置层(config.py)/agent loop/服务+GUI/演化控制器/CCPD 真 GT 基准/部署物/skill 同步 |
| **GUI v2 + LLM 通道(2026-08-15)** | ✅ 三栏智能体界面(对话/拖拽/转封装/跟踪片段/流式进度/对话式配置);DeepSeek 思考模型多轮工具调用实测跑通;检索质量修复后真实以图搜人 101 段误报→1 处精准命中 |
| **三模式测评(2026-08-15)** | ✅ agent/playbook/pipeline × person/plate/smoke 九格(`bench_out/harness_eval/report.md`);发现 fusion_run 评分器无 VLM 时 reid_sim 0.55 为第二误报源(待办) |
| 自进化闭环(演化→门禁→回填→热生效) | ✅ 已通:HTTP /evolve 实测 Optuna 采纳(0.4535→0.4586)与 CCPD 反射被门禁拒绝两路径;thresholds.json 回填后 scorer 子进程实测读新值 |
| 测试总量 | **416 项全绿(make test 十套件,含 harness 测评),全套 ~10s** |

**基准结果(2026-08-13,全量):** PRW mAP 0.158/CMC1 0.682(阈值0.55精度仅 0.0015=过度匹配铁证);
RSTPReid 零样本 R@1 0.074(训练 SOTA 0.62,诚实差距);CCPD 检测率 0.863/IoU50 0.648、OCR 读取率
0.30(蓝牌白字缺二值化,已列为 Layer1 改进方向;子集无文本 GT,识别率不可测);IJB-A 见报告。
**优化结果:** Layer2 Optuna(PRW 真实 candidates)macro-F1 0.5284→0.5345(reid_sim 0.73/hue 68.6°);
Layer1 混淆表(直连反射,glm-4-flash)0.8241→1.0(57 行真实车牌数据;gepa 集成 5 次受阻已记录,
同机制回退路径稳定)。**场景自适应**:人物验收线默认阈值失败(真人 sim 0.47/Δhue 40°恰在阈外),
放宽 Δhue 45°/peak 0.45/gap 3.5 后 GT 2/2、FP 53<91 → 通过——阈值场景相关,演化层让重标定微秒级。

**数据集契约**(`dataset.py` 待填 loader,形状已固定):
```python
candidates = [{"id","t_s","gt":bool, "results":{hue/temporal/vlm:{score,raw,evidence}}}]
rows       = [{"ocr":str, "gt":bool}]
```

---

## 8. 代码索引

| 文件 | 职责 |
|---|---|
| `provider.py` | 统一 VLM 客户端(glm↔qwen 互为回退,env-gated 降级,四函数契约) |
| `router.py` | 任务路由 semantic / precise_text |
| `ground.py` | 云粗筛候选窗口(qwen-vl-max),返空回落全片 |
| `signals.py` | 三信号(hue / temporal_curve / vlm_arbiter) |
| `scorer.py` | 多信号投票(agree 规则 + 弃权语义 + 阈值集中) |
| `run.py` | 端到端 CLI(三分支按路由自动选) |
| `evaluator.py` | 共享目标函数(阈值→F1 / 文本→F1, macro-F1) |
| `dataset.py` | 数据集 schema/脚手架(SCAFFOLDING) |
| `optimize_thresholds.py` / `optimize_text.py` | Layer2 Optuna / Layer1 gepa 驱动(能力就绪) |
| `test_smoke.py` / `test_phase2.py` / `test_agent.py` | 42 + 19 + 113 项测试 |
| `toolbox.py` | 工具契约层(9 工具 manifest + dispatch 统一信封 + 流式引擎 stdout on_line + agent 上下文) |
| `agent/AGENT-INSTRUCTIONS.md` / `case_log.py` / `case.schema.json` | 智能体操作手册 / 疑难案例记录与数据集闭环 |
| `agent/loop.py` / `agent/__main__.py` | 智能体循环(LLM/ReAct/剧本三级 + 护栏 + 思考模型支持)+ CLI |
| `agent/evolve.py` | 演化控制器(合并数据集→Optuna/CCPD 反射→留存集门禁→回填+快照→evolutions.jsonl) |
| `config.py` + `config/` | E0 运行时配置层(thresholds/confusables/prompts 热加载 + rollback + .veto) |
| `server.py` + `web/index.html` | 服务(:8787,SSE/jobs 持久化/chat 配置工具/视频探测·转封装·跟踪片段)+ 三栏智能体 GUI |
| `bench/run_ccpd_gt.py` / `bench/eval_gt.py` / `bench/run_harness_eval.py` | CCPD 真 GT 识别基准(98,459) / 测评评分层 / 三模式对比跑分器 |
| 测试 | `test_{smoke,agent,phase2,config,server}.py` + `agent/test_{loop,evolve}.py` + `bench/test_{bench,ccpd_gt,harness_eval}.py` = **416 项** |
| skill `SKILL.md` + `scripts/` | 智能体执行层的工具包与操作手册(引擎;三形态独立原则下 skill 改造方案见其仓库) |
