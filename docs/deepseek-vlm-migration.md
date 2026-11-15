# 方案：用 DeepSeek V4-Flash-Vision 替换 qwen/glm（dsh-video-retrieval）

> 目标：把检索管线里的 VLM 调用从 「GLM(glm-4v-flash) + Qwen(qwen-vl-max)」迁移到
> **DeepSeek V4-Flash-Vision-Exp**，在同等甚至更高精度下大幅降本，并按阶段 A/B 验证性能是否有提升。
> 现状：agent 循环 LLM 已经用 DeepSeek（`LLM_BASE_URL`/`LLM_MODEL`），但 **视觉管线**（`server/provider.py`）
> 仍单独走 `DASHSCOPE_API_KEY` / `ZHIPUAI_API_KEY`。

---

## 1. 现状：VLM 在管线里到底被谁调用

| 调用点 | 文件 | 输入 | 当前模型 | 作用 | 属于哪一分支 |
|---|---|---|---|---|---|
| `refine_query` | `server/provider.py:239` | 文本 | `glm-5.1`/`qwen-plus`（**纯文本**） | 自由描述 → YOLO-World 类别短语 | semantic(全部) |
| `ground_candidates` | `server/ground.py:192` | **视频** | `qwen-vl-max`（**原生 `video_url`**） | 云端粗定位时间窗（漏斗宽口） | semantic |
| `verify_target` | `server/provider.py:280` | 单张裁剪图 | `glm-4v-flash`/`qwen-vl-max` | 判断裁剪图是否匹配描述 + 读取车牌/标识 | semantic(text/image), precise_text |
| `match_image`→`vlm_arbiter` | `server/signals.py:158` | 两张图(参考+候选) | `glm-4v-flash`/`qwen-vl-max` | 图对图身份终审（三信号中的决定性信号） | semantic+ref |
| `disambiguate` | `server/provider.py:253` | 多张裁剪图 | `glm-4v-flash`/`qwen-vl-max` | 从多个候选挑最佳（种子选择） | 底层引擎 |

**关键上下文**：`glm-4v-flash` 是免费档，但实测弱（很多账号付费档 `glm-5v`/`glm-4v-plus` 直接 `429 余额不足`），
所以生产里视觉调用**实际大多落到 `qwen-vl-max`**。上一轮报告明确：VLM 在身份判断上「自信幻觉——是不是同一个人→全 yes」，
因此视觉仲裁是**最大的精度隐患**，也是最值得用强模型替换的环节。

---

## 2. DeepSeek V4-Flash-Vision-Exp 能力与约束（已核对官方文档）

- **模型 id**：`deepseek-v4-flash-vision-exp`
- **Base URL（OpenAI 格式）**：`https://api.deepseek.com` → `/chat/completions`（与现有 `_post` 完全兼容）
- **图像输入**：原生 OpenAI 兼容的 `content` 数组，支持
  (a) base64 `data:image/...` data-URL、(b) 外部 http(s) URL、(c) Files API `file_id`（≤64 MiB）
- **图像格式**：JPEG / PNG / GIF / WebP
- **⚠️ 不支持原生 `video_url`**（文档只列图像输入，没有视频块）——这是与 Qwen 的**唯一硬差异**，影响 `ground_candidates`
- **推理模式**：支持 thinking / non-thinking（默认 thinking）；注意 FIM 对 vision 不支持
- **计费**（每 1M token，错峰/高峰；图像按尺寸折算为输入 token）：
  - 输入 cache-miss：`$0.22` / `$0.44`
  - 输入 cache-hit：`$0.007` / `$0.014`
  - 输出：`$0.66` / `$1.32`
  - 同代文本模型 `deepseek-v4-flash` 同价（更便宜档位）

> 结论：`deepseek-v4-flash-vision-exp` 是「接近前沿（类 Opus-4.8）的视觉推理 + Flash 级价格」，
> 相比 `qwen-vl-max` 是**同精度档、爆炸性降价**；相比免费的 `glm-4v-flash` 则**精度大幅提升**（需真实测）。
> 参考：[Vision API 指南](https://api-docs.deepseek.com/guides/vision/)、[模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)、[上线公告](https://api-docs.deepseek.com/zh-cn/news/news260821/)。

---

## 3. 分级替换方案（按收益/成本排序）

### A【推荐先做，收益最大、改动最小】替换图片级三处：`verify_target` / `match_image` / `disambiguate`

这些都是 `image_url` 单图或多图请求，DeepSeek 原生兼容，**零重写**。只需在 `provider.py` 加一个 `deepseek` 分支。

- 改动点：`server/provider.py`
  - `DS_BASE_URL = "https://api.deepseek.com/chat/completions"`
  - `DS_VISION_MODEL = "deepseek-v4-flash-vision-exp"`、`DS_TEXT_MODEL = "deepseek-v4-flash"`
  - `_ds_key()` → `DEEPSEEK_API_KEY`
  - `_cfg("deepseek")`、`_order()` 把 `deepseek` 加进 try-order、`active_provider()` 允许 `deepseek`、`provider_status()` 增加 `deepseek` 键
  - 把 `VLM_PROVIDER` 的合法值从 `glm|qwen` 扩到 `glm|qwen|deepseek`
  - `_model_known()` / `_substitute_model()` 增加 deepseek 前缀匹配（`deepseek`/`deepseek-`）

**为什么这里收益最大**：`verify_target`（每候选裁剪都要跑）和 `vlm_arbiter`（决定三信号投票是否命中）是精度的
**决定性环节**，也是之前「全 yes 幻觉」的源头。换成前沿视觉推理模型，直接冲「精度 / 读数准确率」。

### B【几乎纯赚】替换文本调用：`refine_query` → `deepseek-v4-flash`

`refine_query` 是纯文本、每查询一次、极便宜。用 DeepSeek 文本模型（最便宜档）替换 `glm-5.1`/`qwen-plus`。
改动点：`provider.refine_query` 里 `model="glm-5.1", kind="text"` 改为走 deepseek 文本模型。成本可忽略，质量只会更好。

### C【唯一不能无损替换】云 grounding：`ground_candidates`（原生视频）

DeepSeek 目前**不接受 native `video_url`**，所以这是唯一需要改造的点。两个方向：

1. **改为帧采样（推荐先试）**：按 `fps` 抽 K 帧 → 转 JPEG → 作为**图像数组**（多个 `image_url` 块）
   一次性发给 DeepSeek，让它输出时间窗。与原 bbox 无关，但会**丢失原生视频的连续性信息**。
   - 代价：需要写一个采样器（复用 `engine/common.py` 的帧读取，或 ffmpeg），payload 受 48 MiB 请求体限制。
2. **混合保留**：`ground_candidates` 继续用 Qwen（原生视频），其余三处（verify/arbiter/disambiguate）用 DeepSeek。
   这是**最稳妥的落地**——先把 A/B 做了，ground 保持原样，避免一次性大改引入回归。

> 建议：**先做「混合」**——A+B 用 DeepSeek，ground 保留 Qwen。跑通后再单独立项评估「DeepSeek 帧采样 ground」能否打过原生视频。
> 原因：ground 是「召回/降本」杠杆，吃视频时序；与其赌采样，不如保持已验证的原生视频路径。

### D【可选】仲裁/验证用 thinking 模式，快速路用 non-thinking

`deepseek-v4-flash-vision-exp` 默认 thinking。对**高价值、低频**调用（identity 终审 `vlm_arbiter`）可开 thinking 提升推理质量；
对**高吞吐、逐帧**调用（`verify_target` 每个裁剪都跑）应关 thinking 控制延迟/成本。在 `provider.chat` 增加 `thinking` 开关，
`verify_target` 传 `thinking=False`，`match_image` 可传 `True`（做成阈值可调）。

---

## 4. 性能预期（哪些会提升，哪些不会）

| 环节 | 预期 | 原因 |
|---|---|---|
| `verify_target` 读数/匹配 | ⬆（最明显） | 前沿视觉模型：车牌/编号/衣着颜色误读率下降（京 vs 苏E 这类混淆应改善） |
| `vlm_arbiter` 身份终审 | ⬆ 精度（降假阳） | 从「全 yes」到 honest 推理 —— 这是上一轮最痛的点 |
| `refine_query` 出词 | ⬆ 微幅 | 更强文本模型，类别短语更贴合 YOLO-World |
| **总体** | **精度提升 > 召回提升** | 它改善的是「判断对不对 + 读得准不准」，不是「找没找到」 |
| 成本 | ⬇⬇（相对 qwen-vl-max） | vision-exp 输入 $0.22/M vs qwen-vl-max，约 1/10 量级 |
| 延迟 | ⚠ 需实测 | thinking 模式会放大 latency；建议仲裁开、逐帧关 |
| `ground_candidates`（若换采样） | 待 A/B | 采样 vs 原生视频时序，不确定 |

**结论**：这是一次 **「精度/读数提升 + 大幅降本」** 的替换，不是「速度」替换。是否真的提升，必须用现有 bench
（`server/bench`、`server/optimize_text.py`、`dataset/`）做 **A/B** 量化，而不能靠直觉。

---

## 5. 落地步骤

### 5.1 前置
- 需要一个 `DEEPSEEK_API_KEY`（当前 `.env` 只有 `DASHSCOPE_API_KEY`/`ZHIPUAI_API_KEY`，**没有 DeepSeek key**——硬前置）。
- 环境变量接入：后端 env 里加 `DEEPSEEK_API_KEY`；`server/server.py` 的 `sets` 配置面板（`ZHIPUAI_API_KEY`/`DASHSCOPE_API_KEY` 的
  邻居）补一个 `deepseek_api_key` 输入并写入 `updates`；mask 列表同步。

### 5.2 代码（先 A+B，混合 ground）
1. `server/provider.py`：加 deepseek 分支（见 §3.A），`VLM_PROVIDER=deepseek`。
2. `server/provider.py::refine_query`：文本模型切到 deepseek（§3.B）。
3. `server/provider.py::chat`：加 `thinking` 开关（§3.D），`verify_target=False`。
4. `server/server.py`：`sets` 按键面板加 `deepseek_api_key` 读写 + mask。
5. `README.md`：新增「VLM 提供商」说明（deepseek 可选 + 价格），保留 qwen/glm 作为 fallback。
6. `.env`（本地开发）：加 `DEEPSEEK_API_KEY`、`VLM_PROVIDER=deepseek`。

### 5.3 A/B 评估（决定是否「有提升」）
用现有评测路径，同一批 query/video，分别跑：
- 基线：`VLM_PROVIDER=qwen`（现在实际生效的）
- 待测：`VLM_PROVIDER=deepseek`
比较：命中区间 IoU、`match` 分布（exact/partial/假阳）、读数（OCR 对照）、单次成本/延迟。
`server/bench/` 与 `server/optimize_text.py` 是现成载体，可直接扩展为 provider 对比 runner。

---

## 6. 风险与注意

- **ground 不支持原生视频**：混合方案里 ground 仍走 Qwen，需保留 `DASHSCOPE_API_KEY`；若决定全切 DeepSeek 则要写帧采样器。
- **thinking 模式成本/延迟**：务必给 `verify_target` 关 thinking，否则逐帧放大。
- **图像 token 按尺寸计费**：`verify_target`/`match_image` 的裁剪图要限制尺寸（当前 JPEG 质量 80/90，尺寸受 `--imgsz` 管控），
  避免大图翻倍输入 token。
- **网络/证书**：当前机器走企业 MITM，需 `SSL_CERT_FILE=<venv>/site-packages/certifi/cacert.pem` 才能出网到 `api.deepseek.com`。
- **不要动 git 上传 key**：`DEEPSEEK_API_KEY` 只进 `.env`/运行时，不进仓库（现有 `.gitignore` 已排除 `.env`）。

---

## 7. 一句话结论

> **值得做。** 用 `deepseek-v4-flash-vision-exp` 替换**图片级**的 `verify_target` / `match_image`(vlm_arbiter) / `disambiguate`，
> 并把 `refine_query` 文本模型切到 `deepseek-v4-flash`，是「精度/读数提升 + 约 1/10 成本」的确定性收益；
> 云 grounding 因 DeepSeek 不支持原生视频，**先保留 Qwen**（混合），后续再单独 A/B「帧采样 vs 原生视频」。
> 是否真的提升，交给现有 bench 做 A/B 量化，不靠直觉。

---

## 8. 决策记录：是否把 agent LLM 也统一到 vision-exp（已评审）

**背景**：agent loop 的 LLM 已是 DeepSeek（`.env`：`LLM_BASE_URL=https://api.deepseek.com/v1`、`LLM_MODEL=deepseek-v4-pro`）。
"把 LLM 也换成 vision-exp" 并不会减少 provider（本来就是 DeepSeek），只是把 agent 大脑从 **pro → flash** 降级。

**结论（已采纳）**：

| 项 | 决定 |
|---|---|
| 管道 VLM | ✅ `deepseek-v4-flash-vision-exp`（A+B） |
| agent loop LLM | ✅ **保留 `deepseek-v4-pro`**（不强求单模型 id） |
| Key 统一 | ✅ `DEEPSEEK_API_KEY` 与 `LLM_API_KEY` **复用同一个 DeepSeek key**；`provider._ds_key()` 优先 `DEEPSEEK_API_KEY`、回退 `LLM_API_KEY`，无需在 `.env` 重复填 key |
| 云 grounding | ⚠️ 保留 Qwen（`DASHSCOPE_API_KEY`），因 DeepSeek 不支持原生 `video_url` |
| Base URL | ✅ 同一 host；agent 走 litellm `/v1`，管道 `_post` 拼 `/chat/completions`，属正常差异 |

**刻意不做（列为可选后续）**：
- 让 agent 也用 vision-exp 并**直接看证据图**（把 tool 返回的 `crops/*.jpg` 塞进消息）—— 这是真正意义上的"一个大脑看懂+想清楚"，但需大改 agent loop，且接受 pro→flash 规划深度下降。**单独立项评估。**
- 云 grounding 帧采样切 DeepSeek —— 有回归风险，**单独 A/B 后决定**。

**为什么要分开**：agent 的职责（多步编排/plan/工具调度）是**复杂度最高的纯文本任务**，flash 档视觉模型深度不足、其视觉能力对纯文本编排用不上；"统一到同一个 key/base_url" 已经拿到大部分简洁收益，不必为省一个模型 id 而牺牲编排深度。
