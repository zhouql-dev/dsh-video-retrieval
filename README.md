# dsh-video-retrieval

DeepSeek Harness 原生视频检索插件：把监控视频时空目标定位能力打包成标准 DSH 模式，一键安装即可在 DSH 网页里使用。

## 功能

- **自然语言 / 参考图** → 时间区间 + 逐帧 bbox 轨迹 + 标注视频
- **云边协同漏斗**：云端粗筛 + 本地核验 + 多信号投票（颜色 / 时间曲线 / VLM 终审）
- **递归自进化**：案例确认 → Optuna 阈值优化 + 提示词/混淆表反射 → holdout 门禁 → 热加载
- **嵌入式三列控制台**：保留左侧会话列表，随时切回 DeepSeek 对话
- **16 个原生工具**：`vr_search` / `vr_preflight` / `vr_job_cancel` / `vr_evolve` …

## 安装

### 方式一（推荐）：直接作为插件安装

```bash
# 1. 安装插件包（dsh.bundle 会自动合入 profile 的 bundles，`prepare` 脚本会自动构建 dist/）
dsh plugin --profile web add github:zhouql-dev/dsh-video-retrieval

# 2. 部署预设 + 下载权重（install.sh 会从 profile 里定位已安装的包）
git clone https://github.com/zhouql-dev/dsh-video-retrieval.git /tmp/dsh-video-retrieval
bash /tmp/dsh-video-retrieval/scripts/install.sh

# 3. 重启 DSH（bundle 补丁是启动时事实）
npx @deepseek-ai/dsh web
```

### 方式二：从源码安装（开发）

```bash
git clone https://github.com/zhouql-dev/dsh-video-retrieval.git
cd dsh-video-retrieval
bash scripts/install.sh          # 构建 + 安装包（file:）+ 部署预设 + 下载权重
npx @deepseek-ai/dsh web
```

然后在新建会话界面选择 **视频检索模式**。

### 前置条件

- Node ≥ 22、pnpm
- Python ≥ 3.10 + venv（含 torch / ultralytics / opencv-python / rapidocr-onnxruntime / insightface / transformers / torchreid / scipy / optuna / fastapi / uvicorn / litellm）
- ffmpeg、tesseract（车牌 OCR）
- 可选 API key：`ZHIPUAI_API_KEY` / `DASHSCOPE_API_KEY`（无 key 时自动降级到本地全扫）

> 权重（`yolov8s-worldv2.pt` / OSNet / CLIP ViT-B-32）不在仓库里，由 `scripts/setup.sh` 从 GitHub Releases 下载。若你自建了权重源，用 `WEIGHTS_URL_PREFIX` 环境变量覆盖。

## 目录结构

```
dsh-video-retrieval/
├── package.json              # dsh.bundle + dsh.client + prepare 构建脚本
├── cordis.patch.yml          # bundle 补丁（全局客户端控制台行，便携路径）
├── preset/video-retrieval/   # 模式组合 + 自带 skill
│   ├── agent.cordis.yml      # 模板；install.sh 把 __...__ 占位符替换为绝对路径
│   ├── preset.yml
│   └── skills/video-retrieval/SKILL.md
├── src/index.js              # 宿主半（16 个 vr_* 工具）
├── dist-client/index.js      # 客户端 bundle（__ModuleLoader__ 格式）
├── engine/                   # 引擎脚本（vendored）
├── server/                   # fusion 镜像（vendored）+ web GUI
├── scripts/
│   ├── install.sh            # 部署预设 + 下载权重 + 冒烟（也可兜底安装 file: 包）
│   ├── setup.sh              # 下载权重
│   ├── sync.sh               # 【开发者专用】从私有 fusion/skill 源同步 engine/server
│   ├── serve.sh              # 控制台标签页启动
│   ├── smoke.mjs             # 纯 Node 冒烟（CI 用）
│   └── cdp.mjs               # Chrome DevTools 调试辅助
├── build.mjs                 # 宿主半构建（dist/index.js）
├── .github/workflows/        # ci.yml + release.yml
├── weights/                  # .gitignore；由 setup.sh 下载
├── config/                   # 运行时演化配置
└── data/                     # 运行时案例/任务/上传
```

## 更新

```bash
# 普通用户：重新安装最新版即可
dsh plugin --profile web add github:zhouql-dev/dsh-video-retrieval
# 若预设/脚本有更新，重跑 install.sh
bash scripts/install.sh
# 浏览器刷新页面即可（客户端 bundle 内容变更无需重启 dsh）
```

## 开发者：同步 vendored 源码

`engine/` 和 `server/` 是本插件 vendored 的引擎脚本与 fusion 后端镜像，**它们的权威源不在本仓库**（在开发者的私有 `fusion/` 检出和 skill 源里）。普通用户无需关心；插件开发者用 `sync.sh` 单向同步：

```bash
VTL_REPO_DIR=/path/to/your/video-retrieval   # 含 fusion/ 的私有仓库根
VTL_SKILL_DIR=/path/to/skills/video-target-localize   # 引擎脚本 skill 源
bash scripts/sync.sh
```

未设置这两个环境变量时，`sync.sh` 会安全跳过对应部分。

## 卸载

```bash
rm -rf ~/.dsh/.agent-presets/video-retrieval
dsh plugin --profile web remove dsh-video-retrieval
# 重启 dsh web
```

## 许可

MIT
