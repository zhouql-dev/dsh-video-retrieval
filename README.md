# dsh-video-retrieval

DeepSeek Harness 原生视频检索插件：把监控视频时空目标定位能力打包成标准 DSH 模式，一键安装即可在 DSH 网页里使用。

## 功能

- **自然语言 / 参考图** → 时间区间 + 逐帧 bbox 轨迹 + 标注视频
- **云边协同漏斗**：云端粗筛 + 本地核验 + 多信号投票（颜色 / 时间曲线 / VLM 终审）
- **递归自进化**：案例确认 → Optuna 阈值优化 + 提示词/混淆表反射 → holdout 门禁 → 热加载
- **嵌入式三列控制台**：保留左侧会话列表，随时切回 DeepSeek 对话
- **16 个原生工具**：`vr_search` / `vr_preflight` / `vr_job_cancel` / `vr_evolve` …

## 安装

```bash
git clone https://github.com/zhouql-dev/dsh-video-retrieval.git
cd dsh-video-retrieval
bash scripts/install.sh          # 构建 + 部署预设 + 安装包 + 下载权重
npx @deepseek-ai/dsh web         # 重启 DSH（bundle 补丁是启动时事实）
```

然后在新建会话界面选择 **视频检索模式**。

### 前置条件

- Node ≥ 22、pnpm
- Python ≥ 3.10 + venv（含 torch / ultralytics / opencv-python / rapidocr-onnxruntime / insightface / transformers / torchreid / scipy / optuna / fastapi / uvicorn / litellm）
- ffmpeg、tesseract（车牌 OCR）
- 可选 API key：`ZHIPUAI_API_KEY` / `DASHSCOPE_API_KEY`（无 key 时自动降级到本地全扫）

## 目录结构

```
dsh-video-retrieval/
├── package.json              # dsh.bundle + dsh.client + peerDependencies
├── cordis.patch.yml          # bundle 补丁（全局客户端控制台行）
├── preset/video-retrieval/   # 模式组合 + 自带 skill
│   ├── agent.cordis.yml      # install.sh 会把 __...__ 占位符替换为绝对路径
│   ├── preset.yml
│   └── skills/video-retrieval/SKILL.md
├── src/index.js              # 宿主半（16 个 vr_* 工具）
├── dist-client/index.js      # 客户端 bundle（__ModuleLoader__ 格式）
├── engine/                   # 引擎脚本（vendored）
├── server/                   # fusion 镜像（vendored）+ web GUI
├── scripts/
│   ├── install.sh            # 一键安装
│   ├── setup.sh              # 下载权重
│   ├── sync.sh               # 从权威源单向同步 engine/server
│   ├── serve.sh              # 控制台标签页启动
│   └── cdp.mjs               # Chrome DevTools 调试辅助
├── build.mjs                 # 宿主半编译（esbuild → dist/）
├── weights/                  # .gitignore；由 setup.sh 下载
├── config/                   # 运行时演化配置
└── data/                     # 运行时案例/任务/上传
```

## 更新

```bash
cd dsh-video-retrieval
bash scripts/sync.sh           # 从你的 fusion/ 和 skill 源拉最新代码
bash scripts/install.sh        # 重新构建 + 部署
# 浏览器刷新页面即可（bundle 内容变更无需重启 dsh）
```

## 卸载

```bash
rm -rf ~/.dsh/.agent-presets/video-retrieval
dsh plugin --profile web remove dsh-video-retrieval
# 重启 dsh web
```

## 许可

MIT