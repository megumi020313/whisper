# Remote-V2 声纹识别系统

基于 ERes2NetV2 和 Silero VAD 的声纹识别系统（V14.2）

## 版本信息

- **当前版本**: V14.2
- **最后更新**: 2025-12-22

## 核心特性

- 🎯 **V14.0 批量推理**: 10-50x性能提升，2分钟音频从60秒降至3-5秒
- 🔊 **V13.0 VAD引导**: VAD为王策略，解决Silence异常问题
- 🎤 **V12.0 全库识别**: 多人对话区分，网格投票法
- 🛡️ **AS-Norm评分**: 自适应归一化，提高跨场景鲁棒性
- 💾 **V14.2 显存保护**: Mini-Batch切分防止OOM

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 11.8+)
- 16GB+ RAM (推荐 32GB)
- NVIDIA GPU (推荐 RTX 3090/4090)

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd remote-v2

# 安装依赖
pip install -r requirements.txt

# 下载模型文件
# ERes2NetV2: 放置到 models/eres2netv2/pretrained_eres2netv2.ckpt
# Silero VAD: 放置到 models/vad/
```

### 启动服务

#### 方式1: 仅启动API服务

```bash
conda activate dooropen_py311
python -m src.api.app
```

API文档: http://localhost:8000/docs

#### 方式2: 启动API + Web测试界面 (推荐)

终端1 - 启动API:
```bash
conda activate dooropen_py311
python -m src.api.app
```

终端2 - 启动Web:
```bash
conda activate dooropen_py311
cd web
python app.py
```

访问: http://localhost:5000

### 健康检查（Sanity Check）

确保主流程的前端与模型加载保持一致，可在每次环境/代码更新后跑一次健康检查：

```bash
conda activate dooropen_py311
python data/test/verify_real_speech.py
```

- 脚本路径: [data/test/verify_real_speech.py](data/test/verify_real_speech.py)
- 需要自备两段 16k 单声道语音（不同说话人），更新脚本里的 `wav_a_path`、`wav_b_path`
- 内部使用 Kaldi fbank + 全局 CMVN，输出向量已归一化；保持与线上推理前端完全一致
- 期望：不同人相似度 ~0.1-0.3，同一人相似度 ≥0.7
- 实际部署建议先做 VAD 裁剪长静音，推荐有效语音 2~6 秒

## 目录结构

```
remote-v2/
├── config/                    # 配置文件
│   ├── model_config.yaml     # 模型配置(V14.0参数)
│   └── server_config.yaml    # 服务器配置
├── src/                      # 源代码
│   ├── api/                  # API接口层
│   ├── core/                 # 核心业务层
│   ├── models/               # 模型层
│   ├── pipeline/             # 推理流水线
│   └── utils/                # 工具层
├── web/                      # Web测试界面
├── scripts/                  # 工具脚本
└── docs/                     # 文档
```

## 版本历史

### V14.2 (2025-12-22)
- 显存保护修复：Mini-Batch切分防止OOM

### V14.1 (2025-12-22)
- 时间戳异常修复
- 前端字段兼容性修复

### V14.0 (2025-12-22)
- 批量推理：10-50x性能提升
- 严格拒识：提高阈值防止误识
- VAD参数平衡：300ms静音时长

### V13.0 (2025-12-22)
- VAD引导：VAD为王策略
- 窗口投票：解决Silence异常

### V12.0 (2025-12-21)
- 多人全库识别
- 网格投票法

## 文档

- [项目说明书](../项目说明书.md)
- [开发日志](docs/logs/项目开发工作日志.txt)
- [V14.0方案](docs/V14.0.md)
- [V13.0方案](docs/V13.0.md)

## 许可证

Copyright © 2025 项目开发团队
