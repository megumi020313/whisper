# Brain-Core: 通用大模型推理服务

**项目代号**：`Brain-Core`  
**核心定位**：系统的"大脑" —— 通用大模型推理基础设施

## 项目概述

本项目是一个**无状态**的 API 服务，不包含任何业务逻辑，只负责将 Qwen2.5 大模型加载到显存中，并以极高的吞吐量响应外部的推理请求。兼容 OpenAI API 格式，方便任何客户端调用。

## 目录结构

```
brain_core/
├── backend/                      # 后端代码
│   ├── config/                  # 配置层
│   │   └── settings.py          # 模型路径、显存配额、并发参数
│   ├── utils/                   # 工具层
│   │   ├── gpu_utils.py         # 显存检测工具
│   │   └── logger.py            # 日志配置
│   ├── common/                  # 通用层
│   │   └── chat_templates.py    # 对话模板适配
│   ├── schema/                  # 验证层
│   │   └── openai_protocol.py   # OpenAI 协议定义
│   ├── modules/                 # 核心业务模块
│   │   └── llm_engine/          # LLM 引擎模块
│   │       └── vllm_runner.py   # vLLM AsyncLLMEngine 单例封装
│   ├── service/                 # 服务层
│   │   └── inference_service.py # 推理服务
│   ├── controller/              # 控制层
│   │   └── chat_routes.py       # HTTP 路由
│   └── main.py                  # FastAPI 启动入口
├── models/                       # 模型存储目录（方案二：项目内相对路径）
│   └── llm/
│       └── Qwen2.5-7B-Instruct-GPTQ-Int4/  # LLM 模型文件
├── scripts/                      # 脚本目录
│   ├── download_model.sh         # 模型下载脚本
│   └── start_server.sh           # 启动脚本
├── requirements.txt              # Python 依赖
├── Dockerfile                    # Docker 镜像构建文件
└── README.md                     # 项目说明文档
```

## 模型存储

**方案**：项目内相对路径（方案二）

模型存储在 `models/llm/Qwen2.5-7B-Instruct-GPTQ-Int4/` 目录下，使用项目根目录的相对路径。

**模型下载**：
```bash
cd brain_core
bash scripts/download_model.sh
```

## 快速开始

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **下载模型**：
```bash
bash scripts/download_model.sh
```

3. **启动服务**：
```bash
bash scripts/start_server.sh
```

## 配置说明

关键配置在 `backend/config/settings.py`：
- `MODEL_PATH`: 模型路径（项目内相对路径）
- `GPU_MEMORY_UTILIZATION`: 显存占用限制（默认 0.5，即 50%）
- `PORT`: 服务端口（默认 8000）

## API 接口

- `POST /v1/chat/completions` - OpenAI 兼容的对话接口

## 显存管理

- 显存占用限制：50%（约 16GB on V100 32GB）
- 与 Project 1 (Audio-Sentry) 共存，Project 1 占用 40%（约 12GB）
- 系统预留：10%（约 4GB）

