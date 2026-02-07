# LLM 模型存储目录

## 模型路径

模型应存储在：`models/llm/Qwen2.5-7B-Instruct-GPTQ-Int4/`

## 下载模型

使用以下命令下载模型：

```bash
# 安装 huggingface-cli（如果未安装）
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --local-dir models/llm/Qwen2.5-7B-Instruct-GPTQ-Int4
```

或者使用脚本：

```bash
cd brain_core
bash scripts/download_model.sh
```

## 模型信息

- **模型名称**：Qwen2.5-7B-Instruct-GPTQ-Int4
- **量化方式**：GPTQ Int4
- **模型大小**：约 4-8GB
- **显存占用**：约 16GB（V100 32GB 的 50%）

## 注意事项

- 确保有足够的磁盘空间（建议至少 10GB）
- 下载可能需要较长时间，请耐心等待
- 模型文件较大，建议使用稳定的网络连接

