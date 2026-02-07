# 测试脚本目录说明

本目录包含所有测试、验证和调试脚本。

## 📊 文件分类

### 🧪 功能测试脚本（8个）

| 文件名 | 功能说明 | 测试内容 |
|--------|---------|---------|
| `test_v3_diarization.py` | V3.0对话智能引擎测试 | 声纹识别流水线、对话智能引擎、并行感知算法 |
| `test_similarity_fix.py` | 相似度计算修复测试 | 声纹向量提取、余弦相似度、AS-Norm归一化 |
| `test_vad_isolation.py` | VAD隔离测试 | VAD模型加载、语音活动检测、音频片段切分 |
| `test_v20_optimization.py` | V2.0质量优化测试 | VAD优化、会话融合、Z-score提升 |
| `test_iotdb.py` | IoTDB集成测试 | IoTDB连接、数据写入、数据查询 |
| `test_cuda_setup.py` | CUDA/cuDNN配置测试 | CUDA可用性、cuDNN配置、GPU设备信息 |
| `test_all_features.py` | 系统全功能集成测试 | API健康检查、用户注册、声纹识别、ASR |
| `test_asr.py` | ASR功能测试 | Whisper服务、音频转录、中文识别 |

### 🔍 调试脚本（3个）

| 文件名 | 功能说明 | 诊断内容 |
|--------|---------|---------|
| `debug_model_collapse.py` | 模型坍塌调试 | 模型输出有效性、BN层统计量、激活函数 |
| `debug_cohort_leak.py` | Cohort泄漏调试 | Cohort数据泄漏、相似度分布、嫌疑向量 |
| `debug_similarity.py` | 相似度计算调试 | 模型工作状态、余弦相似度、音频质量 |

### 🩺 诊断脚本（1个）

| 文件名 | 功能说明 | 诊断内容 |
|--------|---------|---------|
| `diagnose_model_loading.py` | 模型加载诊断 | Checkpoint结构、State dict、权重参数 |

## 🚀 快速启动

所有脚本的启动方式：

```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/<脚本名称>.py
```

**注意**：`test_all_features.py` 需要先启动API服务器。

## 📋 测试场景推荐

### 场景1：初次部署验证

```bash
# 1. 测试CUDA配置
python scripts/test/test_cuda_setup.py

# 2. 测试VAD功能
python scripts/test/test_vad_isolation.py

# 3. 测试ASR功能
python scripts/test/test_asr.py

# 4. 测试全功能（需先启动API）
python scripts/test/test_all_features.py
```

### 场景2：识别问题诊断

```bash
# 1. 检查模型是否坍塌
python scripts/test/debug_model_collapse.py

# 2. 检查相似度计算
python scripts/test/debug_similarity.py

# 3. 检查Cohort数据泄漏
python scripts/test/debug_cohort_leak.py

# 4. 检查模型加载
python scripts/test/diagnose_model_loading.py
```

### 场景3：功能优化验证

```bash
# 1. 测试V2.0优化效果
python scripts/test/test_v20_optimization.py

# 2. 测试V3.0对话引擎
python scripts/test/test_v3_diarization.py

# 3. 测试相似度修复
python scripts/test/test_similarity_fix.py
```

### 场景4：集成功能测试

```bash
# 1. 测试IoTDB集成
python scripts/test/test_iotdb.py

# 2. 测试ASR集成
python scripts/test/test_asr.py

# 3. 测试全功能集成
python scripts/test/test_all_features.py
```

## 📝 脚本详细说明

每个脚本文件开头都包含详细的注释说明：
- **【功能说明】**：脚本的功能和测试内容
- **【启动方式】**：完整的启动命令
- **【前置条件】**：运行前需要满足的条件
- **【预期输出】**：脚本运行后的输出内容

请查看各脚本文件头部注释获取详细信息。

## ⚠️ 注意事项

1. **环境要求**
   - 必须在 `vioce` conda环境中运行
   - 确保CUDA环境已正确配置（GPU模式）

2. **路径要求**
   - 所有脚本都假设从项目根目录运行
   - 测试音频文件路径需要根据实际情况调整

3. **依赖要求**
   - 某些测试需要先构建Cohort矩阵
   - 某些测试需要先启动API服务器
   - 某些测试需要准备测试音频文件

4. **权限要求**
   - 确保有读写 `data/` 目录的权限
   - 确保有读写 `models/` 目录的权限

## 🔗 相关文档

- [Scripts目录总览](../README.md)
- [快速使用指南](../../快速使用)
- [配置文件说明](../../config/model_config.yaml)

---

**最后更新**：2025-12-25 15:30:50

