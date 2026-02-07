# Scripts 目录说明

本目录包含声纹识别系统的工具脚本和测试脚本。

## 📁 目录结构

```
scripts/
├── test/                    # 测试和验证脚本
│   ├── test_*.py           # 功能测试脚本
│   ├── debug_*.py          # 调试脚本
│   └── diagnose_*.py       # 诊断脚本
├── build_cohort.py         # Cohort矩阵构建工具
├── calibrate_threshold.py  # 阈值校准工具
├── gen_directional_calibration.py  # 定向麦克风校准数据生成
├── fix_checkpoint.py       # 模型权重清洗工具
└── start_api.sh           # API服务启动脚本
```

## 🔧 工具脚本（scripts/根目录）

### 1. build_cohort.py - Cohort矩阵构建工具

**功能**：构建AS-Norm所需的Cohort矩阵

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/build_cohort.py
```

**前置条件**：
- Cohort音频文件已准备（data/calibration/cohort_wavs/）
- ERes2NetV2模型已加载

**输出**：data/vector_db/cohort.npy

---

### 2. calibrate_threshold.py - 阈值校准工具

**功能**：校准声纹识别系统的AS-Norm阈值

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/calibrate_threshold.py
```

**前置条件**：
- Cohort矩阵已构建
- Target和Imposter音频已准备

**输出**：推荐的high_threshold和low_threshold配置

---

### 3. gen_directional_calibration.py - 定向麦克风校准数据生成

**功能**：为定向麦克风场景生成加噪增强的校准数据

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/gen_directional_calibration.py
```

**前置条件**：
- 背景噪音文件已准备（data/noise/restaurant_bg.wav.wav）
- Target用户音频已准备

**输出**：增强音频文件（*_aug_*.wav）

---

### 4. fix_checkpoint.py - 模型权重清洗工具

**功能**：从Fusion版本checkpoint中提取标准版权重

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/fix_checkpoint.py
```

**输入**：models/eres2netv2/pretrained_eres2netv2.ckpt

**输出**：models/eres2netv2/eres2netv2_clean.pth

---

### 5. start_api.sh - API服务启动脚本

**功能**：启动声纹识别API服务

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
bash scripts/start_api.sh
```

---

## 🧪 测试脚本（scripts/test/目录）

### 功能测试脚本

#### 1. test_v3_diarization.py - V3.0对话智能引擎测试

**功能**：测试并行感知 + 对话智能引擎流水线

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_v3_diarization.py
```

**测试内容**：
- 声纹识别流水线
- 对话智能引擎（DiarizationEngine）
- 并行感知算法
- 会话片段合并和平滑

---

#### 2. test_similarity_fix.py - 相似度计算修复测试

**功能**：验证智能融合中的相似度计算

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_similarity_fix.py
```

**测试内容**：
- 声纹向量提取
- 余弦相似度计算
- AS-Norm归一化
- 智能融合算法

---

#### 3. test_vad_isolation.py - VAD隔离测试

**功能**：诊断Silero VAD是否正确切分音频片段

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_vad_isolation.py
```

**测试内容**：
- VAD模型加载
- 语音活动检测
- 音频片段切分
- VAD参数配置

---

#### 4. test_v20_optimization.py - V2.0声纹识别质量优化测试

**功能**：测试智能会话融合算法效果

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_v20_optimization.py
```

**测试内容**：
- VAD优化效果
- 会话融合算法
- Z-score提升
- 识别准确率

---

#### 5. test_iotdb.py - IoTDB集成测试

**功能**：测试IoTDB连接器基本功能

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_iotdb.py
```

**测试内容**：
- IoTDB连接建立
- 音频事件写入
- 识别结果写入
- 数据查询功能

---

#### 6. test_cuda_setup.py - CUDA/cuDNN配置测试

**功能**：验证CUDA/cuDNN库配置

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_cuda_setup.py
```

**测试内容**：
- CUDA可用性和版本
- cuDNN可用性和版本
- GPU设备信息
- Whisper模型GPU运行能力

---

#### 7. test_all_features.py - 系统全功能集成测试

**功能**：测试声纹识别系统所有核心功能

**启动方式**：
```bash
# 1. 先启动API服务器
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python -m src.api.app

# 2. 在另一个终端运行测试
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_all_features.py
```

**测试内容**：
- API健康检查
- 用户注册功能
- 用户列表查询
- 声纹识别功能
- ASR语音识别功能

---

#### 8. test_asr.py - ASR功能测试

**功能**：测试Whisper ASR服务和推理流水线集成

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_asr.py
```

**测试内容**：
- Whisper服务初始化
- 音频转录功能
- 中文语音识别准确性
- ASR与声纹识别流水线集成

---

### 调试脚本

#### 9. debug_model_collapse.py - 模型坍塌调试

**功能**：检测ERes2NetV2模型是否存在"模型坍塌"问题

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_model_collapse.py
```

**诊断内容**：
- 模型是否对不同输入产生不同输出
- 声纹向量的有效性
- BN层统计量是否正确

---

#### 10. debug_cohort_leak.py - Cohort泄漏调试

**功能**：检测AS-Norm的Cohort矩阵是否存在数据泄漏

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_cohort_leak.py
```

**诊断内容**：
- Cohort矩阵是否包含注册用户声纹
- Target与Cohort的相似度分布
- 高相似度嫌疑向量来源

---

#### 11. debug_similarity.py - 相似度计算调试

**功能**：调试同一用户不同音频样本的相似度计算

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_similarity.py
```

**诊断内容**：
- 模型是否正常工作
- 原始余弦相似度是否合理
- 音频质量（RMS能量、时长）

---

### 诊断脚本

#### 12. diagnose_model_loading.py - 模型加载诊断

**功能**：诊断ERes2NetV2模型加载问题

**启动方式**：
```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/diagnose_model_loading.py
```

**诊断内容**：
- Checkpoint文件结构
- State dict键名和形状
- 权重参数数量
- 模型架构匹配性

---

## 📝 使用流程建议

### 初次部署流程

1. **构建Cohort矩阵**
   ```bash
   python scripts/build_cohort.py
   ```

2. **校准阈值**
   ```bash
   python scripts/calibrate_threshold.py
   ```

3. **测试CUDA配置**（如使用GPU）
   ```bash
   python scripts/test/test_cuda_setup.py
   ```

4. **运行全功能测试**
   ```bash
   # 终端1：启动API
   python -m src.api.app
   
   # 终端2：运行测试
   python scripts/test/test_all_features.py
   ```

### 问题诊断流程

1. **模型问题诊断**
   ```bash
   python scripts/test/diagnose_model_loading.py
   python scripts/test/debug_model_collapse.py
   python scripts/test/debug_similarity.py
   ```

2. **VAD问题诊断**
   ```bash
   python scripts/test/test_vad_isolation.py
   ```

3. **Cohort问题诊断**
   ```bash
   python scripts/test/debug_cohort_leak.py
   ```

### 功能验证流程

1. **声纹识别验证**
   ```bash
   python scripts/test/test_v3_diarization.py
   python scripts/test/test_similarity_fix.py
   ```

2. **ASR功能验证**
   ```bash
   python scripts/test/test_asr.py
   ```

3. **系统优化验证**
   ```bash
   python scripts/test/test_v20_optimization.py
   ```

---

## ⚠️ 注意事项

1. **环境要求**
   - 所有脚本都需要在`vioce` conda环境中运行
   - 确保CUDA环境已正确配置（GPU模式）

2. **路径要求**
   - 所有脚本都假设从项目根目录运行
   - 相对路径基于项目根目录

3. **数据要求**
   - 测试脚本需要准备相应的测试音频文件
   - 工具脚本需要准备相应的输入数据

4. **权限要求**
   - 确保有读写data/目录的权限
   - 确保有读写models/目录的权限

---

## 📚 相关文档

- [快速使用指南](../快速使用)
- [配置文件说明](../config/model_config.yaml)
- [API文档](https://localhost:8000/docs)

---

**最后更新**：2025-12-25

