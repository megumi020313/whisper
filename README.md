# Faster-Whisper ASR 模块

基于 Faster-Whisper 的高性能语音识别模块，支持中文和英文音频转文字。

## 功能特性

- 🎯 **高精度识别**：基于 Whisper-Large-v3，字准确率 95%+
- ⚡ **高性能推理**：4-5 倍速度提升（CTranslate2 优化）
- 🇨🇳 **中文优化**：直接输出简体中文，无需后处理
- ⏰ **词级别时间戳**：精确到词的时间定位（±0.1秒）
- 🔧 **灵活配置**：支持多种参数调节（beam_size、compute_type 等）
- 🎵 **多格式支持**：支持 WAV、MP3、FLAC 等音频格式

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境（如果使用conda环境）
conda activate your_env_name  # 替换为您的环境名称

# 安装依赖
pip install -r requirements.txt

# 或安装核心依赖（推荐）
pip install faster-whisper numpy librosa soundfile pyyaml loguru

# 下载模型（约 3GB）
huggingface-cli download Systran/faster-whisper-large-v3 \
  --local-dir models/faster-whisper-large-v3

# 测试CPU模式（可选）
python test_cpu_mode.py
```

### 2. 使用启动脚本

#### 使用默认配置路径
```bash
# 使用配置文件中的默认输入路径（data文件夹）
./run_asr.sh

# 或使用Python脚本
python scripts/run_asr.py
```

#### 处理单个音频文件
```bash
# 使用Shell脚本
./run_asr.sh audio.wav

# 或使用Python脚本
python scripts/run_asr.py --input audio.wav
```

#### 处理整个文件夹
```bash
# 处理文件夹中的所有音频文件
./run_asr.sh audio_folder/

# 指定输出目录
./run_asr.sh -o results audio_folder/
```

#### 高级选项
```bash
# 输出带时间戳的结果
./run_asr.sh --timestamps audio.wav

# 输出词级别时间戳
./run_asr.sh --words audio.wav

# 指定语言和参数
./run_asr.sh --language zh --beam-size 5 --verbose audio.wav
```

### 3. Python API 使用

```python
from backend.modules.audio_analysis import WhisperService

# 初始化服务
asr = WhisperService()

# 转录音频文件
text = asr.transcribe(audio="audio.wav", language="zh")
print(f"识别结果: {text}")
```

### 3. 高级功能

```python
# 带时间戳的转录
segments = asr.transcribe_with_timestamps(audio="audio.wav")
for seg in segments:
    print(f"[{seg['start']:.2f}s] {seg['text']}")

# 词级别时间戳
words = asr.transcribe_with_word_timestamps(audio="audio.wav")
for word in words:
    print(f"[{word['start']:.2f}s] {word['word']}")
```

## 配置说明

### 配置文件

编辑 `config/model_config.yaml`：

```yaml
# 模型路径配置
model:
  whisper_path: models/faster-whisper-large-v3  # ASR模型路径

# GPU设备分配
device:
  asr_model: cpu  # ASR模型使用的设备（cpu/cuda:0）
  default: cpu  # 默认设备

# ASR 配置
asr:
  enabled: true  # 是否启用ASR
  language: zh  # 语言代码：zh（中文）、en（英文）、auto（自动检测）
  initial_prompt: "以下是普通话的句子。"  # 引导词
  beam_size: 5  # Beam search 大小：1（最快）、5（平衡）、10（最高精度）
  compute_type: float16  # 计算精度：float16（推荐）、int8、float32
  vad_filter: false  # 是否使用Whisper内置VAD过滤

# 音频处理配置
audio:
  input_path: data  # 输入音频路径：单个文件或文件夹路径（相对于项目根目录）
  output_dir: output  # 输出目录，用于保存转录结果
  supported_formats: [".wav", ".mp3", ".m4a", ".flac", ".ogg"]  # 支持的音频格式
  sample_rate: 16000  # 目标采样率
  recursive: true  # 是否递归处理子文件夹
```

### 参数说明

#### 设备参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `asr_model` | str | "cpu" | ASR 使用的设备：cpu（CPU模式）/cuda:0（GPU模式） |
| `default` | str | "cpu" | 默认设备 |

**注意**:
- **智能设备选择**: 系统会自动检测CUDA可用性，如果不可用自动回退到CPU模式
- **GPU模式**: 推荐使用 "cuda:0"，性能最好但需要兼容的CUDA环境
- **CPU模式**: 兼容性最好，自动使用float32计算类型
- **自动调整**: CPU模式下会自动将float16调整为float32以确保兼容性

#### ASR 参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `language` | str | "zh" | 语言代码：zh/en/auto |
| `beam_size` | int | 5 | Beam search 大小，越高精度越好但速度越慢 |
| `compute_type` | str | "float16" | 计算精度：float16（推荐）/int8/float32 |
| `initial_prompt` | str | "以下是普通话的句子。" | 引导模型输出风格的提示词 |
| `vad_filter` | bool | false | 是否使用 Whisper 内置 VAD 过滤静音 |

#### 音频处理参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_path` | str | "data" | 输入音频路径（相对于项目根目录） |
| `output_dir` | str | "output" | 输出目录，用于保存转录结果 |
| `supported_formats` | list | [".wav", ".mp3", ".m4a", ".flac", ".ogg"] | 支持的音频格式 |
| `sample_rate` | int | 16000 | 目标采样率 |
| `recursive` | bool | true | 是否递归处理子文件夹 |

## 启动脚本使用

### Shell脚本 (推荐)

`run_asr.sh` 是最简单的使用方式：

```bash
# 查看帮助
./run_asr.sh --help

# 处理单个文件
./run_asr.sh audio.wav

# 处理文件夹
./run_asr.sh audio_folder/

# 自定义输出目录
./run_asr.sh -o results audio.wav

# 输出时间戳
./run_asr.sh --timestamps audio.wav

# 详细输出
./run_asr.sh --verbose audio.wav
```

### Python脚本

`scripts/run_asr.py` 提供更多控制选项：

```bash
# 使用默认配置路径
python scripts/run_asr.py

# 基本使用
python scripts/run_asr.py --input audio.wav

# 指定配置文件
python scripts/run_asr.py --config config/model_config.yaml --input audio.wav

# 高级选项
python scripts/run_asr.py \
  --input audio.wav \
  --output results \
  --language zh \
  --beam-size 5 \
  --timestamps \
  --verbose
```

### 输出结果

每次运行会创建一个以时间戳命名的文件夹，包含所有音频的处理结果：

```
output/
└── 20241229_154500/          # 时间戳命名的文件夹
    ├── audio1.txt           # 基本转录结果
    ├── audio1_timestamps.json  # 带时间戳结果
    ├── audio2.txt           # （如果有多个文件）
    └── audio2_timestamps.json
```

**输出文件格式**：

1. **基本转录** (`audio.txt`)
   ```
   这是识别出的文本内容。
   ```

2. **带时间戳** (`audio_timestamps.json`) - 默认输出
   ```json
   [
     {
       "text": "这是第一句话。",
       "start": 0.0,
       "end": 2.5
     },
     {
       "text": "这是第二句话。",
       "start": 2.5,
       "end": 5.0
     }
   ]
   ```

3. **词级别时间戳** (`--words` → `audio_words.json`)
   ```json
   [
     {
       "word": "这",
       "start": 0.0,
       "end": 0.2,
       "confidence": 0.98
     },
     {
       "word": "是",
       "start": 0.2,
       "end": 0.4,
       "confidence": 0.95
     }
   ]
   ```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `language` | str | "zh" | 语言代码：zh/en/auto |
| `beam_size` | int | 5 | Beam search 大小，越高精度越好但速度越慢 |
| `compute_type` | str | "float16" | 计算精度：float16（推荐）/int8/float32 |
| `initial_prompt` | str | "以下是普通话的句子。" | 引导模型输出风格的提示词 |
| `vad_filter` | bool | false | 是否使用 Whisper 内置 VAD 过滤静音 |

## API 接口

### WhisperService 类

#### 初始化参数

```python
WhisperService(
    model_path=None,      # 模型路径，默认从配置文件读取
    device="cuda",        # 设备：cuda/cpu
    compute_type="float16" # 计算类型
)
```

#### 主要方法

##### `transcribe(audio, language="zh", beam_size=5, vad_filter=False, initial_prompt=None) -> str`

转录音频为文本。

**参数：**
- `audio`: 音频数据，支持 numpy 数组或文件路径
- `language`: 语言代码
- `beam_size`: Beam search 大小
- `vad_filter`: 是否使用 VAD 过滤
- `initial_prompt`: 初始提示词

**返回值：** 识别的文本字符串

##### `transcribe_with_timestamps(audio, language="zh", beam_size=5, initial_prompt=None) -> list[dict]`

转录音频并返回带时间戳的片段。

**返回值：**
```python
[
    {
        "text": "识别的文本",
        "start": 0.0,    # 开始时间（秒）
        "end": 2.5       # 结束时间（秒）
    }
]
```

##### `transcribe_with_word_timestamps(audio, language="zh", beam_size=5, initial_prompt=None) -> list[dict]`

转录音频并返回词级别时间戳。

**返回值：**
```python
[
    {
        "word": "词语",
        "start": 0.0,      # 开始时间（秒）
        "end": 0.5,        # 结束时间（秒）
        "confidence": 0.98 # 置信度
    }
]
```

## 测试与验证

### 运行测试

```bash
# 运行 ASR 测试
python scripts/test/test_asr.py
```

### 性能指标

| 配置 | 处理速度 | 准确率 | 显存占用 |
|------|----------|--------|----------|
| beam_size=1 | 最快 | 中等 | ~1.5GB |
| beam_size=5 | 中等 | 良好 | ~2.2GB |
| beam_size=10 | 较慢 | 最高 | ~3.5GB |

## 音频格式要求

### 支持格式
- **WAV**：16kHz 单声道（推荐）
- **MP3/M4A/FLAC**：自动转换
- **采样率**：自动重采样到 16kHz
- **声道**：自动转换为单声道

### 质量建议
- **信噪比**：> 10dB
- **单段时长**：< 10分钟
- **标准普通话**：获得最佳识别效果

## 常见问题

### Q1: 处理速度太慢？
**解决方案：**
- 降低 `beam_size` 至 3 或 1
- 使用 `compute_type=int8`
- 将长音频分段处理

### Q2: 识别准确率低？
**解决方案：**
- 提高音频质量（降噪）
- 增加 `beam_size` 至 10
- 使用合适的 `initial_prompt`

### Q3: 简繁混杂？
**解决方案：**
```python
initial_prompt = "以下是普通话的句子。"
```

### Q4: 显存不足（OOM）？
**解决方案：**
- 使用 `compute_type=float16` 或 `int8`
- 减少音频长度
- 减少批处理数量

## 项目结构

```
faster-whisper/
├── backend/
│   ├── core/
│   │   ├── config.py        # 配置管理
│   │   └── exceptions.py    # 异常定义
│   ├── modules/
│   │   └── audio_analysis/
│   │       ├── __init__.py
│   │       └── whisper_svc.py  # 核心服务
│   └── utils/
│       ├── constants.py     # 常量定义
│       └── logger.py        # 日志工具
├── config/
│   └── model_config.yaml    # 配置文件
├── models/
│   └── faster-whisper-large-v3/  # 模型目录（需要下载）
├── scripts/
│   └── test/
│       └── test_asr.py      # 测试脚本
├── docs/
│   └── 音频转文字技术实现说明书.md  # 技术文档
├── requirements.txt         # 依赖列表
└── README.md               # 本文档
```

## 许可证

本项目基于原有声纹识别系统的 ASR 模块提取，遵循相应许可证协议。

## 技术支持

如有问题，请参考：
- [音频转文字技术实现说明书](docs/音频转文字技术实现说明书.md)
- 测试脚本：`scripts/test/test_asr.py`
