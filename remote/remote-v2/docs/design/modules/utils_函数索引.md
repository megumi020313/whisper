# Utils模块函数索引

## 模块说明

`src/utils/` 包含纯工具函数，无业务依赖，可被其他所有模块调用。

## 📁 子模块

### audio_utils.py - 音频处理工具

#### 音频加载与保存

##### load_audio
```python
def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]
```
**功能**: 加载音频文件并转换为指定格式

**参数**:
- `audio_path`: 音频文件路径
- `sample_rate`: 目标采样率，默认16000Hz
- `mono`: 是否转换为单声道，默认True

**返回**: `(audio_data, sample_rate)` 元组

**异常**: `AudioFormatError` - 音频格式不支持或加载失败

---

##### save_audio
```python
def save_audio(
    audio: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = 16000
) -> None
```
**功能**: 保存音频数据到文件

**参数**:
- `audio`: 音频数据（numpy数组）
- `output_path`: 输出文件路径
- `sample_rate`: 采样率，默认16000Hz

---

#### 音频预处理

##### resample_audio
```python
def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray
```
**功能**: 重采样音频到目标采样率

**参数**:
- `audio`: 原始音频数据
- `orig_sr`: 原始采样率
- `target_sr`: 目标采样率，默认16000Hz

**返回**: 重采样后的音频数据

---

##### validate_audio
```python
def validate_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: float = 0.5
) -> bool
```
**功能**: 验证音频格式和时长

**参数**:
- `audio`: 音频数据
- `sample_rate`: 采样率
- `max_duration`: 最大时长（秒），None表示不限制
- `min_duration`: 最小时长（秒），默认0.5秒

**返回**: 验证是否通过

**异常**: `AudioFormatError` - 音频不符合要求

---

##### check_audio_energy
```python
def check_audio_energy(
    audio_numpy: np.ndarray,
    threshold_db: float = -25.0
) -> bool
```
**功能**: 检查音频RMS能量是否超过阈值

**参数**:
- `audio_numpy`: 音频数据数组
- `threshold_db`: 能量阈值（dB），默认-25.0

**返回**: True表示能量足够，False表示静音

**使用场景**: 注册时过滤静音样本

---

#### 音频增强

##### augment_with_noise
```python
def augment_with_noise(
    clean_audio: np.ndarray,
    min_factor: float = 0.1,
    max_factor: float = 0.3
) -> np.ndarray
```
**功能**: 动态加噪，将音频与背景噪音按随机比例混合

**参数**:
- `clean_audio`: 纯净音频数据
- `min_factor`: 最小噪音因子，默认0.1
- `max_factor`: 最大噪音因子，默认0.3

**返回**: 增强后的音频数据

**使用场景**: 注册时数据增强，提升鲁棒性

---

#### 音频编解码

##### decode_opus_stream
```python
def decode_opus_stream(
    opus_bytes: bytes,
    sample_rate: int = 16000
) -> np.ndarray
```
**功能**: 解码Opus格式音频流

**参数**:
- `opus_bytes`: Opus编码的字节流
- `sample_rate`: 目标采样率，默认16000Hz

**返回**: 解码后的音频数据

**异常**: `AudioFormatError` - 解码失败

---

#### 说话人分割（Diarization）

##### calculate_diarization_intervals
```python
def calculate_diarization_intervals(
    raw_segments: List[Dict[str, Any]],
    total_duration: float,
    stride: float = 0.1,
    max_silence_gap: float = 15.0,
    identification_threshold: float = 1.8
) -> List[Dict[str, Any]]
```
**功能**: V12.0多人全库识别 + 网格投票算法

**参数**:
- `raw_segments`: 原始片段列表（包含start/end/user/score）
- `total_duration`: 音频总时长（秒）
- `stride`: 时间分辨率（秒），默认0.1
- `max_silence_gap`: 最大静音间隙（秒），默认15.0
- `identification_threshold`: 识别阈值（Z-Score），默认1.8

**返回**: 说话人分段列表，每个分段包含：
```python
{
    'start': float,      # 开始时间（秒）
    'end': float,        # 结束时间（秒）
    'user': str,         # 用户ID或'silence'/'unknown'
    'type': str,         # 'target'/'silence'/'unknown'
    'score': float,      # 置信度百分比（0-100）
    'raw_z_score': float,# 原始Z-score
    'window_count': int  # 窗口数量
}
```

**核心算法**:
1. 网格投票：将时间轴划分为stride间隔的网格
2. 窗口映射：每个识别窗口投票给覆盖的网格单元
3. 强力平滑：连续同用户片段合并
4. 静音检测：低于阈值标记为silence

**详细文档**: `docs/V12.0.md`

---

##### calculate_precise_intervals
```python
def calculate_precise_intervals(
    raw_windows: list,
    target_uid: Optional[str],
    total_duration: float,
    stride: float = 0.1,
    min_gap: float = 0.3,
    score: Optional[float] = None
) -> list
```
**功能**: 精细分段，将滑动窗口结果转换为连续时间段

**参数**:
- `raw_windows`: 原始窗口列表
- `target_uid`: 目标用户ID
- `total_duration`: 总时长
- `stride`: 窗口步长，默认0.1秒
- `min_gap`: 最小间隙，默认0.3秒
- `score`: 平均得分（可选）

**返回**: 精细分段列表

---

##### calculate_session_intervals
```python
def calculate_session_intervals(
    raw_segments: list,
    target_uid: str,
    max_silence_gap: float = 15.0,
    min_interruption_duration: float = 1.0,
    noise_floor: float = 0.20,
    interruption_rms_ratio: float = 0.6
) -> list
```
**功能**: 会话分段，将精细分段合并为会话级别

**参数**:
- `raw_segments`: 精细分段列表
- `target_uid`: 目标用户ID
- `max_silence_gap`: 最大静音间隙（秒），默认15.0
- `min_interruption_duration`: 最小打断时长（秒），默认1.0
- `noise_floor`: 噪音底噪阈值，默认0.20
- `interruption_rms_ratio`: 打断RMS比率，默认0.6

**返回**: 会话分段列表

---

##### map_score_to_percentage
```python
def map_score_to_percentage(
    z_score: float,
    low_threshold: float = 1.8,
    high_threshold: float = 4.0
) -> float
```
**功能**: 将AS-Norm Z-score映射为0-100百分比

**参数**:
- `z_score`: AS-Norm Z-score
- `low_threshold`: 低阈值，默认1.8
- `high_threshold`: 高阈值，默认4.0

**返回**: 百分比得分（0-100）

**映射规则**:
- `z < low`: 线性映射到0-50%
- `low <= z <= high`: 线性映射到50-95%
- `z > high`: 95-100%

---

### logger.py - 日志工具

##### setup_logger
```python
def setup_logger() -> None
```
**功能**: 配置日志系统

**说明**: 配置loguru日志器，设置控制台和文件输出格式

---

##### get_logger
```python
def get_logger(name: str = "remote-v2") -> logger
```
**功能**: 获取logger实例

**参数**:
- `name`: 日志器名称，默认"remote-v2"

**返回**: 配置好的logger实例

---

## 使用示例

### 音频处理流程
```python
from src.utils.audio_utils import load_audio, validate_audio, augment_with_noise

# 1. 加载音频
audio, sr = load_audio("test.wav", sample_rate=16000)

# 2. 验证音频
if validate_audio(audio, sr, min_duration=1.0):
    # 3. 数据增强
    augmented = augment_with_noise(audio, min_factor=0.1, max_factor=0.3)
```

### 说话人分割
```python
from src.utils.audio_utils import calculate_diarization_intervals

# 原始识别窗口结果
raw_segments = [
    {'start': 0.0, 'end': 1.0, 'user': 'user1', 'score': 3.5, 'rms': 0.1},
    {'start': 0.1, 'end': 1.1, 'user': 'user1', 'score': 3.8, 'rms': 0.12},
    # ...
]

# 网格投票分割
final_segments = calculate_diarization_intervals(
    raw_segments=raw_segments,
    total_duration=30.0,
    stride=0.1,
    identification_threshold=2.5
)

# 输出格式
for seg in final_segments:
    print(f"{seg['start']:.1f}s-{seg['end']:.1f}s: {seg['user']} ({seg['score']:.1f}%)")
```

---

**最后更新**: 2025-12-22  
**维护者**: 项目开发团队

