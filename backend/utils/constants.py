"""项目常量定义

定义项目中使用的所有常量，避免魔法数字。

⚠️ 重要说明：
- 本文件只包含真正的常量（不会改变的固定值）
- 可配置的参数应该在 model_config.yaml 中定义
- 代码中应该使用 config.xxx 而不是这里的常量

常量 vs 配置的区别：
- 常量：HTTP状态码、错误码、物理常量等（永远不变）
- 配置：阈值、参数、路径等（可能需要调整）
"""

# ==================== 物理常量 ====================

# 音频采样率（标准值，不可配置）
STANDARD_SAMPLE_RATE = 16000  # 标准采样率（Hz）

# 数学常量
EPSILON = 1e-8  # 避免除零的极小值
MILLISECONDS_PER_SECOND = 1000  # 毫秒转秒

# ==================== HTTP状态码（标准） ====================

HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

# ==================== 业务错误码（自定义） ====================

ERROR_CODE_SUCCESS = 200
ERROR_CODE_VALIDATION_ERROR = 1001
ERROR_CODE_RESOURCE_NOT_FOUND = 1002
ERROR_CODE_PERMISSION_DENIED = 1003
ERROR_CODE_AUDIO_FORMAT_ERROR = 1004
ERROR_CODE_MODEL_INFERENCE_ERROR = 1005
ERROR_CODE_CONFIG_ERROR = 1006
ERROR_CODE_MODEL_LOAD_ERROR = 1007

# ==================== 文件大小限制（业务规则） ====================

MAX_FILE_SIZE_MB = 10  # 最大文件大小（MB）
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 最大文件大小（字节）

# ==================== 日志常量 ====================

LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# ==================== 模型常量（固定值） ====================

# Mel频谱参数（模型架构固定）
DEFAULT_MEL_BINS = 80  # 默认Mel频谱数量
DEFAULT_N_FFT = 512  # 默认FFT点数
DEFAULT_HOP_LENGTH = 160  # 默认跳跃长度

# ==================== 注册质量常量 ====================

# 注册样本数量范围（业务规则）
MIN_REGISTRATION_SAMPLES = 3  # 最小注册样本数
STANDARD_REGISTRATION_SAMPLES = 8  # 标准注册样本数
MAX_REGISTRATION_SAMPLES = 10  # 最大注册样本数

# 注册质量阈值（业务规则）
MIN_SNR_DB = 5.0  # 最小信噪比（dB）
MIN_TOTAL_DURATION_SECONDS = 5.0  # 最小总时长（秒）

# ==================== 字符串常量 ====================

# 用户ID标识
UNKNOWN_USER_ID = "unknown"
SILENCE_USER_ID = "silence"
CUSTOMER_USER_ID = "customer"

# 状态标识
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

