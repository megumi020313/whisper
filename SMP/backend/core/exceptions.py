"""自定义异常类 - 提供细粒度的错误分类"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional


class ErrorCode(Enum):
    """错误代码枚举"""
    # 模型相关错误 (E001-E099)
    MODEL_LOAD_FAILED = "E001"
    MODEL_INFERENCE_FAILED = "E002"
    MODEL_NOT_FOUND = "E003"
    MODEL_INVALID_FORMAT = "E004"
    MODEL_VERSION_MISMATCH = "E005"
    
    # 音频处理错误 (E100-E199)
    AUDIO_FORMAT_ERROR = "E100"
    AUDIO_INVALID_SAMPLE_RATE = "E101"
    AUDIO_INVALID_DURATION = "E102"
    AUDIO_INVALID_CHANNELS = "E103"
    AUDIO_CONVERSION_FAILED = "E104"
    
    # 推理相关错误 (E200-E299)
    INFERENCE_TIMEOUT = "E200"
    INFERENCE_FAILED = "E201"
    INFERENCE_INVALID_INPUT = "E202"
    
    # 资源相关错误 (E300-E399)
    RESOURCE_EXHAUSTED = "E300"
    MEMORY_ERROR = "E301"
    DEVICE_NOT_AVAILABLE = "E302"
    FILE_NOT_FOUND = "E303"
    FILE_PERMISSION_ERROR = "E304"
    
    # 配置相关错误 (E400-E499)
    CONFIG_INVALID = "E400"
    CONFIG_MISSING_REQUIRED = "E401"
    CONFIG_VALIDATION_FAILED = "E402"
    
    # 运行时错误 (E500-E599)
    RUNTIME_ERROR = "E500"
    INITIALIZATION_FAILED = "E501"
    CLEANUP_FAILED = "E502"


class AudioProcessingError(Exception):
    """音频处理基础异常类"""
    
    def __init__(
        self, 
        code: ErrorCode, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化异常
        
        Args:
            code: 错误代码
            message: 错误消息
            details: 错误详情字典
            cause: 原始异常（如果有）
        """
        self.code = code
        self.message = message
        self.details = details or {}
        self.cause = cause
        
        # 构造完整的错误消息
        full_message = f"[{code.value}] {message}"
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            full_message += f" ({detail_str})"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式"""
        return {
            "error_code": self.code.value,
            "error_name": self.code.name,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class ModelLoadError(AudioProcessingError):
    """模型加载错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.MODEL_LOAD_FAILED, message, details, cause)


class ModelInferenceError(AudioProcessingError):
    """模型推理错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.MODEL_INFERENCE_FAILED, message, details, cause)


class AudioFormatError(AudioProcessingError):
    """音频格式错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.AUDIO_FORMAT_ERROR, message, details, cause)


class InferenceTimeoutError(AudioProcessingError):
    """推理超时错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.INFERENCE_TIMEOUT, message, details, cause)


class ResourceExhaustedError(AudioProcessingError):
    """资源耗尽错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.RESOURCE_EXHAUSTED, message, details, cause)


class ConfigurationError(AudioProcessingError):
    """配置错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CONFIG_INVALID, message, details, cause)
