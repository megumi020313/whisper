"""自定义异常类定义

定义了声纹识别系统中使用的所有自定义异常类，用于更精确的错误处理和定位。
"""
from typing import Optional


class RemoteV2Exception(Exception):
    """声纹识别系统基础异常类"""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class AudioProcessingError(RemoteV2Exception):
    """音频处理错误"""
    pass


class AudioFormatError(AudioProcessingError):
    """音频格式错误"""
    pass


class ModelLoadError(RemoteV2Exception):
    """模型加载错误"""
    pass


class ModelInferenceError(RemoteV2Exception):
    """模型推理错误"""
    pass


class InferenceTimeoutError(RemoteV2Exception):
    """推理超时错误"""
    pass


class VectorStorageError(RemoteV2Exception):
    """向量存储错误"""
    pass


class UserNotFoundError(RemoteV2Exception):
    """用户不存在错误"""
    pass


class UserAlreadyExistsError(RemoteV2Exception):
    """用户已存在错误"""
    pass


class ResourceExhaustedError(RemoteV2Exception):
    """系统资源耗尽错误"""
    pass


class ConfigError(RemoteV2Exception):
    """配置错误"""
    pass

