"""错误处理器 - 提供统一的错误处理和恢复机制"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Any, Dict
from pathlib import Path

from .exceptions import (
    AudioProcessingError, 
    ErrorCode, 
    ModelLoadError,
    InferenceTimeoutError,
    ResourceExhaustedError
)


class ErrorHandler:
    """统一的错误处理器"""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        初始化错误处理器
        
        Args:
            log_file: 日志文件路径（可选）
        """
        self.logger = self._setup_logger(log_file)
        self.error_count: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCode, Callable] = {}
        self._register_default_strategies()
    
    def _setup_logger(self, log_file: Optional[Path] = None) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("AudioProcessing")
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器（如果指定）
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_default_strategies(self) -> None:
        """注册默认的错误恢复策略"""
        self.recovery_strategies[ErrorCode.MODEL_LOAD_FAILED] = self._fallback_to_backup_model
        self.recovery_strategies[ErrorCode.INFERENCE_TIMEOUT] = self._restart_inference_engine
        self.recovery_strategies[ErrorCode.RESOURCE_EXHAUSTED] = self._cleanup_resources
    
    def handle_error(self, error: AudioProcessingError, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        处理错误并尝试恢复
        
        Args:
            error: 音频处理错误
            context: 错误上下文信息
            
        Returns:
            恢复成功返回True，否则返回False
        """
        # 记录错误
        self._log_error(error, context)
        
        # 更新错误计数
        error_key = f"{error.code.value}:{error.message}"
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        
        # 尝试恢复
        if error.code in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error.code]
                return recovery_func(error, context)
            except Exception as e:
                self.logger.error(f"Recovery failed for {error.code}: {e}")
                return False
        
        return False
    
    def _log_error(self, error: AudioProcessingError, context: Optional[Dict[str, Any]] = None) -> None:
        """记录错误详情"""
        log_data = {
            "error_code": error.code.value,
            "error_name": error.code.name,
            "message": error.message,
            "details": error.details,
        }
        
        if context:
            log_data["context"] = context
        
        if error.cause:
            log_data["cause"] = str(error.cause)
        
        self.logger.error(
            f"Error {error.code.value}: {error.message}",
            extra=log_data
        )
    
    def _fallback_to_backup_model(self, error: AudioProcessingError, context: Optional[Dict[str, Any]]) -> bool:
        """回退到备份模型"""
        self.logger.info("Attempting to fallback to backup model...")
        # 这里应该实现实际的备份模型加载逻辑
        # 目前只是一个占位符
        return False
    
    def _restart_inference_engine(self, error: AudioProcessingError, context: Optional[Dict[str, Any]]) -> bool:
        """重启推理引擎"""
        self.logger.info("Attempting to restart inference engine...")
        # 这里应该实现实际的引擎重启逻辑
        return False
    
    def _cleanup_resources(self, error: AudioProcessingError, context: Optional[Dict[str, Any]]) -> bool:
        """清理资源"""
        self.logger.info("Attempting to cleanup resources...")
        # 这里应该实现实际的资源清理逻辑
        return False
    
    def register_recovery_strategy(self, error_code: ErrorCode, strategy: Callable) -> None:
        """
        注册自定义的错误恢复策略
        
        Args:
            error_code: 错误代码
            strategy: 恢复策略函数
        """
        self.recovery_strategies[error_code] = strategy
        self.logger.info(f"Registered recovery strategy for {error_code.name}")
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计信息"""
        return self.error_count.copy()
    
    def reset_error_stats(self) -> None:
        """重置错误统计"""
        self.error_count.clear()


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    重试装饰器 - 在发生错误时自动重试
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍增系数
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator
