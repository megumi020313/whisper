"""日志配置模块"""
from loguru import logger
from pathlib import Path
import sys

from backend.core.config import config


def setup_logger() -> None:
    """
    配置日志系统
    
    配置loguru日志器，设置控制台和文件输出格式。
    """
    # 移除默认handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <level>{level}</level> - <level>{message}</level>",
        level=config.log_level
    )
    
    # 添加文件输出
    logger.add(
        config.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
        level=config.log_level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8"
    )


def get_logger(name: str = "faster-whisper-asr") -> logger:
    """
    获取logger实例

    Args:
        name: 日志器名称，默认"faster-whisper-asr"

    Returns:
        配置好的logger实例
    """
    return logger.bind(name=name)

