"""依赖注入模块"""
from functools import lru_cache
from backend.pipeline.inference_pipeline import InferencePipeline
from backend.core.config import get_config


@lru_cache()
def get_pipeline() -> InferencePipeline:
    """
    获取推理流水线实例（单例）
    
    Returns:
        InferencePipeline实例（使用lru_cache缓存）
    """
    # 从配置文件读取是否启用ASR
    config = get_config()
    enable_asr = getattr(config, 'asr_enabled', False)
    
    return InferencePipeline(enable_asr=enable_asr)

