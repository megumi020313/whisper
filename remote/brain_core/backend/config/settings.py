"""配置管理模块"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置类"""
    
    # 项目根目录
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # 模型配置（使用项目内相对路径）
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "llm" / "Qwen2.5-7B-Instruct-GPTQ-Int4")
    MODEL_NAME: str = "Qwen2.5-7B"
    
    # 显存关键配置 (配合 Project 1 共存)
    GPU_MEMORY_UTILIZATION: float = 0.5  # ⚠️ 严格限制只用 50% 显存
    MAX_MODEL_LEN: int = 4096            # 上下文长度限制
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: str = "sk-brain-core-secret"  # 简单的鉴权
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

