#!/usr/bin/env python3
"""
Web服务配置管理
"""
from pathlib import Path
from typing import Optional

# 导入模型配置
from backend.config.model_config import ModelConfig

# 基础路径（基于项目根目录SMP/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
API_DIR = BACKEND_DIR / "api"
TRAINING_DIR = BACKEND_DIR / "training"

# 上传和存储路径
UPLOAD_DIR = API_DIR / "uploads"
AUDIO_STORAGE_DIR = UPLOAD_DIR / "registered"

# 模型路径（从模型配置读取）
MODELS_DIR = ModelConfig.get_models_dir()
SPEECHBRAIN_MODEL_DIR = ModelConfig.get_encoder_root()

# 模型选择配置（从模型配置读取）
SELECTED_MODEL = ModelConfig.get_selected_model()

# 音频参数（从模型配置读取）
SAMPLE_RATE = ModelConfig.get_sample_rate()
BIT_DEPTH = ModelConfig.get_bit_depth()
CHANNELS = ModelConfig.get_channels()

# 识别参数（从模型配置读取）
SIMILARITY_THRESHOLD = ModelConfig.get_similarity_threshold()
NUM_MFCC = ModelConfig.get_num_mfcc()

# Flask配置
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 51003
FLASK_DEBUG = False  # 关闭debug模式以修复HTTPS握手问题

# CORS配置
CORS_ORIGINS = ["*"]  # 开发环境允许所有来源，生产环境应限制

# 音频文件配置（从模型配置读取）
MAX_AUDIO_SIZE = ModelConfig.get_max_audio_size()
ALLOWED_EXTENSIONS = ModelConfig.get_allowed_extensions()


class Config:
    """配置管理类"""
    
    @staticmethod
    def get_latest_speaker_embeddings() -> Optional[Path]:
        """
        获取speaker_embeddings.json文件路径
        
        从模型配置模块读取
        """
        return ModelConfig.get_latest_speaker_embeddings()
    
    @staticmethod
    def ensure_directories() -> None:
        """确保必要的目录存在"""
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_speaker_audio_path(speaker_id: str) -> Path:
        """获取说话人音频文件路径"""
        return AUDIO_STORAGE_DIR / f"{speaker_id}.wav"


# 初始化配置
Config.ensure_directories()

