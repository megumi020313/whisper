"""全局配置管理模块

从配置文件加载系统配置，提供统一的配置访问接口。
"""
import os
from pathlib import Path
from typing import Optional, Union
import yaml
from loguru import logger

from backend.core.exceptions import ConfigError


class Config:
    """全局配置类，单例模式"""
    
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置"""
        if self._initialized:
            return
        
        # 项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        
        # 加载配置文件
        self._load_config()
        
        # 确保必要的目录存在
        self._ensure_directories()
        
        self._initialized = True
    
    def _load_config(self) -> None:
        """从YAML文件加载配置"""
        config_file = self.project_root / "config" / "model_config.yaml"
        
        if not config_file.exists():
            raise ConfigError(f"配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")
        
        # 模型路径
        self.eres2netv2_path = self.project_root / config_data['model']['eres2netv2_path']
        self.vad_path = self.project_root / config_data['model']['vad_path']
        
        # VAD 参数 (V14.0)
        vad_config = config_data['vad']
        self.vad_threshold = vad_config['threshold']
        self.min_speech_duration_ms = vad_config['min_speech_duration_ms']
        self.min_silence_duration_ms = vad_config['min_silence_duration_ms']
        self.speech_pad_ms = vad_config['speech_pad_ms']
        
        # 声纹识别参数 (V14.0 - 严格拒识)
        speaker_config = config_data['speaker']
        self.high_threshold = speaker_config['high_threshold']  # S-Norm Z-Score高置信度阈值
        self.low_threshold = speaker_config['low_threshold']    # S-Norm Z-Score识别阈值
        self.top_k_ratio = speaker_config['top_k_ratio']
        self.batch_size = speaker_config.get('batch_size', 64)  # V14.2 显存保护
        
        # 音频参数
        audio_config = config_data['audio']
        self.sample_rate = audio_config['sample_rate']
        self.rms_gate_db = audio_config['rms_gate_db']  # V14.0: -100 关闭物理门限
        self.min_segment_duration = audio_config['min_segment_duration']
        self.max_segment_duration = audio_config['max_segment_duration']
        self.max_audio_duration = audio_config.get('max_audio_duration')
        
        # 会话参数
        session_config = config_data['session']
        self.max_silence_gap = session_config['max_silence_gap']
        self.min_interruption_duration = session_config['min_interruption_duration']
        self.interruption_rms_ratio = session_config['interruption_rms_ratio']
        self.noise_floor = session_config.get('noise_floor', 0.20)  # 占位
        
        # 注册参数
        reg_config = config_data['registration']
        self.num_registration_samples = reg_config['num_samples']
        self.noise_augmentation_enabled = reg_config['noise_augmentation']['enabled']
        self.noise_ratio_min = reg_config['noise_augmentation']['noise_ratio_min']
        self.noise_ratio_max = reg_config['noise_augmentation']['noise_ratio_max']
        
        # 存储配置
        storage_config = config_data['storage']
        self.vector_db_path = self.project_root / storage_config['local_path']
        self.cohort_path = self.project_root / storage_config['cohort_path']
        
        # Redis 配置
        redis_config = storage_config['redis']
        self.redis_enabled = redis_config['enabled']
        self.redis_host = redis_config['host']
        self.redis_port = redis_config['port']
        self.redis_db = redis_config['db']
        
        # 系统配置
        system_config = config_data['system']
        self.device = system_config['device']
        self.log_level = system_config['log_level']
        self.log_file = self.project_root / system_config['log_file']
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在
        
        创建日志目录、向量数据库目录、音频样本目录等。
        """
        directories = [
            self.vector_db_path,
            self.log_file.parent,
            self.project_root / "data" / "audio_samples",
            self.project_root / "data" / "noise"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 创建全局配置实例
config = Config()


def get_config() -> Config:
    """
    获取全局配置实例
    
    Returns:
        全局Config单例实例
    """
    return config
