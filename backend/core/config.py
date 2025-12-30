"""ASR 配置管理模块

从配置文件加载 ASR 相关配置，提供统一的配置访问接口。
"""
import os
from pathlib import Path
from typing import Optional
import yaml

from backend.core.exceptions import ConfigError


class Config:
    """ASR 配置类，单例模式"""

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
        self.whisper_path = self.project_root / config_data['model'].get('whisper_path', 'models/faster-whisper-large-v3')

        # GPU设备分配
        device_config = config_data.get('device', {})
        self.asr_model_device = device_config.get('asr_model', 'cuda:0')
        self.default_device = device_config.get('default', 'cuda:0')

        # 系统配置
        system_config = config_data['system']
        self.log_level = system_config['log_level']
        self.log_file = self.project_root / system_config['log_file']

        # ASR 配置
        asr_config = config_data.get('asr', {})
        self.asr_enabled = asr_config.get('enabled', True)
        self.asr_language = asr_config.get('language', 'zh')
        self.asr_initial_prompt = asr_config.get('initial_prompt', '以下是普通话的句子。')
        self.asr_beam_size = asr_config.get('beam_size', 5)
        self.asr_compute_type = asr_config.get('compute_type', 'float16')
        self.asr_vad_filter = asr_config.get('vad_filter', False)

        # 音频处理配置
        audio_config = config_data.get('audio', {})
        self.input_path = audio_config.get('input_path')
        self.output_dir = audio_config.get('output_dir', 'output')
        self.supported_formats = audio_config.get('supported_formats', ['.wav', '.mp3', '.m4a', '.flac', '.ogg'])
        self.audio_sample_rate = audio_config.get('sample_rate', 16000)
        self.recursive = audio_config.get('recursive', True)

    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        directories = [
            self.log_file.parent,
            self.project_root / "data" / "audio_samples"
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
