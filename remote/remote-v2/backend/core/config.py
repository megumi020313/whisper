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
        self.funasr_model_path = self.project_root / config_data['model'].get(
            'funasr_model_path',
            'models/paraformer/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
        )
        self.funasr_vad_model_path = self.project_root / config_data['model'].get(
            'funasr_vad_model_path',
            'models/paraformer/speech_fsmn_vad_zh-cn-16k-common-pytorch'
        )
        self.funasr_punc_model_path = self.project_root / config_data['model'].get(
            'funasr_punc_model_path',
            'models/paraformer/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
        )
        
        # GPU设备分配
        device_config = config_data.get('device', {})
        self.speaker_model_device = device_config.get('speaker_model', 'cuda:0')
        self.vad_model_device = device_config.get('vad_model', 'cuda:0')
        self.asr_model_device = device_config.get('asr_model', 'cuda:0')
        self.default_device = device_config.get('default', 'cuda:0')
        
        # VAD 参数
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
        self.speaker_batch_size = speaker_config.get('batch_size', 16)  # 批量推理batch_size（用于Pipeline）
        
        # V3.1 滑动窗口声纹流配置
        self.sv_window_size_s = speaker_config.get('sv_window_size_s', 1.5)  # 滑动窗口大小（秒）
        self.sv_window_step_s = speaker_config.get('sv_window_step_s', 0.5)  # 滑动窗口步长（秒）
        
        # V3.1.1 方案F：特征提取专用padding
        self.extraction_pad_ms = speaker_config.get('extraction_pad_ms', 300)  # 特征提取padding（毫秒）
        
        # 音频参数
        audio_config = config_data['audio']
        self.sample_rate = audio_config['sample_rate']
        self.rms_gate_db = audio_config['rms_gate_db']  # V14.0: -100 关闭物理门限
        self.min_segment_duration = audio_config['min_segment_duration']
        self.max_segment_duration = audio_config['max_segment_duration']
        self.max_audio_duration = audio_config.get('max_audio_duration')
        
        # 音频文件配置
        file_config = config_data.get('file', {})
        self.max_audio_size = file_config.get('max_audio_size', 524288000)  # 默认500MB
        self.allowed_extensions = set(file_config.get('allowed_extensions', ['.wav', '.mp3', '.mp4', '.m4a', '.webm', '.ogg', '.flac']))
        
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
        
        # 声纹模板存储（Vector DB）
        vector_db_config = storage_config.get('vector_db')
        if vector_db_config is None:
            raise ConfigError("配置缺少 storage.vector_db")
        self.storage_mode = vector_db_config.get('mode', 'local')  # local, redis, hybrid
        self.vector_db_path = self.project_root / vector_db_config.get('local_path', storage_config.get('local_path', 'data/vector_db'))
        self.cohort_path = self.project_root / vector_db_config.get('cohort_path', storage_config.get('cohort_path', 'data/vector_db/cohort.npy'))
        
        # Redis 配置
        redis_config = vector_db_config.get('redis', storage_config.get('redis', {}))
        self.redis_enabled = redis_config.get('enabled', False)
        self.redis_host = redis_config.get('host', 'localhost')
        self.redis_port = redis_config.get('port', 6379)
        self.redis_db = redis_config.get('db', 0)
        self.redis_password = redis_config.get('password')
        self.redis_timeout = redis_config.get('timeout', 5)
        self.redis_max_connections = redis_config.get('max_connections', 10)
        
        # IoTDB 配置（时序数据存储）
        iotdb_config = storage_config.get('iotdb', {})
        self.iotdb_enabled = iotdb_config.get('enabled', False)
        self.iotdb_host = iotdb_config.get('host', '127.0.0.1')
        self.iotdb_port = iotdb_config.get('port', 6667)
        self.iotdb_username = iotdb_config.get('username', 'root')
        self.iotdb_password = iotdb_config.get('password', 'root')
        self.iotdb_database = iotdb_config.get('database', 'root.store_01')
        self.iotdb_timeout = iotdb_config.get('timeout', 5)
        
        # 系统配置
        system_config = config_data['system']
        self.device = system_config['device']
        self.log_level = system_config['log_level']
        self.log_file = self.project_root / system_config['log_file']
        
        # ASR 配置（多模态融合）
        asr_config = config_data.get('asr', {})
        self.asr_enabled = asr_config.get('enabled', False)
        self.asr_language = asr_config.get('language', 'zh')
        self.asr_initial_prompt = asr_config.get('initial_prompt', '以下是普通话的句子。')
        self.asr_beam_size = asr_config.get('beam_size', 5)
        self.asr_device = asr_config.get('device', 'cuda')
        self.asr_hotword = asr_config.get('hotword')
        self.asr_batch_size_s = asr_config.get('batch_size_s', 300)
        
        # V3.0 对话智能引擎配置
        diarization_config = config_data.get('diarization', {})
        self.diarization_enabled = diarization_config.get('enabled', True)
        self.diarization_time_threshold = diarization_config.get('time_threshold', 1.5)
        self.diarization_similarity_threshold = diarization_config.get('similarity_threshold', 0.7)
        self.diarization_low_confidence_threshold = diarization_config.get('low_confidence_threshold', 2.5)
        self.diarization_max_merge_duration = diarization_config.get('max_merge_duration', 10.0)
        self.diarization_anchor_correction = diarization_config.get('anchor_correction', True)
        # V3.1混合模式已废弃：SV直接使用全局VAD生成的optimized_segments
        # self.diarization_vad_min_silence_ms = diarization_config.get('vad_min_silence_ms', 200)
        
        # V3.1 词-说话人对齐配置（IoU > Center > Score三级决策）
        self.alignment_iou_threshold = diarization_config.get('alignment_iou_threshold', 0.6)
        
        # V3.1.1 方案B：ASR过滤容忍度配置
        self.asr_filter_iou_threshold = diarization_config.get('asr_filter_iou_threshold', 0.3)  # ASR词IoU阈值
        self.asr_filter_boundary_tolerance_ms = diarization_config.get('asr_filter_boundary_tolerance_ms', 100)  # 边界容忍距离
        
        # V3.1.1 重叠质量检查配置
        self.diarization_min_overlap_ratio = diarization_config.get('min_overlap_ratio', 0.5)
        
        # V3.2 词-说话人对齐的声纹验证配置
        self.diarization_min_similarity_for_alignment = diarization_config.get('min_similarity_for_alignment', 0.65)
        
        # V3.3 两阶段校正配置（核心机制 - 分段动态梯度）
        boundary_correction_config = diarization_config.get('boundary_correction', {})
        self.boundary_correction_enabled = boundary_correction_config.get('enabled', True)
        self.boundary_correction_window_s = boundary_correction_config.get('window_s', 1.5)
        
        # 分段动态梯度参数
        self.boundary_correction_base_diff = boundary_correction_config.get('base_diff', 0.08)
        self.boundary_correction_penalty_factor = boundary_correction_config.get('penalty_factor', 0.2)
        self.boundary_correction_out_of_zone_threshold = boundary_correction_config.get('out_of_zone_threshold', 0.35)

        # V3.5 短词声纹覆盖（结构性修正）
        self.voiceprint_override_min_diff = diarization_config.get('voiceprint_override_min_diff', 0.25)
        self.voiceprint_override_min_duration = diarization_config.get('voiceprint_override_min_duration', 0.2)
        self.voiceprint_override_max_duration = diarization_config.get('voiceprint_override_max_duration', 0.8)
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在
        
        创建日志目录、向量数据库目录、音频样本目录等。
        """
        directories = [
            self.vector_db_path,
            self.log_file.parent,
            self.project_root / "logs" / "Speaker Recognition",  # 声纹识别详细日志
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
