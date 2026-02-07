#!/usr/bin/env python3
"""
模型配置管理模块
统一管理所有模型相关参数，从 model_config.yaml 读取配置
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class ModelConfig:
    """模型配置管理类"""
    
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None
    
    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """加载配置文件"""
        if cls._config is not None:
            return cls._config
        
        # 获取配置文件路径
        config_file = Path(__file__).resolve().parent / "model_config.yaml"
        cls._config_path = config_file
        
        if not config_file.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)
        
        return cls._config
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取完整配置"""
        return cls._load_config()
    
    @classmethod
    def get_selected_model(cls) -> Optional[str]:
        """获取选定的模型名称"""
        config = cls._load_config()
        return config.get("model_selection", {}).get("selected_model")
    
    @classmethod
    def get_models_dir(cls) -> Path:
        """获取模型目录路径（相对于项目根目录）"""
        config = cls._load_config()
        project_root = Path(__file__).resolve().parent.parent.parent
        models_dir = config.get("model_selection", {}).get("models_dir", "backend/training/models")
        return project_root / models_dir
    
    @classmethod
    def get_encoder_root(cls) -> Path:
        """获取编码器根目录路径（相对于项目根目录）"""
        config = cls._load_config()
        project_root = Path(__file__).resolve().parent.parent.parent
        encoder_root = config.get("model_selection", {}).get("encoder_root", "backend/training/model/speechbrain")
        return project_root / encoder_root
    
    @classmethod
    def get_sample_rate(cls) -> int:
        """获取采样率"""
        config = cls._load_config()
        return config.get("audio", {}).get("sample_rate", 16000)
    
    @classmethod
    def get_bit_depth(cls) -> int:
        """获取位深度"""
        config = cls._load_config()
        return config.get("audio", {}).get("bit_depth", 16)
    
    @classmethod
    def get_channels(cls) -> int:
        """获取声道数"""
        config = cls._load_config()
        return config.get("audio", {}).get("channels", 1)
    
    @classmethod
    def get_similarity_threshold(cls) -> float:
        """获取相似度阈值"""
        config = cls._load_config()
        return config.get("recognition", {}).get("similarity_threshold", 0.75)
    
    @classmethod
    def get_num_mfcc(cls) -> int:
        """获取MFCC特征数量"""
        config = cls._load_config()
        return config.get("recognition", {}).get("num_mfcc", 13)
    
    @classmethod
    def get_multi_speaker(cls) -> bool:
        """获取是否支持多说话人"""
        config = cls._load_config()
        return config.get("recognition", {}).get("multi_speaker", True)
    
    @classmethod
    def get_max_audio_size(cls) -> int:
        """获取最大音频文件大小（字节）"""
        config = cls._load_config()
        return config.get("file", {}).get("max_audio_size", 10 * 1024 * 1024)
    
    @classmethod
    def get_allowed_extensions(cls) -> set[str]:
        """获取允许的音频文件扩展名"""
        config = cls._load_config()
        extensions = config.get("file", {}).get("allowed_extensions", [".wav"])
        return set(extensions)
    
    @classmethod
    def get_latest_speaker_embeddings(cls) -> Optional[Path]:
        """
        获取speaker_embeddings.json文件路径
        
        优先级：
        1. 如果设置了selected_model，使用指定的模型
        2. 否则自动使用最新的模型（按修改时间排序）
        """
        models_dir = cls.get_models_dir()
        
        if not models_dir.exists():
            return None
        
        # 如果指定了特定模型，优先使用
        selected_model = cls.get_selected_model()
        if selected_model:
            model_path = models_dir / selected_model / "speaker_embeddings.json"
            if model_path.exists():
                print(f"✅ 使用指定模型: {selected_model}")
                return model_path
            else:
                print(f"⚠️  警告: 指定的模型 '{selected_model}' 不存在，将使用最新模型")
        
        # 否则使用最新的模型
        pattern = "speechbrain_model_*/speaker_embeddings.json"
        matches = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            latest_model = matches[0].parent.name
            print(f"✅ 自动使用最新模型: {latest_model}")
            return matches[0]
        
        return None
    
    @classmethod
    def reload_config(cls) -> None:
        """重新加载配置文件"""
        cls._config = None
        cls._load_config()

