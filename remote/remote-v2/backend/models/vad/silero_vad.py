"""Silero VAD 模型封装"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from backend.core.exceptions import ModelLoadError, ModelInferenceError
from backend.core.config import get_config
from backend.utils.logger import get_logger


class SileroVAD:
    """Silero VAD 语音活动检测模型"""
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        初始化VAD模型
        
        Args:
            model_path: 模型文件路径
            device: 设备 (cuda:0, cuda:1, cpu)
        """
        self.logger = get_logger()
        self.config = get_config()
        
        self.device = device or self.config.device
        self.model_path = model_path or self.config.vad_path / "silero_vad.jit"
        
        self.model = None
        self.utils = None  # 存储Silero VAD工具函数
        self._load_model()
    
    def _load_model(self):
        """加载VAD模型"""
        try:
            self.logger.info(f"Loading Silero VAD model on {self.device}")
            
            # 从本地models/vad文件夹加载（不联网下载）
            vad_repo_path = self.config.vad_path
            
            if not vad_repo_path.exists():
                raise ModelLoadError(
                    f"VAD repository not found at {vad_repo_path}. "
                    "Please ensure the models/vad folder exists.",
                    {"expected_path": str(vad_repo_path)}
                )
            
            # 使用本地仓库加载模型和工具函数
            self.model, self.utils = torch.hub.load(
                repo_or_dir=str(vad_repo_path),
                model='silero_vad',
                source='local',  # 关键：使用本地源
                onnx=False,
                trust_repo=True
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"✅ Silero VAD loaded from local: {vad_repo_path}")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Silero VAD model: {str(e)}",
                {"vad_path": str(self.config.vad_path), "device": self.device}
            )
    
    def detect(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        threshold: Optional[float] = None
    ) -> bool:
        """
        检测音频中是否有人声（使用官方get_speech_timestamps方法）
        
        Args:
            audio: 音频数据 (numpy array)
            sample_rate: 采样率
            threshold: VAD阈值，默认使用配置值
            
        Returns:
            是否检测到人声
            
        Raises:
            ModelInferenceError: 推理失败
        """
        threshold = threshold or self.config.vad_threshold
        
        try:
            # 转换为torch tensor
            if not isinstance(audio, torch.Tensor):
                audio_tensor = torch.FloatTensor(audio).to(self.device)
            else:
                audio_tensor = audio.to(self.device)
            
            # 确保是1D tensor
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # 使用官方get_speech_timestamps函数自动处理长音频
            get_speech_timestamps = self.utils[0]
            
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=250,
                min_silence_duration_ms=50,
                window_size_samples=512,  # 16kHz采样率使用512
                speech_pad_ms=10
            )
            
            # 如果检测到任何语音片段，返回True
            has_speech = len(speech_timestamps) > 0
            
            if has_speech:
                total_speech_duration = sum(
                    (ts['end'] - ts['start']) / sample_rate 
                    for ts in speech_timestamps
                )
                self.logger.debug(
                    f"VAD detection: found {len(speech_timestamps)} speech segments, "
                    f"total duration={total_speech_duration:.2f}s"
                )
            else:
                self.logger.debug("VAD detection: no speech detected")
            
            return has_speech
            
        except Exception as e:
            raise ModelInferenceError(
                f"VAD inference failed: {str(e)}",
                {"audio_shape": audio.shape, "sample_rate": sample_rate}
            )
    
    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: Optional[float] = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 30,
        merge_short_segments: bool = True
    ) -> List[Dict[str, int]]:
        """
        获取语音片段的时间戳（返回所有人声片段的起止位置）
        
        V13.0 优化：增加 min_silence_duration_ms 到 500ms，防止说话停顿导致片段破碎。
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            threshold: VAD阈值
            min_speech_duration_ms: 最小语音持续时间（毫秒，默认250ms）
            min_silence_duration_ms: 最小静音持续时间（毫秒，默认500ms，V13.0关键参数）
            speech_pad_ms: 语音片段前后填充（毫秒）
            merge_short_segments: 是否合并短片段以满足最小时长要求
            
        Returns:
            时间戳列表 [{'start': int, 'end': int}, ...]
        """
        threshold = threshold or self.config.vad_threshold
        
        try:
            # 转换为torch tensor
            if not isinstance(audio, torch.Tensor):
                audio_tensor = torch.FloatTensor(audio).to(self.device)
            else:
                audio_tensor = audio.to(self.device)
            
            # 确保是1D tensor
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # 使用Silero VAD的官方工具函数
            get_speech_timestamps = self.utils[0]
            
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                window_size_samples=512,  # 16kHz采样率使用512
                speech_pad_ms=speech_pad_ms
            )
            
            # 合并短片段以满足最小时长要求
            if merge_short_segments and timestamps:
                timestamps = self._merge_short_segments(
                    timestamps, 
                    min_duration_ms=min_speech_duration_ms,
                    max_gap_ms=min_silence_duration_ms * 2,  # 允许的最大间隔
                    sample_rate=sample_rate
                )
            
            return timestamps
            
        except Exception as e:
            self.logger.warning(f"Failed to get speech timestamps: {str(e)}")
            return []
    
    def _merge_short_segments(
        self,
        timestamps: List[Dict[str, int]],
        min_duration_ms: int = 3000,
        max_gap_ms: int = 600,
        sample_rate: int = 16000
    ) -> List[Dict[str, int]]:
        """
        合并短片段以满足最小时长要求
        
        策略：
        1. 如果片段<3秒且下一个片段间隔<600ms，则合并
        2. 如果片段<3秒且孤立，保留（总比丢弃好）
        
        Args:
            timestamps: 原始时间戳列表
            min_duration_ms: 目标最小时长（毫秒）
            max_gap_ms: 允许合并的最大间隔（毫秒）
            sample_rate: 采样率
            
        Returns:
            合并后的时间戳列表
        """
        if not timestamps:
            return timestamps
        
        min_duration_samples = int(min_duration_ms * sample_rate / 1000)
        max_gap_samples = int(max_gap_ms * sample_rate / 1000)
        
        merged = []
        current = timestamps[0].copy()
        
        for next_seg in timestamps[1:]:
            current_duration = current['end'] - current['start']
            gap = next_seg['start'] - current['end']
            
            # 如果当前片段太短且间隔不大，尝试合并
            if current_duration < min_duration_samples and gap <= max_gap_samples:
                # 合并到下一个片段
                current['end'] = next_seg['end']
                self.logger.debug(
                    f"Merged segment: {current_duration/sample_rate:.2f}s + gap {gap/sample_rate:.2f}s + {(next_seg['end']-next_seg['start'])/sample_rate:.2f}s"
                )
            else:
                # 保存当前片段（即使短于3秒也保留）
                merged.append(current)
                current = next_seg.copy()
        
        # 添加最后一个片段
        merged.append(current)
        
        # 统计合并效果
        original_count = len(timestamps)
        merged_count = len(merged)
        long_segments = sum(1 for seg in merged if (seg['end'] - seg['start']) >= min_duration_samples)
        
        self.logger.info(
            f"Segment merging: {original_count} → {merged_count} segments, "
            f"{long_segments}/{merged_count} meet 3s requirement"
        )
        
        return merged
