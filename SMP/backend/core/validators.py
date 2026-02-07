"""音频数据验证器 - 提供输入验证和安全检查"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union


class AudioValidator:
    """音频数据格式验证器"""
    
    @staticmethod
    def validate_audio_format(
        audio: np.ndarray, 
        expected_rate: int = 16000,
        max_duration: float | None = 30.0,
        allow_stereo: bool = False
    ) -> bool:
        """
        验证音频数据格式是否符合要求
        
        Args:
            audio: 音频数据数组
            expected_rate: 期望的采样率
            max_duration: 最大允许时长（秒），None表示不限制
            allow_stereo: 是否允许立体声
            
        Returns:
            验证通过返回True
            
        Raises:
            ValueError: 音频格式不符合要求
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Audio must be numpy.ndarray, got {type(audio)}")
        
        # 检查维度
        if audio.ndim > 2:
            raise ValueError(f"Audio must be 1D (mono) or 2D (stereo), got {audio.ndim}D")
        
        if audio.ndim == 2 and not allow_stereo:
            raise ValueError("Stereo audio not allowed, expected mono (1D array)")
        
        if audio.ndim == 1 and audio.size == 0:
            raise ValueError("Audio array is empty")
        
        # 检查时长（如果设置了max_duration）
        if max_duration is not None and max_duration > 0:
            max_samples = int(expected_rate * max_duration)
            actual_samples = audio.shape[0] if audio.ndim == 1 else audio.shape[1]
            
            if actual_samples > max_samples:
                raise ValueError(
                    f"Audio too long: {actual_samples} samples ({actual_samples/expected_rate:.2f}s) "
                    f"> {max_samples} samples ({max_duration}s)"
                )
        
        # 检查数值范围
        if audio.dtype in [np.float32, np.float64]:
            if np.any(np.abs(audio) > 1.0):
                raise ValueError(
                    f"Float audio values must be in range [-1, 1], "
                    f"got range [{audio.min():.3f}, {audio.max():.3f}]"
                )
        elif audio.dtype == np.int16:
            if np.any(np.abs(audio) > 32768):
                raise ValueError(
                    f"Int16 audio values must be in range [-32768, 32767], "
                    f"got range [{audio.min()}, {audio.max()}]"
                )
        
        return True
    
    @staticmethod
    def sanitize_path(path: Union[str, Path], base_dir: Path) -> Path:
        """
        验证并清理文件路径，防止路径遍历攻击
        
        Args:
            path: 待验证的路径
            base_dir: 基础目录，路径必须在此目录下
            
        Returns:
            清理后的安全路径
            
        Raises:
            ValueError: 路径不安全
        """
        if not isinstance(base_dir, Path):
            base_dir = Path(base_dir)
        
        clean_path = Path(path).resolve()
        base_dir_resolved = base_dir.resolve()
        
        # 检查路径是否在基础目录下
        try:
            clean_path.relative_to(base_dir_resolved)
        except ValueError:
            raise ValueError(
                f"Invalid path: '{clean_path}' is outside base directory '{base_dir_resolved}'"
            )
        
        return clean_path
    
    @staticmethod
    def validate_frame_size(
        frame_size: int,
        sample_rate: int = 16000,
        min_ms: float = 10.0,
        max_ms: float = 100.0
    ) -> bool:
        """
        验证帧大小是否合理
        
        Args:
            frame_size: 帧大小（样本数）
            sample_rate: 采样率
            min_ms: 最小帧长（毫秒）
            max_ms: 最大帧长（毫秒）
            
        Returns:
            验证通过返回True
            
        Raises:
            ValueError: 帧大小不合理
        """
        if frame_size <= 0:
            raise ValueError(f"Frame size must be positive, got {frame_size}")
        
        frame_ms = (frame_size / sample_rate) * 1000.0
        
        if frame_ms < min_ms or frame_ms > max_ms:
            raise ValueError(
                f"Frame duration {frame_ms:.1f}ms is outside valid range "
                f"[{min_ms:.1f}ms, {max_ms:.1f}ms]"
            )
        
        return True
    
    @staticmethod
    def validate_sample_rate(sample_rate: int, allowed_rates: list[int] = None) -> bool:
        """
        验证采样率是否支持
        
        Args:
            sample_rate: 采样率
            allowed_rates: 允许的采样率列表
            
        Returns:
            验证通过返回True
            
        Raises:
            ValueError: 采样率不支持
        """
        if allowed_rates is None:
            allowed_rates = [8000, 16000, 22050, 44100, 48000]
        
        if sample_rate not in allowed_rates:
            raise ValueError(
                f"Sample rate {sample_rate} not supported. "
                f"Allowed rates: {allowed_rates}"
            )
        
        return True
