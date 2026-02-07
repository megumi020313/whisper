#!/usr/bin/env python3
"""
音频格式验证工具
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


class AudioValidator:
    """音频格式验证器"""
    
    REQUIRED_SAMPLE_RATE = 16000
    REQUIRED_BIT_DEPTH = 16
    REQUIRED_CHANNELS = 1
    
    @classmethod
    def validate_audio_file(cls, file_path: Path) -> Tuple[bool, str]:
        """
        验证音频文件格式
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # 检查文件是否存在
            if not file_path.exists():
                return False, f"文件不存在: {file_path}"
            
            # 检查文件扩展名
            if file_path.suffix.lower() != ".wav":
                return False, f"不支持的文件格式: {file_path.suffix}，请使用WAV格式"
            
            # 读取音频文件
            audio, sample_rate = sf.read(str(file_path))
            
            # 检查采样率
            if sample_rate != cls.REQUIRED_SAMPLE_RATE:
                return False, f"采样率不匹配: {sample_rate}Hz，需要 {cls.REQUIRED_SAMPLE_RATE}Hz"
            
            # 检查声道数
            if audio.ndim == 1:
                channels = 1
            else:
                channels = audio.shape[1]
            
            if channels != cls.REQUIRED_CHANNELS:
                return False, f"声道数不匹配: {channels}，需要单声道"
            
            # 检查音频时长（至少1秒）
            duration = len(audio) / sample_rate
            if duration < 1.0:
                return False, f"音频时长过短: {duration:.2f}秒，至少需要1秒"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证失败: {str(e)}"
    
    @classmethod
    def validate_audio_data(cls, audio: np.ndarray, sample_rate: int) -> Tuple[bool, str]:
        """
        验证音频数据格式
        
        Args:
            audio: 音频数据（numpy数组）
            sample_rate: 采样率
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 检查采样率
            if sample_rate != cls.REQUIRED_SAMPLE_RATE:
                return False, f"采样率不匹配: {sample_rate}Hz，需要 {cls.REQUIRED_SAMPLE_RATE}Hz"
            
            # 检查声道数
            if audio.ndim > 1:
                return False, f"音频不是单声道: {audio.ndim}维"
            
            # 检查音频时长（至少1秒）
            duration = len(audio) / sample_rate
            if duration < 1.0:
                return False, f"音频时长过短: {duration:.2f}秒，至少需要1秒"
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证失败: {str(e)}"
    
    @classmethod
    def convert_to_required_format(
        cls,
        input_path: Path,
        output_path: Path
    ) -> Tuple[bool, str]:
        """
        转换音频到要求的格式 (16kHz/16bit/单声道)
        
        支持多种格式：WebM, Ogg, MP3, M4A, MP4, WAV, FLAC等
        优先使用 soundfile，失败则使用 ffmpeg（ffmpeg支持几乎所有音频/视频格式）
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径
            
        Returns:
            (success, message)
        """
        import subprocess
        import shutil
        
        # 方法1: 尝试使用 soundfile 直接读取（支持WAV、FLAC、OGG等）
        try:
            audio, sample_rate = sf.read(str(input_path))
            
            # 转换为单声道
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # 重采样到16kHz
            if sample_rate != cls.REQUIRED_SAMPLE_RATE:
                # 使用线性插值重采样
                duration = len(audio) / sample_rate
                target_len = int(duration * cls.REQUIRED_SAMPLE_RATE)
                x_original = np.linspace(0.0, duration, num=len(audio), endpoint=False)
                x_target = np.linspace(0.0, duration, num=target_len, endpoint=False)
                audio = np.interp(x_target, x_original, audio)
            
            # 确保是float32格式
            audio = audio.astype(np.float32)
            
            # 保存为16bit WAV
            sf.write(
                str(output_path),
                audio,
                cls.REQUIRED_SAMPLE_RATE,
                subtype='PCM_16'
            )
            
            return True, "转换成功"
            
        except Exception as sf_error:
            # 方法2: 使用 ffmpeg 转换（支持所有格式，包括WebM）
            try:
                # 检查 ffmpeg 是否可用
                ffmpeg_path = shutil.which('ffmpeg')
                if not ffmpeg_path:
                    return False, f"soundfile读取失败且未找到ffmpeg: {str(sf_error)}"
                
                # 使用 ffmpeg 转换
                cmd = [
                    ffmpeg_path,
                    '-i', str(input_path),
                    '-ar', str(cls.REQUIRED_SAMPLE_RATE),  # 采样率
                    '-ac', str(cls.REQUIRED_CHANNELS),     # 声道数
                    '-sample_fmt', 's16',                   # 16bit
                    '-y',                                    # 覆盖输出文件
                    str(output_path)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return True, "转换成功（使用ffmpeg）"
                else:
                    return False, f"ffmpeg转换失败: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                return False, "转换超时（超过30秒）"
            except Exception as ffmpeg_error:
                return False, f"所有转换方法都失败: soundfile={str(sf_error)}, ffmpeg={str(ffmpeg_error)}"

