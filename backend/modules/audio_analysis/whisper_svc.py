"""Whisper ASR 服务封装

使用 Faster-Whisper 实现高效的语音识别。
"""
from typing import Optional, Union
import numpy as np
from pathlib import Path

from backend.utils.logger import get_logger
from backend.core.config import get_config

logger = get_logger()


class WhisperService:
    """Whisper ASR 服务
    
    使用 Faster-Whisper 进行语音识别，支持中文和英文。
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """初始化 Whisper 服务

        Args:
            model_path: 模型路径，默认从配置文件读取
            device: 设备类型 (cuda/cpu)
            compute_type: 计算类型 (float16/int8/float32)
                - float16: V100 上效率最高
                - int8: 更快但精度略低
                - float32: 最高精度但最慢
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper 未安装，请运行: pip install faster-whisper")
            raise

        # 获取配置
        config = get_config()

        # 确定模型路径
        if model_path is None:
            # 从配置文件读取
            model_path = config.whisper_path

        # 转换为字符串路径（支持 Path 对象）
        if isinstance(model_path, Path):
            model_path_str = str(model_path)
        else:
            model_path_str = str(model_path)

        # 检查模型是否存在
        model_path_obj = Path(model_path_str)
        if not model_path_obj.exists():
            error_msg = f"模型路径不存在: {model_path_str}"
            logger.error(error_msg)
            logger.error("请先下载模型: huggingface-cli download Systran/faster-whisper-large-v3 --local-dir models/asr/faster-whisper-large-v3")
            raise FileNotFoundError(error_msg)

        logger.info(f"正在加载 Whisper 模型: {model_path_str}")
        logger.info(f"设备: {device}, 计算类型: {compute_type}")

        # 加载模型（使用字符串路径）
        self.model = WhisperModel(
            model_path_str,
            device=device,
            compute_type=compute_type
        )
        
        self.device = device
        self.compute_type = compute_type
        
        logger.info("✅ Whisper ASR 模型加载完成")
    
    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: str = "zh",
        beam_size: int = 5,
        vad_filter: bool = False,
        initial_prompt: Optional[str] = None
    ) -> str:
        """转录音频为文本
        
        Args:
            audio: 音频数据
                - numpy array: 16kHz 采样率的音频数组
                - str: 音频文件路径
            language: 语言代码 (zh/en/auto)
            beam_size: Beam search 大小
                - 1: 最快，适合实时流
                - 5: 平衡速度和精度
                - 10: 最高精度但最慢
            vad_filter: 是否使用 VAD 过滤静音
            initial_prompt: 初始提示词，用于引导模型输出风格
                - 例如："以下是普通话的句子。" 可以引导输出简体中文
        
        Returns:
            识别的文本内容
        """
        try:
            # 如果是文件路径，直接传给模型
            if isinstance(audio, str):
                audio_input = audio
            # 如果是 numpy 数组，确保是 float32 类型
            elif isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                audio_input = audio
            else:
                raise ValueError(f"不支持的音频类型: {type(audio)}")
            
            # 执行转录
            segments, info = self.model.transcribe(
                audio_input,
                language=language if language != "auto" else None,
                beam_size=beam_size,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt  # 引导模型输出简体中文
            )
            
            # 合并所有片段的文本
            text = "".join([segment.text for segment in segments])
            
            # 去除首尾空格
            text = text.strip()
            
            logger.debug(f"ASR 识别结果: {text[:50]}..." if len(text) > 50 else f"ASR 识别结果: {text}")
            
            return text
            
        except Exception as e:
            logger.error(f"ASR 转录失败: {e}", exc_info=True)
            return ""
    
    def transcribe_with_timestamps(
        self,
        audio: Union[np.ndarray, str],
        language: str = "zh",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None
    ) -> list[dict]:
        """转录音频并返回带时间戳的片段
        
        Args:
            audio: 音频数据（numpy array 或文件路径）
            language: 语言代码
            beam_size: Beam search 大小
        
        Returns:
            片段列表，每个片段包含:
                - text: 文本内容
                - start: 开始时间（秒）
                - end: 结束时间（秒）
        """
        try:
            # 处理音频输入
            if isinstance(audio, str):
                audio_input = audio
            elif isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                audio_input = audio
            else:
                raise ValueError(f"不支持的音频类型: {type(audio)}")
            
            # 执行转录
            segments, info = self.model.transcribe(
                audio_input,
                language=language if language != "auto" else None,
                beam_size=beam_size,
                initial_prompt=initial_prompt  # 引导模型输出简体中文
            )
            
            # 收集所有片段
            result = []
            for segment in segments:
                text = segment.text.strip()
                result.append({
                    "text": text,
                    "start": segment.start,
                    "end": segment.end
                })
            
            logger.debug(f"ASR 识别到 {len(result)} 个片段")
            
            return result
            
        except Exception as e:
            logger.error(f"ASR 转录失败: {e}", exc_info=True)
            return []
    
    def transcribe_with_word_timestamps(
        self,
        audio: Union[np.ndarray, str],
        language: str = "zh",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None
    ) -> list[dict]:
        """转录音频并返回带词级别时间戳的结果
        
        这是V3.0方案的核心接口，返回每个词的精确时间戳。
        
        Args:
            audio: 音频数据（numpy array 或文件路径）
            language: 语言代码
            beam_size: Beam search 大小
            initial_prompt: 初始提示词
        
        Returns:
            词列表，每个词包含:
                - word: 词语文本
                - start: 开始时间（秒）
                - end: 结束时间（秒）
                - confidence: 置信度（如果可用）
        """
        try:
            # 处理音频输入
            if isinstance(audio, str):
                audio_input = audio
            elif isinstance(audio, np.ndarray):
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                audio_input = audio
            else:
                raise ValueError(f"不支持的音频类型: {type(audio)}")
            
            # 执行转录，启用词级别时间戳
            segments, info = self.model.transcribe(
                audio_input,
                language=language if language != "auto" else None,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
                word_timestamps=True  # 关键：启用词级别时间戳
            )
            
            # 收集所有词
            words = []
            for segment in segments:
                # 检查是否有词级别信息
                if hasattr(segment, 'words') and segment.words:
                    for word_info in segment.words:
                        words.append({
                            "word": word_info.word.strip(),
                            "start": word_info.start,
                            "end": word_info.end,
                            "confidence": getattr(word_info, 'probability', 1.0)
                        })
                else:
                    # 如果没有词级别信息，使用片段级别
                    logger.warning("未获取到词级别时间戳，使用片段级别")
                    words.append({
                        "word": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": 1.0
                    })
            
            logger.debug(f"ASR 识别到 {len(words)} 个词")
            
            return words
            
        except Exception as e:
            logger.error(f"ASR 词级别转录失败: {e}", exc_info=True)
            return []
    
    def get_info(self) -> dict:
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "model_loaded": self.model is not None
        }

