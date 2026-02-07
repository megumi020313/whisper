"""FunASR ASR 服务封装

使用 FunASR (Paraformer) 实现语音识别，并输出与既有流水线兼容的结果格式。
"""
from typing import Optional, Union, List, Dict
from pathlib import Path
import numpy as np
import jieba
import re

from backend.utils.logger import get_logger
from backend.core.config import get_config

logger = get_logger()

_PUNCTUATION = set(
    "，。！？；：、“”‘’（）()[]{}《》<>、,.!?;:\"'…—-～·"
)


class FunASRService:
    """FunASR ASR 服务

    使用 FunASR (Paraformer) 进行语音识别，输出字符级时间戳并适配为词级结果。
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        hotword: Optional[str] = None,
        batch_size_s: Optional[int] = None
    ):
        """初始化 FunASR 服务

        Args:
            model_path: 模型路径，默认从配置文件读取
            device: 设备类型 (cuda/cpu)
            hotword: 热词（可选）
            batch_size_s: 单次解码的最大时长（秒）
        """
        try:
            from funasr import AutoModel
        except ImportError:
            logger.error("funasr 未安装，请运行: pip install funasr modelscope")
            raise

        config = get_config()

        # 从配置读取设备信息（优先使用 device.asr_model，否则使用 asr.device）
        if device == "cuda":
            asr_device = getattr(config, 'asr_model_device', None) or getattr(config, 'asr_device', 'cpu')
            device = "cuda" if str(asr_device).startswith('cuda') else "cpu"

        # 模型路径
        if model_path is None:
            model_path = getattr(config, 'funasr_model_path', None)

        if model_path is None:
            raise ValueError("未配置 FunASR 模型路径，请在 config/model_config.yaml 中设置 model.funasr_model_path")

        model_path_str = str(model_path)
        model_path_obj = Path(model_path_str)
        if not model_path_obj.exists():
            error_msg = f"模型路径不存在: {model_path_str}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.device = device
        self.model_path = model_path_str
        self.hotword = hotword or getattr(config, 'asr_hotword', None)
        self.batch_size_s = batch_size_s or getattr(config, 'asr_batch_size_s', 300)
        self.vad_model = getattr(config, 'funasr_vad_model_path', None) or "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.punc_model = getattr(config, 'funasr_punc_model_path', None) or "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

        logger.info(f"正在加载 FunASR 模型: {self.model_path}")
        logger.info(f"设备: {self.device}, batch_size_s: {self.batch_size_s}")

        self.model = AutoModel(
            model=self.model_path,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            device=self.device,
            timestamp_prediction=True,
            disable_update=True
        )

        logger.info("✅ FunASR ASR 模型加载完成")

    def _prepare_audio_input(self, audio: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        if isinstance(audio, str):
            return audio
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio
        raise ValueError(f"不支持的音频类型: {type(audio)}")

    def _generate(self, audio_input: Union[np.ndarray, str]) -> List[Dict]:
        return self.model.generate(
            input=audio_input,
            batch_size_s=self.batch_size_s,
            hotword=self.hotword
        ) or []

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
            audio: 音频数据（numpy array 或文件路径）
            language: 语言代码（保留参数，FunASR不使用）
            beam_size: Beam size（保留参数，FunASR不使用）
            vad_filter: 是否使用VAD过滤（保留参数，FunASR不使用）
            initial_prompt: 初始提示词（保留参数，FunASR不使用）
        """
        try:
            audio_input = self._prepare_audio_input(audio)
            results = self._generate(audio_input)
            text = "".join([item.get('text', '') for item in results]).strip()
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

        Returns:
            片段列表，每个片段包含 text/start/end
        """
        try:
            audio_input = self._prepare_audio_input(audio)
            results = self._generate(audio_input)

            segments: list[dict] = []
            for item in results:
                raw_text = item.get('text') or ''
                timestamps = item.get('timestamp') or []
                if not raw_text or not timestamps:
                    continue

                normalized_text, _ = self._normalize_text_and_timestamps(raw_text, timestamps)
                text = normalized_text.strip() or raw_text.strip()

                start_s = timestamps[0][0] / 1000.0
                end_s = timestamps[-1][1] / 1000.0
                segments.append({
                    "text": text,
                    "start": start_s,
                    "end": end_s
                })

            logger.debug(f"ASR 识别到 {len(segments)} 个片段")
            return segments
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

        Returns:
            词列表，每个词包含 word/start/end/confidence
        """
        try:
            audio_input = self._prepare_audio_input(audio)
            results = self._generate(audio_input)

            words: list[dict] = []
            for item in results:
                raw_text = item.get('text', '') or ''
                timestamps = item.get('timestamp') or []
                if not raw_text or not timestamps:
                    continue

                normalized_text, char_map = self._normalize_text_and_timestamps(raw_text, timestamps)
                if not normalized_text:
                    continue

                tokens = list(jieba.tokenize(normalized_text))

                for word_text, start_idx, end_idx in tokens:
                    word_text = word_text.strip()
                    if not word_text:
                        continue
                    if self._is_punctuation(word_text):
                        if words:
                            words[-1]["word"] += word_text
                        continue

                    valid_ts = []
                    for char_idx in range(start_idx, end_idx):
                        if char_idx < len(char_map) and char_map[char_idx]["ts"] is not None:
                            valid_ts.append(char_map[char_idx]["ts"])

                    if valid_ts:
                        start_ms = valid_ts[0][0]
                        end_ms = valid_ts[-1][1]
                        words.append({
                            "word": word_text,
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "confidence": 0.99
                        })

            words = self._merge_particles(words)

            logger.debug(f"ASR 识别到 {len(words)} 个词")
            return words
        except Exception as e:
            logger.error(f"ASR 词级别转录失败: {e}", exc_info=True)
            return []

    def get_info(self) -> dict:
        """获取模型信息"""
        return {
            "device": self.device,
            "model_path": self.model_path,
            "model_loaded": self.model is not None
        }

    def _normalize_text_and_timestamps(
        self,
        text: str,
        timestamps: List[List[int]]
    ) -> tuple[str, List[Dict[str, Optional[List[int]]]]]:
        """去除空白并对齐时间戳（时间戳通常不包含标点）"""
        aligned: List[Dict[str, Optional[List[int]]]] = []
        normalized_chars: List[str] = []
        ts_idx = 0

        for char in text:
            if char.isspace():
                continue

            is_punc = self._is_punctuation(char)
            ts = None
            if not is_punc and ts_idx < len(timestamps):
                ts = timestamps[ts_idx]
                ts_idx += 1

            aligned.append({"char": char, "ts": ts})
            normalized_chars.append(char)

        return "".join(normalized_chars), aligned

    def _is_punctuation(self, text: str) -> bool:
        """判断文本是否仅由标点符号组成"""
        return re.match(r'^[\s\W_]+$', text) is not None

    def _merge_particles(self, words: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """将短语气词合并到前一个词，减少边界漂移"""
        particles = {"了", "啊", "呀", "嘛", "吧", "呢", "哎", "呃", "哦", "哈"}
        merged: List[Dict[str, float]] = []

        for word in words:
            if not merged:
                merged.append(word)
                continue

            prev = merged[-1]
            gap = word["start"] - prev["end"]
            duration = word["end"] - word["start"]

            if word["word"] in particles and gap <= 0.25 and duration <= 0.35:
                prev["word"] += word["word"]
                prev["end"] = max(prev["end"], word["end"])
                prev["confidence"] = min(prev.get("confidence", 0.99), word.get("confidence", 0.99))
                continue

            merged.append(word)

        return merged
