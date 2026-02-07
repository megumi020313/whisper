"""Speaker-ID convenience wrapper used by the runtime."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# 过滤警告信息
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

from speechbrain.pretrained import EncoderClassifier

# 导入核心模块（使用绝对路径）
from backend.core.validators import AudioValidator
from backend.core.exceptions import ModelLoadError, ModelInferenceError, AudioFormatError


_REQUIRED_ENCODER_FILES = {
    "hyperparams.yaml",
    "classifier.ckpt",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
}


def _resolve_encoder_root(override: Optional[str | Path]) -> Path:
    candidates = []
    if override is not None:
        candidates.append(Path(override).expanduser())
    base = Path(__file__).resolve().parent
    candidates.append(base / "model" / "speechbrain")
    candidates.append(base.parent / "training" / "model" / "speechbrain")
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if not resolved.exists():
            continue
        missing = [name for name in _REQUIRED_ENCODER_FILES if not (resolved / name).exists()]
        if not missing:
            return resolved
    raise FileNotFoundError(
        "Unable to locate SpeechBrain encoder assets. Checked: " + ", ".join(checked)
    )


class SpeakerIDWrapper:
    def __init__(
        self,
        model_path: str,
        sample_rate: int,
        num_mfcc: int = 13,
        multi_speaker: bool = True,
        encoder_root: Optional[str | Path] = None,
    ) -> None:
        # 验证采样率
        AudioValidator.validate_sample_rate(sample_rate)
        
        # 验证num_mfcc
        if num_mfcc <= 0:
            raise ValueError(f"num_mfcc must be positive, got {num_mfcc}")
        
        self.sample_rate = sample_rate
        self.num_mfcc = num_mfcc
        self.multi_speaker = multi_speaker
        self._model_path = Path(model_path)
        
        try:
            self._encoder_root = _resolve_encoder_root(encoder_root)
        except FileNotFoundError as e:
            raise ModelLoadError("Failed to locate encoder assets", cause=e)
        
        # Cache enrolled embeddings before loading the speaker dataset
        self._enrolled_embeddings: dict[str, np.ndarray] = {}
        
        try:
            # 先构建encoder（此时会尝试加载微调权重）
            self._sid = self._build_engine()
            # 然后加载说话人模板
            self.load_model(self._model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize Speaker-ID model",
                               details={"model_path": str(model_path)},
                               cause=e)

    def _build_engine(self) -> EncoderClassifier:
        try:
            # Use speechbrain's pretrained ECAPA-TDNN model for speaker verification
            encoder = EncoderClassifier.from_hparams(
                source=str(self._encoder_root),
                savedir=str(self._encoder_root),
                run_opts={"device": "cpu"}  # Use CPU for deployment on Raspberry Pi
            )
            
            # 尝试加载微调后的权重（如果存在）
            # 微调权重通常保存在模型目录的父目录中（训练输出目录）
            self._load_fine_tuned_weights(encoder)
            
            return encoder
        except Exception as e:
            raise ModelLoadError("Failed to build SpeechBrain encoder",
                               details={"encoder_root": str(self._encoder_root)},
                               cause=e)
    
    def _load_fine_tuned_weights(self, encoder: EncoderClassifier) -> None:
        """
        尝试加载微调后的encoder权重
        
        微调权重文件通常位于模型目录的父目录中：
        - 如果model_path是 speaker_embeddings.json
        - 则微调权重应该在同一个目录下的 fine_tuned_embedding_model.pt
        """
        try:
            # 检查模型目录中是否有微调权重文件
            model_dir = self._model_path.parent if self._model_path.is_file() else self._model_path
            fine_tuned_path = model_dir / "fine_tuned_embedding_model.pt"
            
            if fine_tuned_path.exists():
                # 加载微调后的权重
                state_dict = torch.load(fine_tuned_path, map_location="cpu")
                encoder.mods.embedding_model.load_state_dict(state_dict)
                encoder.eval()  # 设置为评估模式
                print(f"[加载] 已加载微调后的encoder权重: {fine_tuned_path.name}")
            else:
                # 如果没有微调权重，使用原始预训练权重（这是正常的）
                print(f"[提示] 未找到微调权重文件，使用原始预训练encoder")
        except Exception as e:
            # 如果加载失败，继续使用原始预训练权重（不抛出异常）
            print(f"[警告] 加载微调权重失败，使用原始预训练encoder: {e}")

    def load_model(self, model_path: Path | str) -> None:
        # For speechbrain, the model_path is used to load enrolled speaker embeddings
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(f"Speaker model not found: {path}",
                               details={"path": str(path)})
        
        try:
            # Load enrolled speaker embeddings from JSON file
            import json
            self._enrolled_embeddings = {}
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for speaker_id, embedding in data.items():
                    self._enrolled_embeddings[speaker_id] = np.array(embedding)
            
            if not self._enrolled_embeddings:
                raise ModelLoadError("No enrolled speakers found in model file",
                                   details={"path": str(path)})
            
            self._model_path = path
        except json.JSONDecodeError as e:
            raise ModelLoadError("Invalid JSON format in model file",
                               details={"path": str(path)},
                               cause=e)
        except Exception as e:
            raise ModelLoadError("Failed to load speaker embeddings",
                               details={"path": str(path)},
                               cause=e)

    def match(self, frame: bytes | np.ndarray) -> float:
        try:
            if isinstance(frame, bytes):
                if len(frame) == 0:
                    raise AudioFormatError("Empty byte frame")
                audio = np.frombuffer(frame, dtype=np.int16)
            else:
                if frame.size == 0:
                    raise AudioFormatError("Empty numpy array")
                audio = frame.astype(np.int16)
            
            # Convert audio to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # 验证音频格式（不限制时长，允许处理完整音频文件）
            AudioValidator.validate_audio_format(audio, expected_rate=self.sample_rate, max_duration=None)
            
            # Compute embedding for the input audio
            embedding = self._compute_embedding(audio)
            
            # Find the best match among enrolled speakers
            if not self._enrolled_embeddings:
                return 0.0
                
            best_similarity = 0.0
            for speaker_id, enrolled_embedding in self._enrolled_embeddings.items():
                similarity = self._compute_cosine_similarity(embedding, enrolled_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
            
            return best_similarity
        except (AudioFormatError, ModelInferenceError):
            raise
        except Exception as e:
            raise ModelInferenceError("Speaker verification failed",
                                    details={"frame_type": type(frame).__name__},
                                    cause=e)

    def match_multi(self, frame: bytes | np.ndarray) -> tuple[str, float]:
        try:
            if isinstance(frame, bytes):
                if len(frame) == 0:
                    raise AudioFormatError("Empty byte frame")
                audio = np.frombuffer(frame, dtype=np.int16)
            else:
                if frame.size == 0:
                    raise AudioFormatError("Empty numpy array")
                audio = frame.astype(np.int16)
            
            # Convert audio to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # 验证音频格式（不限制时长，允许处理完整音频文件）
            AudioValidator.validate_audio_format(audio, expected_rate=self.sample_rate, max_duration=None)
            
            # Compute embedding for the input audio
            embedding = self._compute_embedding(audio)
            
            # Find the best match among enrolled speakers
            if not self._enrolled_embeddings:
                return "unknown", 0.0
                
            best_speaker = "unknown"
            best_similarity = 0.0
            for speaker_id, enrolled_embedding in self._enrolled_embeddings.items():
                similarity = self._compute_cosine_similarity(embedding, enrolled_embedding)
                if similarity > best_similarity:
                    best_speaker = speaker_id
                    best_similarity = similarity
            
            return best_speaker, best_similarity
        except (AudioFormatError, ModelInferenceError):
            raise
        except Exception as e:
            raise ModelInferenceError("Multi-speaker verification failed",
                                    details={"frame_type": type(frame).__name__},
                                    cause=e)
    
    def _compute_embedding(self, audio: np.ndarray) -> np.ndarray:
        try:
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Ensure sample rate matches the model's expected rate (16000 Hz)
            if self.sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
            
            # Compute embedding using the model
            with torch.no_grad():
                embedding = self._sid.encode_batch(audio_tensor)
                embedding = embedding.squeeze().numpy()
            
            return embedding
        except Exception as e:
            raise ModelInferenceError("Failed to compute embedding",
                                    details={"audio_shape": audio.shape, "sample_rate": self.sample_rate},
                                    cause=e)
    
    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Compute cosine similarity between two embeddings
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def enroll_speaker(self, speaker_id: str, audio: np.ndarray) -> None:
        # Enroll a new speaker by computing their embedding and saving it
        embedding = self._compute_embedding(audio)
        self._enrolled_embeddings[speaker_id] = embedding
    
    def save_enrolled_speakers(self, output_path: str) -> None:
        # Save enrolled speaker embeddings to a JSON file
        import json
        data = {}
        for speaker_id, embedding in self._enrolled_embeddings.items():
            data[speaker_id] = embedding.tolist()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
