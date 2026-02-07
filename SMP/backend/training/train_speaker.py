"""Train or adapt a Speaker Recognition model using SpeechBrain with optional pretrained workflows."""
from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import random
import shutil
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Suppress noisy warnings from third-party dependencies that do not impact training outcomes.
warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\.pretrained' was deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Requested Pretrainer collection using symlinks on Windows.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"'onnxscript\.values\.Op\.param_schemas' is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"'onnxscript\.values\.OnnxFunction\.param_schemas' is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Constant folding - Only steps=1 can be constant folded.*",
    category=UserWarning,
)

try:  # Torch raises TracerWarning during ONNX export even when the results are unused.
    from torch.jit import TracerWarning

    warnings.filterwarnings("ignore", category=TracerWarning)
except Exception:  # pragma: no cover - optional dependency path
    pass


from speechbrain.pretrained import EncoderClassifier


def _patch_speechbrain_stft() -> None:
    """Force SpeechBrain STFT to emit real tensors for ONNX export."""

    from speechbrain.processing import features as sb_features

    forward = getattr(sb_features.STFT, "forward", None)
    if forward is None or getattr(forward, "_onnx_safe", False):
        return

    def _forward(self, x):  # type: ignore[override]
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(device=x.device, dtype=x.dtype),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=False,
        )

        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            stft = stft.transpose(2, 1)

        return stft

    _forward._onnx_safe = True  # type: ignore[attr-defined]
    sb_features.STFT.forward = _forward


_patch_speechbrain_stft()


SPEECHBRAIN_DEFAULT_DIR = Path(__file__).resolve().parent / "model" / "speechbrain"
REQUIRED_ENCODER_FILES = {
    "hyperparams.yaml",
    "classifier.ckpt",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
}


class TrainingResult(NamedTuple):
    """Container describing the outcome of the fine-tuning stage."""

    encoder: EncoderClassifier
    classifier_head: Optional[nn.Module]
    history: List[Dict[str, float]]


def _ensure_windows_symlink_fallback() -> None:
    """Monkey-patch pathlib symlink to gracefully fallback to copy on Windows.

    SpeechBrain relies on pathlib.Path.symlink_to when collecting checkpoint
    assets. On Windows non-admin accounts rarely have the SeCreateSymbolicLink
    privilege which raises WinError 1314. By patching the method to copy the
    source file instead we keep the workflow functional without requiring
    system-wide elevation.
    """

    if getattr(_ensure_windows_symlink_fallback, "_patched", False):  # type: ignore[attr-defined]
        return

    windows_module = __import__("pathlib")
    target_cls = getattr(windows_module, "WindowsPath", None)

    if target_cls is None or not hasattr(target_cls, "symlink_to"):
        _ensure_windows_symlink_fallback._patched = True  # type: ignore[attr-defined]
        return

    original = target_cls.symlink_to

    def _patched(self, target, target_is_directory=False):  # type: ignore[override]
        try:
            return original(self, target, target_is_directory=target_is_directory)
        except OSError as exc:  # pragma: no cover - platform specific
            if getattr(exc, "winerror", None) != 1314 and exc.errno not in (errno.EPERM, errno.EACCES):
                raise
        
            # 处理SpeechBrain的路径解析问题
            target_path = Path(target)
            
            # 尝试多种可能的路径解析策略
            possible_paths = []
            
            # 1. 当前路径（原始尝试）
            possible_paths.append(target_path)
            
            # 2. 相对于模型目录的路径
            model_dir = Path(__file__).resolve().parent / "model" / "speechbrain"
            possible_paths.append(model_dir / target)
            
            # 3. 相对于当前工作目录的路径
            possible_paths.append(Path.cwd() / target)
            
            # 4. 相对于脚本所在目录的路径
            script_dir = Path(__file__).resolve().parent
            possible_paths.append(script_dir / target)
            
            # 寻找第一个存在的路径
            actual_target_path = None
            for path in possible_paths:
                if path.exists():
                    actual_target_path = path
                    break
            
            # 如果仍然找不到，尝试直接查找已知文件名
            if actual_target_path is None:
                # 针对常见的SpeechBrain文件进行特殊处理
                known_files = ["label_encoder.txt", "hyperparams.yaml"]
                for filename in known_files:
                    if str(target).endswith(filename):
                        path_in_model = model_dir / filename
                        if path_in_model.exists():
                            actual_target_path = path_in_model
                            break
            
            if actual_target_path is None:
                # 最后尝试：直接创建目标文件（如果是常见的SpeechBrain文件）
                # 例如label_encoder.txt通常是一个简单的文本文件
                if str(target).endswith("label_encoder.txt"):
                    # 创建一个空的label_encoder.txt文件
                    actual_target_path = Path(target)
                    actual_target_path.parent.mkdir(parents=True, exist_ok=True)
                    actual_target_path.touch()
            
            if actual_target_path is None:
                # 所有尝试都失败，重新抛出异常
                raise FileNotFoundError(f"Could not find target file: {target}") from exc
            
            if target_is_directory:
                shutil.copytree(actual_target_path, self, dirs_exist_ok=True)
            else:
                self.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(actual_target_path, self)

    target_cls.symlink_to = _patched  # type: ignore[assignment]
    _ensure_windows_symlink_fallback._patched = True  # type: ignore[attr-defined]


class SpeakerDataset(Dataset):
    """Dataset that loads speaker waveforms with optional augmentation."""

    def __init__(
        self,
        samples: List[str],
        sample_rate: int,
        augment: bool = False,
        noise_level: float = 0.05,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        volume_range: Tuple[float, float] = (0.8, 1.2),
        time_shift_prob: float = 0.3,
        specaugment: bool = False,
        specaugment_time_mask: int = 27,
        specaugment_freq_mask: int = 12,
        reverb_prob: float = 0.0,
        reverb_room_scale: float = 0.3,
        speaker_to_idx: Optional[Dict[str, int]] = None,
        negative_samples: Optional[List[str]] = None,
        multi_speaker_mix_prob: float = 0.3,
        multi_speaker_mix_ratio: Tuple[float, float] = (0.6, 0.8),
        target_employee_prefix: str = "emp_",
    ) -> None:
        self.samples = list(samples)
        self.sample_rate = sample_rate
        self.augment = augment
        self.noise_level = noise_level
        self.speed_range = speed_range
        self.volume_range = volume_range
        self.time_shift_prob = time_shift_prob
        self.specaugment = specaugment
        self.specaugment_time_mask = specaugment_time_mask
        self.specaugment_freq_mask = specaugment_freq_mask
        self.reverb_prob = reverb_prob
        self.reverb_room_scale = reverb_room_scale
        self.speaker_ids: List[str] = []
        self.speaker_to_idx: Dict[str, int] = dict(speaker_to_idx or {})
        
        # 多人对话混合相关参数
        self.negative_samples = list(negative_samples) if negative_samples else []
        self.multi_speaker_mix_prob = multi_speaker_mix_prob
        self.multi_speaker_mix_ratio = multi_speaker_mix_ratio
        self.target_employee_prefix = target_employee_prefix

        for wav in self.samples:
            wav_path = Path(wav)
            speaker_id = _extract_speaker_id(wav_path)
            if speaker_id is None:
                raise ValueError(f"无法从文件名提取说话人ID: {wav}")
            if speaker_id not in self.speaker_to_idx:
                self.speaker_to_idx[speaker_id] = len(self.speaker_to_idx)
            self.speaker_ids.append(speaker_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        wav_path = Path(self.samples[index])
        speaker_id = self.speaker_ids[index]
        label = self.speaker_to_idx[speaker_id]

        waveform = _prepare_waveform(wav_path, self.sample_rate)
        if self.augment:
            waveform = self._apply_augmentation(waveform)

        return torch.from_numpy(waveform), label, speaker_id

    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        augmented = np.copy(waveform)

        # 加性噪声增强（50%概率）
        if random.random() < 0.5:
            noise = np.random.normal(0.0, self.noise_level, size=augmented.shape[0])
            augmented = np.clip(augmented + noise.astype(np.float32), -1.0, 1.0)

        # 语速调整增强（50%概率）
        if random.random() < 0.5:
            speed_factor = random.uniform(*self.speed_range)
            duration = augmented.shape[0] / self.sample_rate
            if speed_factor > 0 and duration > 0:
                target_len = max(1, int(duration * speed_factor * self.sample_rate))
                x_original = np.linspace(0.0, duration, num=augmented.shape[0], endpoint=False)
                x_target = np.linspace(0.0, duration, num=target_len, endpoint=False)
                augmented = np.interp(x_target, x_original, augmented).astype(np.float32)

        # 音量缩放增强（50%概率）
        if random.random() < 0.5:
            volume_factor = random.uniform(*self.volume_range)
            augmented = np.clip(augmented * volume_factor, -1.0, 1.0)

        # 时间偏移增强（可配置概率）
        if random.random() < self.time_shift_prob:
            max_shift = int(self.sample_rate * 0.1)  # 最大偏移0.1秒
            if max_shift > 0 and augmented.shape[0] > max_shift:
                shift = random.randint(-max_shift, max_shift)
                if shift > 0:
                    augmented = np.concatenate([augmented[shift:], np.zeros(shift, dtype=augmented.dtype)])
                elif shift < 0:
                    augmented = np.concatenate([np.zeros(-shift, dtype=augmented.dtype), augmented[:shift]])

        # SpecAugment增强（频谱域掩码）
        if self.specaugment and random.random() < 0.5:
            augmented = self._apply_specaugment(augmented)
        
        # 混响效果增强
        if random.random() < self.reverb_prob:
            augmented = self._apply_reverb(augmented)
        
        # 多人对话混合增强（在最后应用，模拟真实多人对话场景）
        if self.augment and self.negative_samples and random.random() < self.multi_speaker_mix_prob:
            augmented = self._apply_multi_speaker_mix(augmented)

        return augmented.astype(np.float32)
    
    def _apply_specaugment(self, waveform: np.ndarray) -> np.ndarray:
        """Apply SpecAugment: time and frequency masking in spectrogram domain."""
        try:
            # 转换为torch tensor进行STFT
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            
            # 计算STFT
            n_fft = 512
            hop_length = 160
            win_length = 400
            window = torch.hann_window(win_length)
            
            stft = torch.stft(
                waveform_tensor,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )
            
            # 转换为幅度谱
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # 时间掩码
            if self.specaugment_time_mask > 0:
                time_mask_param = min(self.specaugment_time_mask, magnitude.shape[-1] // 2)
                if time_mask_param > 0:
                    t0 = random.randint(0, magnitude.shape[-1] - time_mask_param)
                    magnitude[:, :, t0:t0+time_mask_param] = 0.0
            
            # 频率掩码
            if self.specaugment_freq_mask > 0:
                freq_mask_param = min(self.specaugment_freq_mask, magnitude.shape[-2] // 2)
                if freq_mask_param > 0:
                    f0 = random.randint(0, magnitude.shape[-2] - freq_mask_param)
                    magnitude[:, f0:f0+freq_mask_param, :] = 0.0
            
            # 重构复数STFT
            stft_augmented = magnitude * torch.exp(1j * phase)
            
            # 逆STFT
            waveform_augmented = torch.istft(
                stft_augmented,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                length=waveform_tensor.shape[-1],
            )
            
            return waveform_augmented.squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            # 如果SpecAugment失败，返回原始波形
            return waveform
    
    def _apply_reverb(self, waveform: np.ndarray) -> np.ndarray:
        """Apply reverb effect using simple impulse response simulation."""
        try:
            # 简单的混响模拟：使用延迟和衰减
            reverb_length = int(self.sample_rate * 0.3)  # 0.3秒混响
            if reverb_length > len(waveform):
                return waveform
            
            # 创建简单的混响冲激响应（指数衰减）
            delay_samples = int(self.sample_rate * 0.05)  # 50ms延迟
            decay = 0.3 * self.reverb_room_scale  # 衰减系数
            
            reverb_signal = np.zeros_like(waveform)
            
            # 添加多个延迟回声
            for i in range(3):
                delay = delay_samples * (i + 1)
                if delay < len(waveform):
                    echo = np.pad(waveform[:-delay], (delay, 0), mode='constant')
                    reverb_signal += echo * (decay ** (i + 1))
            
            # 混合原始信号和混响
            augmented = waveform + reverb_signal * 0.3
            return np.clip(augmented, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            # 如果混响失败，返回原始波形
            return waveform
    
    def _apply_multi_speaker_mix(self, waveform: np.ndarray) -> np.ndarray:
        """
        时间交替混合增强：使用时间分段混合，更真实地模拟多人对话场景
        
        回退到最优模型配置（20251214_140340）：
        1. 使用0.5秒片段，保持足够的声纹特征上下文
        2. 保持原始混合条件逻辑
        3. 非目标员工说话片段中目标员工占比：10%-30%
        
        Args:
            waveform: 目标员工的音频波形
            
        Returns:
            混合后的音频波形
        """
        if not self.negative_samples:
            return waveform
        
        try:
            # 随机选择一个非目标员工语音样本
            negative_sample_path = random.choice(self.negative_samples)
            negative_waveform = _prepare_waveform(Path(negative_sample_path), self.sample_rate)
            
            # 调整负样本长度以匹配目标音频
            target_len = len(waveform)
            if len(negative_waveform) > target_len:
                # 如果负样本更长，随机选择一段
                start_idx = random.randint(0, len(negative_waveform) - target_len)
                negative_waveform = negative_waveform[start_idx:start_idx + target_len]
            elif len(negative_waveform) < target_len:
                # 如果负样本更短，循环填充或静音填充
                if len(negative_waveform) > 0:
                    # 循环填充
                    repeat_times = (target_len // len(negative_waveform)) + 1
                    negative_waveform = np.tile(negative_waveform, repeat_times)[:target_len]
                else:
                    # 如果负样本为空，使用静音
                    negative_waveform = np.zeros(target_len, dtype=waveform.dtype)
            
            # 时间交替混合：将音频分成多个片段，每个片段随机选择混合方式
            # 回退到最优模型配置：使用0.5秒片段，保持足够的声纹特征上下文
            segment_length = int(self.sample_rate * 0.5)  # 回退到0.5秒一段
            num_segments = (target_len // segment_length) + 1
            
            mixed = np.zeros_like(waveform)
            
            # 整体目标员工占比（用于控制整体混合强度）
            overall_target_ratio = random.uniform(*self.multi_speaker_mix_ratio)
            
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, target_len)
                if start >= target_len:
                    break
                
                # 每个片段随机选择混合方式
                # 回退到最优模型配置：保持原始混合条件逻辑
                if random.random() < overall_target_ratio:
                    # 主要使用目标员工，少量负样本（80%-90%目标员工）
                    segment_ratio = random.uniform(*self.multi_speaker_mix_ratio)
                    mixed[start:end] = (
                        waveform[start:end] * segment_ratio + 
                        negative_waveform[start:end] * (1 - segment_ratio)
                    )
                else:
                    # 主要使用负样本，少量目标员工（回退到10%-30%，保持目标员工特征可学习性）
                    segment_ratio = random.uniform(0.1, 0.3)
                    mixed[start:end] = (
                        waveform[start:end] * segment_ratio + 
                        negative_waveform[start:end] * (1 - segment_ratio)
                    )
            
            # 归一化，避免削波
            max_val = np.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val
            
            return np.clip(mixed, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            # 如果混合失败，返回原始波形
            return waveform


def _safe_torch_save(obj: Any, destination: Path) -> None:
    """Save torch objects while avoiding Windows Unicode path issues."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        torch.save(obj, handle)


def _collate_fn(batch: List[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    waveforms, labels, speaker_ids = zip(*batch)
    lengths = torch.tensor([waveform.shape[0] for waveform in waveforms], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    max_len = torch.max(lengths).float().clamp(min=1.0)
    wav_lens = lengths.float() / max_len
    return padded, wav_lens, torch.tensor(labels, dtype=torch.long), list(speaker_ids)


class ContrastiveLoss(nn.Module):
    """Simple contrastive loss operating on cosine similarities."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(embeddings, embeddings.t())
        mask = torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, 0.0)

        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        positives = labels_matrix.float()
        negatives = 1.0 - positives

        positive_term = positives * (1.0 - similarity) ** 2
        negative_term = negatives * F.relu(self.margin - similarity) ** 2

        # Avoid division by zero in degenerate batches
        valid_terms = positives.sum() + negatives.sum()
        if valid_terms == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        loss = (positive_term.sum() + negative_term.sum()) / valid_terms
        return loss


class TripletLoss(nn.Module):
    """Hard-triplet loss that searches hardest positive/negative per anchor."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeddings = F.normalize(embeddings, dim=1)
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        total_loss = embeddings.new_tensor(0.0)
        triplet_count = 0

        for index in range(embeddings.size(0)):
            positive_mask = labels == labels[index]
            positive_mask[index] = False
            negative_mask = labels != labels[index]

            if not torch.any(positive_mask) or not torch.any(negative_mask):
                continue

            hardest_positive = distance_matrix[index][positive_mask].max()
            hardest_negative = distance_matrix[index][negative_mask].min()
            loss = F.relu(hardest_positive - hardest_negative + self.margin)
            total_loss = total_loss + loss
            triplet_count += 1

        if triplet_count == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return total_loss / triplet_count


def _pairwise_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int]:
    """Estimate accuracy by nearest-neighbour matching inside a batch."""

    if embeddings.size(0) < 2:
        return 0.0, 0

    normalized = F.normalize(embeddings, dim=1)
    similarity = torch.matmul(normalized, normalized.t())
    similarity.fill_diagonal_(-1.0)
    predicted_indices = similarity.argmax(dim=1)
    predicted_labels = labels[predicted_indices]
    correct = (predicted_labels == labels).sum().item()
    return float(correct), labels.size(0)


def _get_loss_function(config: Dict[str, Any]) -> Tuple[str, nn.Module]:
    loss_type = (config.get("loss_function") or "contrastive").lower()
    margin = float(config.get("margin", 0.2))
    if loss_type == "softmax":
        return loss_type, nn.CrossEntropyLoss()
    if loss_type == "triplet":
        return loss_type, TripletLoss(margin=margin)
    return "contrastive", ContrastiveLoss(margin=margin)


def _get_optimizer(params: Iterable[torch.nn.Parameter], config: Dict[str, Any]) -> torch.optim.Optimizer:
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_name = str(config.get("optimizer", "adam")).lower()

    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


def _get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    scheduler_type = str(config.get("scheduler", "none")).lower()
    warmup_epochs = int(config.get("warmup_epochs", 0))
    
    # 如果有预热，先创建预热调度器
    if warmup_epochs > 0 and epochs > warmup_epochs:
        # 使用线性预热
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,  # 从10%的学习率开始
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # 主调度器
        if scheduler_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs)
            )
        elif scheduler_type == "step":
            step_size = int(config.get("scheduler_step_size", 5))
            gamma = float(config.get("scheduler_gamma", 0.1))
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            return warmup_scheduler
        
        # 组合预热和主调度器
        from torch.optim.lr_scheduler import SequentialLR
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        # 无预热，直接使用主调度器
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if scheduler_type == "step":
            step_size = int(config.get("scheduler_step_size", 5))
            gamma = float(config.get("scheduler_gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return None


def _train_epoch(
    encoder: EncoderClassifier,
    dataloader: DataLoader,
    loss_type: str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    classifier_head: Optional[nn.Module] = None,
    grad_clip: Optional[float] = None,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    realtime_monitoring: bool = False,
    monitoring_interval: int = 10,
    epoch: int = 0,
) -> Tuple[float, float]:
    encoder.train()
    if classifier_head is not None:
        classifier_head.train()

    running_loss = 0.0
    running_correct = 0.0
    total_examples = 0
    
    optimizer.zero_grad(set_to_none=True)
    accumulation_counter = 0
    
    # 混合精度训练的scaler
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # 实时监控相关变量
    total_batches = len(dataloader)
    batch_start_time = time.time() if realtime_monitoring else None

    for batch_idx, (waveforms, wav_lens, labels, _) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        wav_lens = wav_lens.to(device)
        labels = labels.to(device)

        if use_amp:
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 直接调用encoder，DataParallel会自动处理多GPU分配
                # encoder可能是DataParallel包装的，也可能是EncoderWrapper包装的
                embeddings = encoder(waveforms, wav_lens)

                if loss_type == "softmax":
                    if classifier_head is None:
                        raise RuntimeError("Softmax loss requires a classifier head.")
                    logits = classifier_head(embeddings)
                    loss = loss_fn(logits, labels)
                    predictions = logits.argmax(dim=1)
                    running_correct += (predictions == labels).sum().item()
                    total_examples += labels.size(0)
                else:
                    loss = loss_fn(embeddings, labels)
                    correct, batch_size = _pairwise_accuracy(embeddings.detach(), labels)
                    running_correct += correct
                    total_examples += batch_size

                # 梯度累积：将损失除以累积步数
                loss = loss / accumulation_steps
            
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
        else:
            # 标准精度训练
            # 直接调用encoder，DataParallel会自动处理多GPU分配
            embeddings = encoder(waveforms, wav_lens)

            if loss_type == "softmax":
                if classifier_head is None:
                    raise RuntimeError("Softmax loss requires a classifier head.")
                logits = classifier_head(embeddings)
                loss = loss_fn(logits, labels)
                predictions = logits.argmax(dim=1)
                running_correct += (predictions == labels).sum().item()
                total_examples += labels.size(0)
            else:
                loss = loss_fn(embeddings, labels)
                correct, batch_size = _pairwise_accuracy(embeddings.detach(), labels)
                running_correct += correct
                total_examples += batch_size

            # 梯度累积：将损失除以累积步数
            loss = loss / accumulation_steps
            loss.backward()
        
        accumulation_counter += 1
        
        # 达到累积步数或最后一个批次时，更新参数
        if accumulation_counter % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                # 混合精度训练的梯度裁剪和优化器更新
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    if classifier_head is not None:
                        torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准精度训练的梯度裁剪和优化器更新
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                    if classifier_head is not None:
                        torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0

        # 记录原始损失（未除以累积步数）
        running_loss += loss.item() * accumulation_steps
        
        # 实时监控：每N个batch显示一次进度
        if realtime_monitoring and (batch_idx + 1) % monitoring_interval == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = (running_correct / total_examples) if total_examples else 0.0
            elapsed_time = time.time() - batch_start_time if batch_start_time else 0
            batch_time = elapsed_time / (batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            eta = batch_time * remaining_batches
            
            progress = ((batch_idx + 1) / total_batches) * 100
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_info = f" | GPU={gpu_memory:.2f}GB"
            
            print(f"  [Batch {batch_idx+1:4d}/{total_batches}] Loss={current_loss:.4f} Acc={current_acc*100:.2f}% | "
                  f"进度={progress:5.1f}% | ETA={eta:.1f}s{gpu_info}", end='\r')
    
    # 实时监控：epoch结束时换行
    if realtime_monitoring:
        print()  # 换行，避免覆盖最后一行

    mean_loss = running_loss / max(1, len(dataloader))
    accuracy = (running_correct / total_examples) if total_examples else 0.0
    return mean_loss, accuracy


@torch.no_grad()
def _evaluate(
    encoder: EncoderClassifier,
    dataloader: Optional[DataLoader],
    loss_type: str,
    loss_fn: nn.Module,
    device: torch.device,
    classifier_head: Optional[nn.Module] = None,
    compute_metrics: bool = False,
    collect_embeddings: bool = False,
) -> Tuple[float, float, Optional[Dict[str, float]], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate model and optionally compute detailed metrics (F1, Precision, Recall).
    
    Returns:
        mean_loss: Average loss
        accuracy: Accuracy score
        metrics: Optional dict with 'f1', 'precision', 'recall' if compute_metrics=True
    """
    if dataloader is None:
        return 0.0, 0.0, None, None

    encoder.eval()
    if classifier_head is not None:
        classifier_head.eval()

    running_loss = 0.0
    running_correct = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []
    all_embeddings = [] if collect_embeddings else None

    for waveforms, wav_lens, labels, _ in dataloader:
        waveforms = waveforms.to(device)
        wav_lens = wav_lens.to(device)
        labels = labels.to(device)

        # 直接调用encoder，DataParallel会自动处理多GPU分配
        embeddings = encoder(waveforms, wav_lens)

        if collect_embeddings:
            all_embeddings.append(embeddings.cpu().numpy())

        if loss_type == "softmax":
            if classifier_head is None:
                raise RuntimeError("Softmax loss requires a classifier head.")
            logits = classifier_head(embeddings)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
            if compute_metrics or collect_embeddings:
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        else:
            loss = loss_fn(embeddings, labels)
            correct, batch_size = _pairwise_accuracy(embeddings, labels)
            running_correct += correct
            total_examples += batch_size
            # For contrastive/triplet loss, we can't easily get per-sample predictions
            # So metrics computation is only available for softmax loss

        running_loss += loss.item()

    mean_loss = running_loss / max(1, len(dataloader))
    accuracy = (running_correct / total_examples) if total_examples else 0.0
    
    metrics = None
    embeddings_data = None
    
    if compute_metrics and all_predictions and loss_type == "softmax":
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            metrics = {
                "f1": float(f1_score(all_labels, all_predictions, average="weighted")),
                "precision": float(precision_score(all_labels, all_predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(all_labels, all_predictions, average="weighted", zero_division=0)),
            }
        except ImportError:
            pass  # sklearn not available, skip metrics
    
    if collect_embeddings and all_embeddings:
        embeddings_data = (
            np.vstack(all_embeddings),
            np.array(all_labels) if all_labels else None,
            np.array(all_predictions) if all_predictions else None,
        )
    
    return mean_loss, accuracy, metrics, embeddings_data


def _split_train_val(
    samples_by_speaker: Dict[str, List[str]],
    validation_ratio: float,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    train_samples: List[str] = []
    val_samples: List[str] = []

    for speaker_id, wavs in samples_by_speaker.items():
        bucket = list(wavs)
        rng.shuffle(bucket)
        val_count = max(1, int(len(bucket) * validation_ratio)) if len(bucket) > 1 else 0
        val_subset = bucket[:val_count]
        train_subset = bucket[val_count:]
        if not train_subset and val_subset:
            train_subset.append(val_subset.pop())
        train_samples.extend(train_subset)
        val_samples.extend(val_subset)

    return train_samples, val_samples


def _get_original_encoder(encoder) -> EncoderClassifier:
    """获取原始 encoder（处理 DataParallel 和 EncoderWrapper 包装）"""
    # 处理DataParallel包装
    if hasattr(encoder, "module"):
        encoder = encoder.module
    # 处理EncoderWrapper包装
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder
    return encoder


def _freeze_encoder_layers(encoder: EncoderClassifier, config: Dict[str, Any]) -> None:
    if not config.get("freeze_layers", True):
        print("[模型] 冻结层已禁用，所有层可训练")
        return

    # 统计总层数
    total_params = sum(1 for _ in encoder.mods.embedding_model.named_parameters())
    
    print(f"[模型] 冻结所有编码器层（共{total_params}个参数）...")
    for param in encoder.mods.embedding_model.parameters():
        param.requires_grad = False

    unfreeze_layers: List[str] = list(config.get("unfreeze_layers", []))
    if not unfreeze_layers:
        print("[模型] 未指定解冻层，所有层保持冻结")
        return

    unfrozen_count = 0
    for name, param in encoder.mods.embedding_model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True
            unfrozen_count += 1
    
    if unfrozen_count == 0:
        print(f"[警告] 未解冻任何层，请检查层名称是否匹配: {unfreeze_layers}")
    else:
        print(f"[模型] 已解冻{unfrozen_count}个参数层: {unfreeze_layers}")
        # 如果没有匹配的层，自动尝试解冻一些常见的顶层
        common_top_layers = ["fc", "classifier", "output", "dense"]
        print(f"Attempting to unfreeze common top layers: {common_top_layers}")
        for name, param in encoder.mods.embedding_model.named_parameters():
            if any(layer in name.lower() for layer in common_top_layers):
                param.requires_grad = True
                unfrozen_count += 1
                print(f"  - Auto-unfroze: {name}")
    
    print(f"Total unfrozen layers: {unfrozen_count}")


def _collect_trainable_parameters(
    encoder: EncoderClassifier,
    classifier_head: Optional[nn.Module],
) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    # 处理 DataParallel 和 EncoderWrapper 包装的情况
    original_encoder = _get_original_encoder(encoder)
    for param in original_encoder.mods.embedding_model.parameters():
        if param.requires_grad:
            params.append(param)
    if classifier_head is not None:
        params.extend(list(classifier_head.parameters()))
    if not params:
        raise RuntimeError("No trainable parameters found. Check freeze configuration.")
    return params


def _train_model(
    encoder: EncoderClassifier,
    train_dataset: SpeakerDataset,
    val_dataset: Optional[SpeakerDataset],
    training_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
    device: torch.device,
    run_dir: Path,
    use_parallel: bool = False,
) -> TrainingResult:
    # 获取logger（如果已初始化）
    logger = logging.getLogger(__name__)
    
    # 解析run_dir为绝对路径，确保所有相对路径都被正确解析
    run_dir = run_dir.resolve()
    # 确保run_dir目录存在
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[训练] 训练模型时使用的run_dir: {run_dir}")
    print(f"训练模型时使用的run_dir: {run_dir}")
    
    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 10))
    grad_clip = training_cfg.get("grad_clip")
    grad_clip_value = float(grad_clip) if grad_clip is not None else None

    loss_type, loss_fn = _get_loss_function(training_cfg)

    # 数据加载配置优化
    num_workers = int(training_cfg.get("num_workers", 4))
    # 如果未配置，根据CPU核心数自动调整
    if num_workers == 4:  # 默认值，尝试自动检测
        try:
            import os
            cpu_count = os.cpu_count() or 4
            # 对于多核CPU，使用约1/3的核心数作为num_workers
            num_workers = min(16, max(4, cpu_count // 3))
        except Exception:
            num_workers = 4
    
    pin_memory = training_cfg.get("pin_memory", torch.cuda.is_available())
    if isinstance(pin_memory, bool):
        use_pin_memory = pin_memory
    else:
        use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,  # 保持worker进程，减少启动开销
        prefetch_factor=4 if num_workers > 0 else 2,  # 预取因子，提高数据加载效率，减少GPU等待时间
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else 2,  # 预取因子，提高数据加载效率
        )

    classifier_head: Optional[nn.Module] = None
    if loss_type == "softmax":
        embedding_dim = int(training_cfg.get("embedding_dim", 192))
        num_classes = len(train_dataset.speaker_to_idx)
        classifier_head = nn.Linear(embedding_dim, num_classes, bias=True).to(device)

    _freeze_encoder_layers(encoder, model_cfg)
    
    # 多GPU并行支持
    # 创建一个包装类，将encode_batch方法包装成forward方法，以便DataParallel能够正确处理
    class EncoderWrapper(nn.Module):
        """包装EncoderClassifier，使其能够与DataParallel配合使用"""
        def __init__(self, encoder: EncoderClassifier):
            super().__init__()
            self.encoder = encoder
        
        def forward(self, waveforms: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
            """将encode_batch包装成forward方法，供DataParallel使用"""
            embeddings = self.encoder.encode_batch(waveforms, wav_lens)
            # 如果embeddings是3维的，压缩第2维
            if embeddings.ndim == 3:
                embeddings = embeddings.squeeze(1)
            return embeddings
        
        def encode_batch(self, waveforms: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
            """保持encode_batch接口，用于非DataParallel场景"""
            return self.forward(waveforms, wav_lens)
    
    if use_parallel and torch.cuda.device_count() > 1:
        gpu_count = torch.cuda.device_count()
        print(f"Using DataParallel with {gpu_count} GPUs")
        # 先包装encoder，然后移动到device，最后应用DataParallel
        encoder = EncoderWrapper(encoder)
    encoder = encoder.to(device)
        encoder = nn.DataParallel(encoder, device_ids=list(range(gpu_count)))
    else:
        encoder = encoder.to(device)
    
    encoder.train()

    optimizer = _get_optimizer(_collect_trainable_parameters(encoder, classifier_head), training_cfg)
    scheduler = _get_scheduler(optimizer, training_cfg, epochs)

    early_stopping = bool(training_cfg.get("early_stopping", True))
    patience_limit = int(training_cfg.get("early_stopping_patience", 3))
    improvement_delta = float(training_cfg.get("early_stopping_delta", 1e-3))

    best_val_loss = float("inf")
    best_composite_score = -float("inf")
    patience_counter = 0
    best_state: Dict[str, Any] = {}
    history: List[Dict[str, float]] = []

    loss_fn = loss_fn.to(device)

    save_best_model = bool(validation_cfg.get("save_best_model", True))
    compute_metrics = bool(validation_cfg.get("compute_metrics", False)) and loss_type == "softmax"
    accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    use_amp = bool(training_cfg.get("mixed_precision", False)) and torch.cuda.is_available()
    
    if use_amp:
        print("启用混合精度训练（FP16）")
    
    # 实时监控配置
    realtime_monitoring = bool(training_cfg.get("realtime_monitoring", False))
    monitoring_interval = int(training_cfg.get("monitoring_interval", 10))  # 每N个batch更新一次

    for epoch in range(1, epochs + 1):
        # 实时监控：记录训练开始时间
        epoch_start_time = time.time()
        batch_times = []
        
        train_loss, train_acc = _train_epoch(
            encoder,
            train_loader,
            loss_type,
            loss_fn,
            optimizer,
            device,
            classifier_head,
            grad_clip_value,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            realtime_monitoring=realtime_monitoring,
            monitoring_interval=monitoring_interval,
            epoch=epoch,
        )

        collect_embeddings = bool(validation_cfg.get("visualize_embeddings", False)) and (epoch == epochs)
        val_loss, val_acc, val_metrics, val_embeddings = _evaluate(
            encoder,
            val_loader,
            loss_type,
            loss_fn,
            device,
            classifier_head,
            compute_metrics=compute_metrics,
            collect_embeddings=collect_embeddings,
        )
        
        # 保存最后一个epoch的嵌入向量用于可视化（将在训练完成后处理）

        if scheduler is not None:
            scheduler.step()

        epoch_history = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc * 100.0),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc * 100.0),
            "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
        }
        
        if val_metrics:
            epoch_history.update({
                "val_f1": float(val_metrics["f1"] * 100.0),
                "val_precision": float(val_metrics["precision"] * 100.0),
                "val_recall": float(val_metrics["recall"] * 100.0),
            })
        
        history.append(epoch_history)

        metrics_str = ""
        if val_metrics:
            metrics_str = f" | F1={val_metrics['f1']*100:.2f}% P={val_metrics['precision']*100:.2f}% R={val_metrics['recall']*100:.2f}%"
        
        # 实时监控输出（同时记录到日志）
        epoch_time = time.time() - epoch_start_time
        log_msg = f"[Epoch {epoch:2d}/{epochs}] Train: Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | Val: Loss={val_loss:.4f} Acc={val_acc*100:.2f}%{metrics_str} | 耗时: {epoch_time:.2f}s"
        logger.info(log_msg)
        
        if realtime_monitoring:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{epochs} | 耗时: {epoch_time:.2f}s")
            print(f"训练: Loss={train_loss:.4f} | Acc={train_acc*100:.2f}%")
            print(f"验证: Loss={val_loss:.4f} | Acc={val_acc*100:.2f}%{metrics_str}")
            print(f"学习率: {optimizer.param_groups[0].get('lr', 0.0):.6f}")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                print(f"GPU内存: {gpu_memory:.2f} GB")
            print(f"{'='*80}\n")
        else:
            print(f"[Epoch {epoch:2d}/{epochs}] Train: Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | Val: Loss={val_loss:.4f} Acc={val_acc*100:.2f}%{metrics_str}")
        
        epoch_history["epoch_time"] = float(epoch_time)

        if val_loader is None:
            continue

        # 多指标早停：如果启用了多指标，使用综合分数
        use_multi_metric_early_stop = bool(validation_cfg.get("multi_metric_early_stop", False))
        if use_multi_metric_early_stop and val_metrics:
            # 计算综合分数（准确率和F1的平均值）
            composite_score = (val_acc + val_metrics.get("f1", 0.0)) / 2.0
            
            if composite_score > best_composite_score + improvement_delta:
                best_composite_score = composite_score
                best_val_loss = val_loss
                patience_counter = 0
                # 获取原始encoder（处理DataParallel和EncoderWrapper包装）
                original_encoder = _get_original_encoder(encoder)
                best_state = {
                    "embedding": original_encoder.mods.embedding_model.state_dict(),
                    "classifier": classifier_head.state_dict() if classifier_head is not None else None,
                }
                if save_best_model:
                    save_path = run_dir / "best_model.pt"
                    print(f"[模型] 保存最佳模型: {save_path.name}")
                    _safe_torch_save(best_state, save_path)
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience_limit:
                    print("Early stopping triggered due to validation plateau (multi-metric).")
                    break
        else:
            # 单指标早停：基于验证损失
            if val_loss < best_val_loss - improvement_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # 获取原始encoder（处理DataParallel和EncoderWrapper包装）
                original_encoder = _get_original_encoder(encoder)
                best_state = {
                    "embedding": original_encoder.mods.embedding_model.state_dict(),
                    "classifier": classifier_head.state_dict() if classifier_head is not None else None,
                }
                if save_best_model:
                    save_path = run_dir / "best_model.pt"
                    print(f"[模型] 保存最佳模型: {save_path.name}")
                    _safe_torch_save(best_state, save_path)
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience_limit:
                    print(f"[早停] 验证指标在{patience_limit}个epoch内未改善，停止训练")
                    break

    # 生成混淆矩阵和TSNE可视化（如果启用）
    collect_embeddings_final = bool(validation_cfg.get("visualize_embeddings", False))
    val_embeddings_final = None
    if collect_embeddings_final and val_loader is not None:
        _, _, _, val_embeddings_final = _evaluate(
            encoder, val_loader, loss_type, loss_fn, device, classifier_head,
            compute_metrics=compute_metrics, collect_embeddings=True
        )
    
    if compute_metrics and val_loader is not None and loss_type == "softmax":
        _generate_confusion_matrix_from_eval(encoder, val_loader, loss_type, loss_fn, device, classifier_head, run_dir, train_dataset.speaker_to_idx)
    
    # 生成嵌入向量可视化（如果启用）
    if collect_embeddings_final and val_embeddings_final is not None:
        _visualize_embeddings(val_embeddings_final, run_dir, train_dataset.speaker_to_idx)
    
    # 生成性能分析报告
    _generate_performance_report(history, run_dir, training_cfg, validation_cfg)

    if best_state:
        # 获取原始模型（处理DataParallel和EncoderWrapper包装）
        original_encoder = _get_original_encoder(encoder)
        original_encoder.mods.embedding_model.load_state_dict(best_state["embedding"])
        if classifier_head is not None and best_state.get("classifier") is not None:
            classifier_head.load_state_dict(best_state["classifier"])

    return TrainingResult(encoder=encoder, classifier_head=classifier_head, history=history)


def _save_fine_tuned_weights(result: TrainingResult, run_dir: Path) -> None:
    # 获取原始模型（处理DataParallel和EncoderWrapper包装）
    original_encoder = _get_original_encoder(result.encoder)
    encoder_state = original_encoder.mods.embedding_model.state_dict()
    _safe_torch_save(encoder_state, run_dir / "fine_tuned_embedding_model.pt")
    if result.classifier_head is not None:
        _safe_torch_save(result.classifier_head.state_dict(), run_dir / "classification_head.pt")


def _visualize_training_history(history: List[Dict[str, float]], output_dir: Path) -> None:
    """Generate training curves visualization from training history."""
    if not HAS_MATPLOTLIB or not history:
        return
    
    try:
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        train_acc = [h["train_accuracy"] for h in history]
        val_acc = [h["val_accuracy"] for h in history]
        lr = [h["learning_rate"] for h in history]
        
        # 检查是否有额外指标
        has_metrics = "val_f1" in history[0] if history else False
        
        # 创建综合图表
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Loss曲线
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy曲线
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(epochs, lr, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Learning Rate', fontsize=11)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss和Accuracy对比
        ax4 = plt.subplot(2, 3, 4)
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        line2 = ax4_twin.plot(epochs, train_acc, 'r-', label='Train Acc', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss', color='b', fontsize=11)
        ax4_twin.set_ylabel('Accuracy (%)', color='r', fontsize=11)
        ax4.set_title('Loss vs Accuracy', fontsize=12, fontweight='bold')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. 验证指标（如果有）
        if has_metrics:
            ax5 = plt.subplot(2, 3, 5)
            val_f1 = [h.get("val_f1", 0) for h in history]
            val_prec = [h.get("val_precision", 0) for h in history]
            val_rec = [h.get("val_recall", 0) for h in history]
            ax5.plot(epochs, val_f1, 'g-', label='F1', linewidth=2)
            ax5.plot(epochs, val_prec, 'b-', label='Precision', linewidth=2)
            ax5.plot(epochs, val_rec, 'r-', label='Recall', linewidth=2)
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('Score (%)', fontsize=11)
            ax5.set_title('Validation Metrics', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
        else:
            ax5 = plt.subplot(2, 3, 5)
            ax5.text(0.5, 0.5, 'Metrics available only\nfor softmax loss', 
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Validation Metrics', fontsize=12, fontweight='bold')
            ax5.axis('off')
        
        # 6. 训练总结统计
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        final_train_loss = train_loss[-1] if train_loss else 0
        final_val_loss = val_loss[-1] if val_loss else 0
        final_train_acc = train_acc[-1] if train_acc else 0
        final_val_acc = val_acc[-1] if val_acc else 0
        best_val_acc = max(val_acc) if val_acc else 0
        best_val_epoch = epochs[val_acc.index(best_val_acc)] if val_acc else 0
        
        summary_text = f"""Training Summary
        
Final Train Loss: {final_train_loss:.4f}
Final Val Loss: {final_val_loss:.4f}
Final Train Acc: {final_train_acc:.2f}%
Final Val Acc: {final_val_acc:.2f}%
Best Val Acc: {best_val_acc:.2f}%
Best Epoch: {int(best_val_epoch)}
Total Epochs: {len(epochs)}
"""
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        output_path = output_dir / "training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[可视化] 训练曲线已保存: {output_path.name}")
    except Exception as e:
        print(f"[警告] 生成训练曲线失败: {e}")


def _generate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[List[str]],
    output_dir: Path,
) -> None:
    """Generate confusion matrix visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        import seaborn as sns
        
        cm = sk_confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=class_names if class_names else range(len(np.unique(labels))),
            yticklabels=class_names if class_names else range(len(np.unique(labels))),
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_path}")
    except ImportError:
        print("Warning: seaborn or sklearn not available, skipping confusion matrix")
    except Exception as e:
        print(f"Warning: Failed to generate confusion matrix: {e}")


def _generate_confusion_matrix_from_eval(
    encoder: EncoderClassifier,
    dataloader: DataLoader,
    loss_type: str,
    loss_fn: nn.Module,
    device: torch.device,
    classifier_head: Optional[nn.Module],
    output_dir: Path,
    speaker_to_idx: Dict[str, int],
) -> None:
    """Generate confusion matrix from evaluation results."""
    if not HAS_MATPLOTLIB or loss_type != "softmax" or classifier_head is None:
        return
    
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        encoder.eval()
        if classifier_head is not None:
            classifier_head.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for waveforms, wav_lens, labels, _ in dataloader:
                waveforms = waveforms.to(device)
                wav_lens = wav_lens.to(device)
                labels = labels.to(device)
                
                # 直接调用encoder，DataParallel会自动处理多GPU分配
                embeddings = encoder(waveforms, wav_lens)
                logits = classifier_head(embeddings)
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 创建可视化
        idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        speaker_names = [idx_to_speaker.get(i, f"Speaker_{i}") for i in range(len(speaker_to_idx))]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=speaker_names, yticklabels=speaker_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    except ImportError:
        print("Warning: sklearn or seaborn not available, skipping confusion matrix generation")
    except Exception as e:
        print(f"Warning: Failed to generate confusion matrix: {e}")


def _visualize_embeddings(
    embeddings_data: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
    output_dir: Path,
    speaker_to_idx: Dict[str, int],
) -> None:
    """Visualize embeddings using t-SNE."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        from sklearn.manifold import TSNE
        
        embeddings, labels, predictions = embeddings_data
        
        if labels is None or len(embeddings) < 2:
            return
        
        # 使用t-SNE降维到2D
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建可视化
        plt.figure(figsize=(12, 10))
        
        # 根据标签着色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        for i, label in enumerate(unique_labels):
            mask = labels == label
            speaker_name = idx_to_speaker.get(int(label), f"Speaker_{label}")
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=speaker_name, alpha=0.6, s=50)
        
        plt.title('t-SNE Visualization of Speaker Embeddings', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = output_dir / "embeddings_tsne.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE visualization saved to {output_path}")
    except ImportError:
        print("Warning: sklearn not available, skipping t-SNE visualization")
    except Exception as e:
        print(f"Warning: Failed to generate t-SNE visualization: {e}")


def _generate_performance_report(
    history: List[Dict[str, float]],
    output_dir: Path,
    training_cfg: Dict[str, Any],
    validation_cfg: Dict[str, Any],
) -> None:
    """Generate comprehensive performance analysis report."""
    if not history:
        return
    
    try:
        report = {
            "training_summary": {
                "total_epochs": len(history),
                "final_train_loss": history[-1].get("train_loss", 0.0),
                "final_train_accuracy": history[-1].get("train_accuracy", 0.0),
                "final_val_loss": history[-1].get("val_loss", 0.0),
                "final_val_accuracy": history[-1].get("val_accuracy", 0.0),
            },
            "best_performance": {
                "best_val_accuracy": max([h.get("val_accuracy", 0.0) for h in history]),
                "best_val_epoch": max(range(len(history)), key=lambda i: history[i].get("val_accuracy", 0.0)) + 1,
                "best_val_loss": min([h.get("val_loss", float('inf')) for h in history]),
            },
            "training_config": {
                "batch_size": training_cfg.get("batch_size", "N/A"),
                "learning_rate": training_cfg.get("learning_rate", "N/A"),
                "optimizer": training_cfg.get("optimizer", "N/A"),
                "scheduler": training_cfg.get("scheduler", "N/A"),
                "mixed_precision": training_cfg.get("mixed_precision", False),
                "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
            },
            "validation_config": {
                "compute_metrics": validation_cfg.get("compute_metrics", False),
                "multi_metric_early_stop": validation_cfg.get("multi_metric_early_stop", False),
            },
            "recommendations": [],
        }
        
        # 分析训练趋势
        if len(history) > 1:
            train_loss_trend = history[-1]["train_loss"] - history[0]["train_loss"]
            val_loss_trend = history[-1]["val_loss"] - history[0]["val_loss"]
            
            if train_loss_trend > 0:
                report["recommendations"].append("训练损失上升，可能需要降低学习率或增加正则化")
            if val_loss_trend > 0:
                report["recommendations"].append("验证损失上升，可能存在过拟合，建议增加数据增强或早停")
            
            # 检查过拟合
            final_gap = history[-1]["train_accuracy"] - history[-1]["val_accuracy"]
            if final_gap > 10:
                report["recommendations"].append(f"训练和验证准确率差距较大（{final_gap:.2f}%），可能存在过拟合")
        
        # 检查是否有F1指标
        if "val_f1" in history[0]:
            report["training_summary"]["final_val_f1"] = history[-1].get("val_f1", 0.0)
            report["training_summary"]["final_val_precision"] = history[-1].get("val_precision", 0.0)
            report["training_summary"]["final_val_recall"] = history[-1].get("val_recall", 0.0)
        
        # 保存报告
        report_path = output_dir / "performance_report.json"
        _write_json(report, report_path)
        
        # 打印关键信息
        print("\n" + "="*60)
        print("[报告] 训练性能分析")
        print("="*60)
        print(f"训练轮数: {report['training_summary']['total_epochs']}")
        print(f"最佳验证准确率: {report['best_performance']['best_val_accuracy']:.2f}% (Epoch {report['best_performance']['best_val_epoch']})")
        print(f"最终验证准确率: {report['training_summary']['final_val_accuracy']:.2f}%")
        if report["recommendations"]:
            print("\n[建议]")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Warning: Failed to generate performance report: {e}")


def _visualize_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    speaker_to_idx: Dict[str, int],
    output_dir: Path,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """Visualize speaker embeddings using t-SNE."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        from sklearn.manifold import TSNE
        
        # 如果样本太多，随机采样
        max_samples = 5000
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
        
        print(f"Computing t-SNE for {len(embeddings)} samples...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=0)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建反向映射：idx -> speaker_id
        idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 为每个说话人使用不同颜色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            speaker_id = idx_to_speaker.get(int(label), f"Speaker_{label}")
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=speaker_id,
                alpha=0.6,
                s=50,
            )
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Speaker Embeddings Visualization (t-SNE)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = output_dir / "embeddings_tsne.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE visualization saved to {output_path}")
    except ImportError:
        print("Warning: sklearn not available, skipping t-SNE visualization")
    except Exception as e:
        print(f"Warning: Failed to generate t-SNE visualization: {e}")


def _default_paths() -> tuple[Path, Path, Path, Path, Path]:
    base = Path(__file__).resolve().parent
    clean_dir = base / "processed_data" / "speaker_clean"
    model_path = base / "model" / "speaker-identification" / "speaker_embeddings.json"
    onnx_path = base / "model" / "speaker-identification" / "speaker_model.onnx"
    calibration_path = base / "model" / "speaker-identification" / "speaker_calibration.json"
    template_path = base / "model" / "speaker-identification" / "speaker_templates.json"
    return clean_dir, model_path, calibration_path, template_path, onnx_path


def parse_args() -> argparse.Namespace:
    clean_dir, model_path, calibration_path, template_path, onnx_path = _default_paths()
    encoder_dir = SPEECHBRAIN_DEFAULT_DIR
    parser = argparse.ArgumentParser(description="Train SpeechBrain speaker embeddings for N persons (5-200).")
    parser.add_argument(
        "--clean-dir",
        default=str(clean_dir),
        help="Directory containing speaker WAVs (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(model_path),
        help="Path to save speaker embedding JSON (default: %(default)s).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples required per speaker (default: %(default)s).",
    )
    parser.add_argument(
        "--pretrained-model",
        help="Existing speaker_model.bin used as the base for adaptation (optional).",
    )
    parser.add_argument(
        "--adaptation-mode",
        choices=["fine_tune", "feature_extract"],
        default="fine_tune",
        help="Select fine_tune to continue training or feature_extract for template calibration only.",
    )
    parser.add_argument(
        "--calibration-report",
        default=str(calibration_path),
        help="Where to store similarity statistics + recommended threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--template-store",
        default=str(template_path),
        help="JSON file that keeps centroid templates for incremental updates (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-template-store",
        action="store_true",
        help="Skip generating the template store even if --template-store is provided.",
    )

    parser.add_argument(
        "--config",
        help="Optional YAML config overriding CLI options (see speaker_id_config.yaml).",
    )
    parser.add_argument(
        "--onnx-output",
        default=str(onnx_path),
        help="Destination path for exporting the SpeechBrain encoder as ONNX (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Skip exporting the SpeechBrain encoder to ONNX format.",
    )
    parser.add_argument(
        "--encoder-root",
        default=str(encoder_dir),
        help="Directory containing the SpeechBrain ECAPA model checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="Force enable DataParallel for multi-GPU training (auto-detected by default).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-GPU training even if multiple GPUs are available.",
    )
    return parser.parse_args()


def _load_yaml_config(path_value: str | None) -> Dict[str, Any]:
    if not path_value:
        return {}
    config_path = Path(path_value)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping")
    return data


def _summarize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not config:
        return {}
    summary: Dict[str, Any] = {}
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    validation_cfg = config.get("validation", {}) or {}
    for key in ("freeze_layers", "train_layers", "onnx_output_path"):
        if key in model_cfg:
            summary[key] = model_cfg[key]
    for key in ("speaker_count", "num_mfcc", "n_estimators", "threshold", "sample_rate"):
        if key in training_cfg:
            summary[key] = training_cfg[key]
    if validation_cfg:
        summary["validation_targets"] = validation_cfg
    return summary


def _apply_config_overrides(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    if not config:
        setattr(args, "_config_meta", {})
        setattr(args, "_training_cfg", {})
        setattr(args, "_model_cfg", {})
        setattr(args, "_validation_cfg", {})
        return args

    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    validation_cfg = config.get("validation", {}) or {}

    # 解析配置文件所在目录，用于处理相对路径
    base_dir = Path(__file__).resolve().parent
    config_path = getattr(args, "config", None)
    if config_path:
        config_dir = Path(config_path).resolve().parent
    else:
        config_dir = base_dir

    # Model configuration
    if model_cfg.get("pretrained_path"):
        pretrained_path = Path(model_cfg["pretrained_path"])
        if not pretrained_path.is_absolute():
            pretrained_path = config_dir / pretrained_path
        args.pretrained_model = str(pretrained_path)
    if model_cfg.get("output_path"):
        output_path = Path(model_cfg["output_path"])
        if not output_path.is_absolute():
            output_path = config_dir / output_path
        args.output = str(output_path)
    if model_cfg.get("template_store"):
        template_store = Path(model_cfg["template_store"])
        if not template_store.is_absolute():
            template_store = config_dir / template_store
        args.template_store = str(template_store)
    if model_cfg.get("calibration_report"):
        calibration_report = Path(model_cfg["calibration_report"])
        if not calibration_report.is_absolute():
            calibration_report = config_dir / calibration_report
        args.calibration_report = str(calibration_report)
    if model_cfg.get("onnx_output_path"):
        onnx_output_path = Path(model_cfg["onnx_output_path"])
        if not onnx_output_path.is_absolute():
            onnx_output_path = config_dir / onnx_output_path
        args.onnx_output = str(onnx_output_path)
    if model_cfg.get("encoder_root"):
        encoder_root = Path(model_cfg["encoder_root"])
        if not encoder_root.is_absolute():
            encoder_root = config_dir / encoder_root
        args.encoder_root = str(encoder_root)
    if model_cfg.get("skip_onnx_export") is True:
        args.skip_onnx_export = True
    
    # 从配置文件读取基础保存目录
    if model_cfg.get("base_output_dir"):
        base_output_dir = Path(model_cfg["base_output_dir"])
        if not base_output_dir.is_absolute():
            base_output_dir = config_dir / base_output_dir
        setattr(args, "_base_output_dir", str(base_output_dir))

    # Training configuration
    if training_cfg.get("dataset_path"):
        dataset_path = Path(training_cfg["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = config_dir / dataset_path
        args.clean_dir = str(dataset_path)
    if training_cfg.get("sample_rate"):
        args.sample_rate = int(training_cfg["sample_rate"])
    if training_cfg.get("min_samples"):
        args.min_samples = int(training_cfg["min_samples"])
    if training_cfg.get("adaptation_mode"):
        args.adaptation_mode = str(training_cfg["adaptation_mode"])

    if training_cfg.get("skip_template_store") is True:
        args.skip_template_store = True
    
    # 样本数量限制配置
    if training_cfg.get("max_samples_per_speaker") is not None:
        setattr(args, "_max_samples_per_speaker", int(training_cfg["max_samples_per_speaker"]))
    if training_cfg.get("limit_negative_samples") is not None:
        setattr(args, "_limit_negative_samples", bool(training_cfg["limit_negative_samples"]))
    if training_cfg.get("target_employee_prefix"):
        setattr(args, "_target_employee_prefix", str(training_cfg["target_employee_prefix"]))
    if training_cfg.get("use_parallel") is not None:
        args.use_parallel = bool(training_cfg["use_parallel"])

    setattr(args, "_training_cfg", training_cfg)
    setattr(args, "_model_cfg", model_cfg)
    setattr(args, "_validation_cfg", validation_cfg)
    setattr(args, "_config_meta", _summarize_config(config))
    return args


def _ensure_encoder_assets(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(
            f"SpeechBrain encoder assets not found at {root}. Place the downloaded files in this folder or pass --encoder-root."
        )
    missing = [name for name in REQUIRED_ENCODER_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"SpeechBrain encoder root {root} is missing required files: {missing}."
        )


def _load_encoder(device: str = "cpu", encoder_root: Optional[Path] = None) -> EncoderClassifier:
    root = Path(encoder_root) if encoder_root else SPEECHBRAIN_DEFAULT_DIR
    root = root.expanduser().resolve()
    _ensure_encoder_assets(root)
    _ensure_windows_symlink_fallback()
    sid = EncoderClassifier.from_hparams(
        source=str(root),
        savedir=str(root),
        run_opts={"device": device},
    )
    sid.eval()
    return sid


def _iter_wavs(paths: Iterable[Path]) -> Iterable[Path]:
    for wav in paths:
        if wav.suffix.lower() != ".wav":
            continue
        yield wav


def _extract_speaker_id(wav_path: Path) -> Optional[str]:
    """
    从音频文件路径中提取说话人ID，支持多种命名格式：
    1. emp_SPEAKER_ID_XXX.wav (原始格式)
    2. pSPEAKER_ID_XXX.wav (VCTK格式)
    3. 从目录名识别 (如果文件在子目录中)
    
    Args:
        wav_path: 音频文件路径
        
    Returns:
        说话人ID，如果无法识别则返回None
    """
    name = wav_path.name
    speaker_id = None
    
    # 方式1: 支持 emp_XXX_XX.wav 格式
    if name.startswith("emp_"):
        parts = name.split("_")
        if len(parts) >= 3:
            speaker_id = parts[1]
    
    # 方式2: 支持 pXXX_XX.wav 格式（VCTK格式）
    elif name.startswith("p") and "_" in name:
        parts = name.split("_")
        if len(parts) >= 2:
            # 提取 p225 中的 225，添加 'p' 前缀以区分VCTK格式
            speaker_id = "p" + parts[0][1:]  # 保留 'p' 前缀，如 p225
    
    # 方式3: 如果文件名不符合格式，尝试从目录名识别
    if speaker_id is None:
        parent_dir = wav_path.parent.name
        # 检查父目录是否是说话人目录
        if parent_dir.startswith("emp_"):
            parts = parent_dir.split("_")
            if len(parts) >= 2:
                # 保留 emp_ 前缀以区分目标员工
                speaker_id = parent_dir  # 直接使用目录名，如 emp_001
        elif parent_dir.startswith("p") and len(parent_dir) > 1:
            # VCTK格式目录名，如 p225
            # 如果目录名是 p225，提取为 p225（保留p前缀）
            speaker_id = parent_dir  # 直接使用目录名，如 p225
    
    # 如果从文件名提取的speaker_id没有前缀，但文件名或目录名以emp_开头，添加前缀
    if speaker_id and not speaker_id.startswith(("emp_", "p")):
        # 检查文件名或目录名是否以emp_开头
        if wav_path.name.startswith("emp_") or wav_path.parent.name.startswith("emp_"):
            speaker_id = f"emp_{speaker_id}"
    
    return speaker_id


def _collect_samples(
    dataset_dir: Path, 
    min_samples: int,
    max_samples_per_speaker: Optional[int] = None,
    target_employee_prefix: str = "emp_",
    limit_negative_samples: bool = False
) -> Tuple[List[str], Dict[str, int]]:
    """
    收集训练样本，支持限制每个说话人的样本数量
    
    Args:
        dataset_dir: 数据集目录
        min_samples: 每个说话人至少需要的样本数量
        max_samples_per_speaker: 每个说话人最多使用的样本数量（None表示不限制）
        target_employee_prefix: 目标员工ID前缀（用于区分目标员工和负样本）
        limit_negative_samples: 是否只限制负样本（VCTK）的样本数量，目标员工不受限制
    
    Returns:
        (samples, per_speaker): 样本列表和每个说话人的样本数量统计
    """
    import random
    
    # 第一步：收集所有样本，按说话人分组
    samples_by_speaker: Dict[str, List[str]] = {}
    for wav in sorted(_iter_wavs(dataset_dir.rglob("*.wav"))):
        speaker_id = _extract_speaker_id(wav)
        if speaker_id:
            if speaker_id not in samples_by_speaker:
                samples_by_speaker[speaker_id] = []
            samples_by_speaker[speaker_id].append(str(wav))
    
    if not samples_by_speaker:
        raise FileNotFoundError(
            f"No samples found in {dataset_dir}. Please check the dataset path or ensure your files follow the naming convention:\n"
            f"  - emp_SPEAKER_ID_XXX.wav (original format)\n"
            f"  - pSPEAKER_ID_XXX.wav (VCTK format)\n"
            f"  - Or files in subdirectories named emp_XXX or pXXX"
        )

    # 第二步：应用样本数量限制
    samples: List[str] = []
    per_speaker: Dict[str, int] = {}
    
    for speaker_id, speaker_samples in samples_by_speaker.items():
        # 判断是否是目标员工
        is_target_employee = speaker_id.startswith(target_employee_prefix)
        
        # 确定是否应用限制
        should_limit = False
        if max_samples_per_speaker is not None:
            if limit_negative_samples:
                # 只限制负样本（非目标员工）
                should_limit = not is_target_employee
            else:
                # 限制所有说话人
                should_limit = True
        
        # 应用限制
        if should_limit and len(speaker_samples) > max_samples_per_speaker:
            # 随机选择指定数量的样本
            selected_samples = random.sample(speaker_samples, max_samples_per_speaker)
            samples.extend(selected_samples)
            per_speaker[speaker_id] = max_samples_per_speaker
            print(f"[限制] {speaker_id}: {len(speaker_samples)} -> {max_samples_per_speaker} 个样本")
        else:
            # 不限制或样本数未超过限制
            samples.extend(speaker_samples)
            per_speaker[speaker_id] = len(speaker_samples)
    
    # 检查最小样本数要求
    incomplete = [sid for sid, count in per_speaker.items() if count < min_samples]
    if incomplete:
        print(f"⚠️  Warning: speakers missing samples ({min_samples} required): {incomplete}")
    
    return samples, per_speaker


def _choose_params(headcount: int) -> Tuple[int, int, float]:
    if headcount <= 20:
        return 18, 120, 0.92
    if headcount <= 100:
        return 22, 160, 0.93
    return 24, 200, 0.94


def _load_pretrained_config(model_path: Path, sample_rate: int) -> Dict[str, float]:
    # For SpeechBrain, we load the pre-trained embeddings from JSON file
    try:
        with open(model_path, 'r') as f:
            data = json.load(f)
        config: Dict[str, float] = {
            "num_mfcc": 13,  # Default value, not used by SpeechBrain
            "threshold": 0.85  # Default threshold for cosine similarity
        }
        return config
    except Exception as exc:
        raise RuntimeError(f"Failed to load pretrained model {model_path}: {exc}") from exc


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _resample_audio(audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    if src_rate == target_rate:
        return audio
    duration = len(audio) / src_rate
    target_len = int(duration * target_rate)
    if target_len <= 0:
        raise ValueError("Audio duration too short for resampling.")
    x_original = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_target = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_target, x_original, audio)


def _prepare_waveform(wav_path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio = _ensure_mono(audio)
    audio = _resample_audio(audio, sr, sample_rate)
    return audio.astype(np.float32)


def _group_samples_by_speaker(samples: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for wav in samples:
        wav_path = Path(wav)
        speaker_id = _extract_speaker_id(wav_path)
        if speaker_id:
            grouped.setdefault(speaker_id, []).append(wav)
    return grouped


def _create_resampler(sample_rate: int) -> Optional[torchaudio.transforms.Resample]:
    if sample_rate == 16000:
        return None
    return torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)


def _compute_embedding_from_waveform(
    waveform: np.ndarray,
    encoder: EncoderClassifier,
    resampler: Optional[torchaudio.transforms.Resample],
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    if resampler is not None:
        tensor = resampler(tensor)
    tensor = tensor.to(device)
    
    # 计算 wav_lens（音频长度）
    wav_lens = torch.ones(tensor.shape[0], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # 处理 DataParallel 和 EncoderWrapper 包装
        # 如果被 DataParallel 包装，先获取 module
        if hasattr(encoder, "module"):
            encoder = encoder.module
        # 如果被 EncoderWrapper 包装，获取原始 encoder
        if hasattr(encoder, "encoder"):
            # 是 EncoderWrapper，需要调用 encode_batch 并传入 wav_lens
            embedding = encoder.encode_batch(tensor, wav_lens)
        else:
            # 是原始 EncoderClassifier，调用 encode_batch
            embedding = encoder.encode_batch(tensor, wav_lens)
    
    embedding = embedding.squeeze().cpu().numpy()
    if embedding.ndim > 1:
        embedding = embedding.reshape(-1)
    return embedding.astype(np.float32)


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _load_enrolled_embeddings(model_path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Enrolled embeddings must be a non-empty mapping.")
    if "speakers" in payload:
        payload = payload["speakers"]
        embeddings = {
            speaker_id: np.asarray(info["centroid"], dtype=np.float32)
            for speaker_id, info in payload.items()
            if isinstance(info, dict) and "centroid" in info
        }
    else:
        embeddings = {
            speaker_id: np.asarray(vector, dtype=np.float32)
            for speaker_id, vector in payload.items()
        }
    if not embeddings:
        raise ValueError("No enrolled speaker embeddings found in model file.")
    return embeddings


def _calibrate_model(
    model_path: Path,
    samples: List[str],
    sample_rate: int,
    encoder: EncoderClassifier,
    resampler: Optional[torchaudio.transforms.Resample],
    device: torch.device,
) -> Dict[str, object]:
    if not samples:
        raise ValueError("Calibration requires at least one sample.")

    enrolled = _load_enrolled_embeddings(model_path)
    similarities: List[float] = []
    details: List[Dict[str, object]] = []

    for wav in samples:
        waveform = _prepare_waveform(Path(wav), sample_rate)
        embedding = _compute_embedding_from_waveform(waveform, encoder, resampler, device)
        best_similarity = max(_cosine(embedding, enrolled_vec) for enrolled_vec in enrolled.values())
        similarities.append(best_similarity)
        details.append({"file": wav, "similarity": round(best_similarity, 4)})

    min_sim = min(similarities)
    mean_sim = statistics.mean(similarities)
    recommended = max(0.75, min(min_sim - 0.02, 0.99))
    report = {
        "model_path": str(model_path),
        "sample_rate": sample_rate,
        "samples_evaluated": len(samples),
        "recommended_threshold": round(recommended, 4),
        "mean_similarity": round(mean_sim, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max(similarities), 4),
        "details": details,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return report


def _write_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _build_template_store(
    samples: List[str],
    sample_rate: int,
    encoder: EncoderClassifier,
    resampler: Optional[torchaudio.transforms.Resample],
    device: torch.device,
    cached_embeddings: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Dict[str, object]:
    per_speaker: Dict[str, List[np.ndarray]] = {}
    if cached_embeddings:
        for speaker_id, vectors in cached_embeddings.items():
            valid_vectors = [np.asarray(vec, dtype=np.float32) for vec in vectors]
            if valid_vectors:
                per_speaker[speaker_id] = valid_vectors
    else:
        grouped = _group_samples_by_speaker(samples)
        for speaker_id, wavs in grouped.items():
            vectors: List[np.ndarray] = []
            for wav in wavs:
                waveform = _prepare_waveform(Path(wav), sample_rate)
                vectors.append(
                    _compute_embedding_from_waveform(waveform, encoder, resampler, device)
                )
            if vectors:
                per_speaker[speaker_id] = vectors

    store: Dict[str, object] = {
        "version": 1,
        "sample_rate": sample_rate,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "speakers": {},
    }
    speakers_payload: Dict[str, object] = {}
    for speaker_id, vectors in per_speaker.items():
        stack = np.vstack(vectors)
        centroid = stack.mean(axis=0)
        sims = [_cosine(vec, centroid) for vec in vectors]
        speakers_payload[speaker_id] = {
            "samples": len(vectors),
            "centroid": centroid.tolist(),
            "min_similarity": round(min(sims), 4),
            "mean_similarity": round(statistics.mean(sims), 4),
        }
    store["speakers"] = speakers_payload
    return store


def _export_encoder_to_onnx(
    encoder_root: Path,
    onnx_path: Path,
    sample_rate: int = 16000,
    classifier: Optional[EncoderClassifier] = None,
) -> bool:
    """Export SpeechBrain encoder to ONNX format.
    
    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        # 确保模型在CPU上
        if classifier is None:
            classifier = _load_encoder("cpu", encoder_root)
        else:
            # 更彻底地迁移到CPU：先移到CPU，然后确保所有子模块也在CPU上
            classifier = classifier.cpu()
            # 递归确保所有子模块都在CPU上
            for module in classifier.modules():
                module.cpu()
            # 确保所有参数都在CPU上（处理可能的CUDA残留）
            for param in classifier.parameters():
                if param.is_cuda:
                    param.data = param.data.cpu()
            for buffer in classifier.buffers():
                if buffer.is_cuda:
                    buffer.data = buffer.data.cpu()
            # 确保模型在评估模式
            classifier.eval()
            # 强制同步，确保所有操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # 清空CUDA缓存
                torch.cuda.empty_cache()

        # 再次确保所有组件都在CPU上（双重检查）
        classifier = classifier.cpu()
        for name, param in classifier.named_parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
        for name, buffer in classifier.named_buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()

        class _EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder: EncoderClassifier) -> None:
                super().__init__()
                # 使用encoder的encode_batch方法，但确保所有组件在CPU上
                self.encoder = encoder
                # 递归确保所有子模块都在CPU上
                for module in self.encoder.modules():
                    module.cpu()
                # 确保所有参数和buffer都在CPU上
                for param in self.encoder.parameters():
                    if param.is_cuda:
                        param.data = param.data.cpu()
                for buffer in self.encoder.buffers():
                    if buffer.is_cuda:
                        buffer.data = buffer.data.cpu()

            def forward(self, waveform: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                # 确保输入在CPU上
                if waveform.is_cuda:
                    waveform = waveform.detach().cpu().clone()
                else:
                    waveform = waveform.detach().clone()
                
                # 明确在CPU上创建wav_lens
                wav_lens = torch.ones(waveform.shape[0], dtype=torch.float32, device=torch.device("cpu"))
                
                # 使用torch.no_grad()确保不会创建计算图
                with torch.no_grad():
                    # 调用encode_batch，但确保所有操作在CPU上
                    # 临时将encoder的所有组件移到CPU（如果还没在CPU上）
                    self.encoder = self.encoder.cpu()
                    for module in self.encoder.modules():
                        module.cpu()
                    
                    # 处理 DataParallel 和 EncoderWrapper 包装
                    encoder_to_use = self.encoder
                    if hasattr(encoder_to_use, "module"):
                        encoder_to_use = encoder_to_use.module
                    if hasattr(encoder_to_use, "encoder"):
                        # 是 EncoderWrapper，调用 encode_batch
                        result = encoder_to_use.encode_batch(waveform, wav_lens)
                    else:
                        # 是原始 EncoderClassifier，调用 encode_batch
                        result = encoder_to_use.encode_batch(waveform, wav_lens)
                    
                    # 如果结果在CUDA上，移到CPU
                    if result.is_cuda:
                        result = result.detach().cpu().clone()
                    else:
                        result = result.detach().clone()
                    
                    # 确保结果是连续的CPU tensor
                    result = result.contiguous()
                    
                    # 处理输出维度
                    if result.ndim == 3:
                        result = result.squeeze(1)
                    
                    return result

        wrapper = _EncoderWrapper(classifier)
        wrapper.eval()
        
        # 确保wrapper也在CPU上（三重保险）
        wrapper = wrapper.cpu()
        for module in wrapper.modules():
            module.cpu()
        for param in wrapper.parameters():
            if param.is_cuda:
                param.data = param.data.cpu()
        for buffer in wrapper.buffers():
            if buffer.is_cuda:
                buffer.data = buffer.data.cpu()
        
        # 创建CPU上的示例输入（明确指定device，使用detach确保不共享内存）
        dummy_seconds = 3
        dummy_input = torch.zeros(1, sample_rate * dummy_seconds, dtype=torch.float32, device=torch.device("cpu"))
        dummy_input = dummy_input.detach()  # 确保是独立的tensor
        
        # 再次确保输入在CPU上（双重检查）
        if dummy_input.is_cuda:
            dummy_input = dummy_input.detach().cpu()
        
        # 验证所有组件都在CPU上
        assert not dummy_input.is_cuda, "Dummy input must be on CPU"
        for name, param in wrapper.named_parameters():
            assert not param.is_cuda, f"Parameter {name} is on CUDA"
        for name, buffer in wrapper.named_buffers():
            assert not buffer.is_cuda, f"Buffer {name} is on CUDA"
        
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用torch.no_grad()进行导出
        with torch.no_grad():
            # 在导出前再次确保所有组件在CPU上
            wrapper = wrapper.cpu()
            dummy_input = dummy_input.cpu()
            
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(onnx_path),
                input_names=["waveform"],
                output_names=["embedding"],
                dynamic_axes={"waveform": {1: "samples"}, "embedding": {0: "batch"}},
                opset_version=17,
            )
        return True
    except Exception as exc:  # pragma: no cover - depends on torch install
        summary = str(exc).splitlines()[0]
        if "[Caused by" in summary:
            summary = summary.split("[Caused by", 1)[0].strip()
        print(f"[警告] ONNX导出失败: {summary}")
        return False


def main() -> None:
    args = parse_args()
    # 自动检测GPU数量
    gpu_count = torch.cuda.device_count()
    has_cuda = torch.cuda.is_available()
    
    config = _load_yaml_config(getattr(args, "config", None))
    args = _apply_config_overrides(args, config)
    
    # 在配置文件应用后，重新判断是否启用多GPU训练
    if getattr(args, "no_parallel", False):
        use_parallel = False
        if gpu_count > 1:
            print(f"[GPU] 检测到{gpu_count}个GPU，但已通过--no-parallel禁用多GPU训练")
    else:
        use_parallel = getattr(args, "use_parallel", gpu_count > 1)
        if has_cuda:
            if gpu_count > 1 and use_parallel:
                print(f"[GPU] 检测到{gpu_count}个GPU，启用多GPU并行训练")
            elif gpu_count == 1:
                print(f"[GPU] 检测到1个GPU，使用单GPU训练")
            else:
                print(f"[GPU] CUDA可用，使用GPU训练")
        else:
            print("[CPU] 未检测到GPU，使用CPU训练")
    base_dir = Path(__file__).resolve().parent
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 初始化日志系统（参考 test_noise.py 的实现方式）
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs" / "speaker"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"train_speaker_{run_timestamp}.log"
    log_file_path = log_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Speaker Recognition Training Started")
    logger.info(f"Timestamp: {run_timestamp}")
    logger.info(f"Log File: {log_file_path}")
    logger.info("=" * 80)
    
    # 记录GPU/CPU信息
    if getattr(args, "no_parallel", False):
        if gpu_count > 1:
            logger.info(f"GPU: 检测到{gpu_count}个GPU，但已通过--no-parallel禁用多GPU训练")
    else:
        if has_cuda:
            if gpu_count > 1 and use_parallel:
                logger.info(f"GPU: 检测到{gpu_count}个GPU，启用多GPU并行训练")
            elif gpu_count == 1:
                logger.info(f"GPU: 检测到1个GPU，使用单GPU训练")
            else:
                logger.info(f"GPU: CUDA可用，使用GPU训练")
        else:
            logger.info("GPU: 未检测到GPU，使用CPU训练")
    
    # 优先使用配置文件中的base_output_dir，否则使用默认的models目录
    base_output_dir = Path(getattr(args, "_base_output_dir", str(base_dir / "models")))
    run_dir = base_output_dir / f"speechbrain_model_{run_timestamp}"
    
    # 确保run_dir目录存在
    run_dir = run_dir.resolve()  # 解析为绝对路径
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output Directory: {run_dir}")
    if getattr(args, "config", None):
        logger.info(f"Config File: {args.config}")
    
    # Consolidate SpeechBrain artifacts inside the timestamped run directory.

    def _target_path(path_value: str, default_name: str) -> Path:
        path_obj = Path(path_value) if path_value else Path(default_name)
        name = path_obj.name or default_name
        return run_dir / name

    output_path = _target_path(args.output, "speaker_embeddings.json")
    calibration_target = _target_path(args.calibration_report, "speaker_calibration.json")
    template_target = _target_path(getattr(args, "template_store", ""), "speaker_templates.json")
    onnx_target = _target_path(args.onnx_output, "speaker_model.onnx")

    args.output = str(output_path)
    args.calibration_report = str(calibration_target)
    args.template_store = str(template_target)
    args.onnx_output = str(onnx_target)

    setattr(args, "_run_dir", str(run_dir))
    setattr(args, "_run_timestamp", run_timestamp)

    # 使用clean_dir作为主要数据集目录（兼容现有配置）
    dataset_dir = Path(args.clean_dir)
    
    # 获取样本数量限制参数
    max_samples_per_speaker = getattr(args, "_max_samples_per_speaker", None)
    limit_negative_samples = getattr(args, "_limit_negative_samples", False)
    target_employee_prefix = getattr(args, "_target_employee_prefix", "emp_")
    
    # 尝试从dataset_dir收集样本（递归搜索所有子目录）
    samples, counts = _collect_samples(
        dataset_dir, 
        args.min_samples,
        max_samples_per_speaker=max_samples_per_speaker,
        target_employee_prefix=target_employee_prefix,
        limit_negative_samples=limit_negative_samples
    )
    
    headcount = len(counts)
    training_cfg = getattr(args, "_training_cfg", {}) or {}
    model_cfg = getattr(args, "_model_cfg", {}) or {}
    validation_cfg = getattr(args, "_validation_cfg", {}) or {}

    pretrained_path = Path(args.pretrained_model) if args.pretrained_model else None
    calibration_path = Path(args.calibration_report)
    template_path = Path(args.template_store) if args.template_store else None

    if pretrained_path and not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    encoder_root = Path(args.encoder_root).expanduser()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    encoder = _load_encoder(device_str, encoder_root)
    resampler = _create_resampler(args.sample_rate)

    logger.info("=" * 60)
    logger.info(f"[设备] {device_str}")
    logger.info(f"[数据] 说话人数: {headcount} | 样本数: {len(samples)}")
    logger.info(f"[路径] 数据集: {dataset_dir}")
    logger.info(f"[路径] 输出目录: {run_dir}")
    logger.info("=" * 60)

    print("=" * 60)
    print(f"[设备] {device_str}")
    print(f"[数据] 说话人数: {headcount} | 样本数: {len(samples)}")
    print(f"[路径] 数据集: {dataset_dir}")
    print(f"[路径] 输出目录: {run_dir}")
    print("=" * 60)

    meta = getattr(args, "_config_meta", {})

    if pretrained_path and args.adaptation_mode == "feature_extract":
        import shutil

        shutil.copy2(pretrained_path, args.output)
        report = _calibrate_model(Path(args.output), samples, args.sample_rate, encoder, resampler, device)
        if meta:
            report["config"] = meta
        report["dataset_root"] = str(dataset_dir)
        report["artifacts_directory"] = str(run_dir)
        report["run_timestamp"] = run_timestamp
        _write_json(report, calibration_path)
        print("Calibration report refreshed ✔")

        if template_path and not args.skip_template_store:
            template_store = _build_template_store(samples, args.sample_rate, encoder, resampler, device)
            if meta:
                template_store["config"] = meta
            template_store["dataset_root"] = str(dataset_dir)
            template_store["artifacts_directory"] = str(run_dir)
            template_store["run_timestamp"] = run_timestamp
            _write_json(template_store, template_path)
            print(f"Template store updated at {template_path}")

        if not args.skip_onnx_export:
            # 使用CPU上的encoder进行导出
            print("[导出] 准备导出ONNX模型...")
            cpu_encoder = _load_encoder("cpu", encoder_root)
            if _export_encoder_to_onnx(encoder_root, Path(args.onnx_output), sample_rate=args.sample_rate, classifier=cpu_encoder):
                print(f"[导出] ONNX模型: {Path(args.onnx_output).name}")
        print(f"Feature extraction adaptation complete ✔  Run artifacts stored in {run_dir}")
        return

    samples_by_speaker = _group_samples_by_speaker(samples)
    if not samples_by_speaker:
        raise ValueError("No valid speaker samples detected (expected filenames like emp_001_xxx.wav).")

    validation_enabled = bool(validation_cfg.get("enabled", True))
    validation_ratio = float(training_cfg.get("validation_ratio", 0.2))
    split_seed = int(training_cfg.get("split_seed", 42))

    train_samples, val_samples = _split_train_val(samples_by_speaker, validation_ratio, seed=split_seed)
    if not train_samples and val_samples:
        train_samples, val_samples = val_samples, []

    augment_enabled = bool(training_cfg.get("augment", False))
    noise_level = float(training_cfg.get("noise_level", 0.05))
    speed_cfg = training_cfg.get("speed_range", (0.9, 1.1))
    if isinstance(speed_cfg, (list, tuple)) and len(speed_cfg) == 2:
        speed_range = (float(speed_cfg[0]), float(speed_cfg[1]))
    else:
        speed_range = (0.9, 1.1)
    
    volume_cfg = training_cfg.get("volume_range", (0.8, 1.2))
    if isinstance(volume_cfg, (list, tuple)) and len(volume_cfg) == 2:
        volume_range = (float(volume_cfg[0]), float(volume_cfg[1]))
    else:
        volume_range = (0.8, 1.2)
    
    time_shift_prob = float(training_cfg.get("time_shift_prob", 0.3))
    specaugment = bool(training_cfg.get("specaugment", False))
    specaugment_time_mask = int(training_cfg.get("specaugment_time_mask", 27))
    specaugment_freq_mask = int(training_cfg.get("specaugment_freq_mask", 12))
    reverb_prob = float(training_cfg.get("reverb_prob", 0.0))
    reverb_room_scale = float(training_cfg.get("reverb_room_scale", 0.3))
    
    # 多人对话混合增强参数
    multi_speaker_mix_prob = float(training_cfg.get("multi_speaker_mix_prob", 0.3))
    multi_speaker_mix_ratio_cfg = training_cfg.get("multi_speaker_mix_ratio", [0.6, 0.8])
    if isinstance(multi_speaker_mix_ratio_cfg, (list, tuple)) and len(multi_speaker_mix_ratio_cfg) == 2:
        multi_speaker_mix_ratio = (float(multi_speaker_mix_ratio_cfg[0]), float(multi_speaker_mix_ratio_cfg[1]))
    else:
        multi_speaker_mix_ratio = (0.6, 0.8)
    
    # 收集负样本（非目标员工语音）用于多人对话混合
    target_employee_prefix = str(training_cfg.get("target_employee_prefix", "emp_"))
    negative_samples: List[str] = []
    if augment_enabled and multi_speaker_mix_prob > 0:
        # 从所有样本中筛选出非目标员工样本
        for speaker_id, speaker_samples_list in samples_by_speaker.items():
            if not speaker_id.startswith(target_employee_prefix):
                # 非目标员工样本（如VCTK）
                negative_samples.extend(speaker_samples_list)
        
        if negative_samples:
            print(f"[增强] 收集到 {len(negative_samples)} 个负样本用于多人对话混合")
            logger.info(f"[增强] 多人对话混合: 概率={multi_speaker_mix_prob:.1%}, 混合比例={multi_speaker_mix_ratio[0]:.1%}-{multi_speaker_mix_ratio[1]:.1%}, 负样本数={len(negative_samples)}")
        else:
            print(f"[警告] 未找到负样本，多人对话混合将被禁用")
            logger.warning(f"[警告] 未找到负样本，多人对话混合将被禁用")

    train_dataset = SpeakerDataset(
        train_samples,
        args.sample_rate,
        augment=augment_enabled,
        noise_level=noise_level,
        speed_range=speed_range,
        volume_range=volume_range,
        time_shift_prob=time_shift_prob,
        specaugment=specaugment,
        specaugment_time_mask=specaugment_time_mask,
        specaugment_freq_mask=specaugment_freq_mask,
        reverb_prob=reverb_prob,
        reverb_room_scale=reverb_room_scale,
        negative_samples=negative_samples if augment_enabled and multi_speaker_mix_prob > 0 else None,
        multi_speaker_mix_prob=multi_speaker_mix_prob,
        multi_speaker_mix_ratio=multi_speaker_mix_ratio,
        target_employee_prefix=target_employee_prefix,
    )

    val_dataset: Optional[SpeakerDataset] = None
    if validation_enabled and val_samples:
        # 方案B：验证集也使用多人对话混合，让验证指标更准确
        # 建议2：验证集混合概率7.5%（25%训练集混合概率 * 30% = 7.5%），不要超过10%
        val_mix_prob = multi_speaker_mix_prob * 0.3 if augment_enabled and multi_speaker_mix_prob > 0 else 0.0
        
        val_dataset = SpeakerDataset(
            val_samples,
            args.sample_rate,
            augment=True,  # 改为True，启用增强（但混合概率较低）
            noise_level=noise_level,
            speed_range=speed_range,
            volume_range=volume_range,
            time_shift_prob=time_shift_prob,
            specaugment=False,  # 验证集不使用SpecAugment
            specaugment_time_mask=specaugment_time_mask,
            specaugment_freq_mask=specaugment_freq_mask,
            reverb_prob=0.0,  # 验证集不使用混响
            reverb_room_scale=reverb_room_scale,
            speaker_to_idx=train_dataset.speaker_to_idx,
            negative_samples=negative_samples if augment_enabled and val_mix_prob > 0 else None,
            multi_speaker_mix_prob=val_mix_prob,  # 验证集使用7.5%的混合概率（25%训练集 * 30% = 7.5%）
            multi_speaker_mix_ratio=multi_speaker_mix_ratio,  # 使用相同的混合比例
            target_employee_prefix=target_employee_prefix,
        )
        
        if val_mix_prob > 0:
            print(f"[增强] 验证集也启用多人对话混合: 概率={val_mix_prob:.1%}")
            logger.info(f"[增强] 验证集多人对话混合: 概率={val_mix_prob:.1%}, 混合比例={multi_speaker_mix_ratio[0]:.1%}-{multi_speaker_mix_ratio[1]:.1%}")

    print("Starting SpeechBrain fine-tuning...")
    print(
        f"Training samples: {len(train_dataset)}  Validation samples: {len(val_dataset) if val_dataset else 0}  Loss: {training_cfg.get('loss_function', 'contrastive')}"
    )

    training_result = _train_model(
        encoder,
        train_dataset,
        val_dataset,
        training_cfg,
        model_cfg,
        validation_cfg,
        device,
        run_dir,
        use_parallel,
    )

    _save_fine_tuned_weights(training_result, run_dir)
    if training_result.history:
        metrics_path = run_dir / "training_metrics.json"
        _write_json({"history": training_result.history}, metrics_path)
        print(f"[指标] 训练指标已保存: {metrics_path.name}")
        
        # 生成训练曲线可视化
        _visualize_training_history(training_result.history, run_dir)

    encoder = training_result.encoder
    encoder.eval()
    
    # 生成混淆矩阵和TSNE可视化（如果启用）
    visualize_embeddings = bool(validation_cfg.get("visualize_embeddings", False))
    if visualize_embeddings and val_dataset is not None and len(val_dataset) > 0:
        print("Generating confusion matrix and t-SNE visualization...")
        # 重新创建必要的变量
        loss_type, loss_fn = _get_loss_function(training_cfg)
        batch_size = int(training_cfg.get("batch_size", 32))
        num_workers = int(training_cfg.get("num_workers", 4))
        if num_workers == 4:
            try:
                import os
                cpu_count = os.cpu_count() or 4
                num_workers = min(16, max(4, cpu_count // 3))
            except Exception:
                num_workers = 4
        pin_memory = training_cfg.get("pin_memory", torch.cuda.is_available())
        if not isinstance(pin_memory, bool):
            pin_memory = torch.cuda.is_available()
        
        # 重新评估以收集嵌入向量和预测
        from torch.utils.data import DataLoader
        val_loader_viz = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        
        _, _, val_metrics_viz, embeddings_data = _evaluate(
            encoder,
            val_loader_viz,
            loss_type,
            loss_fn,
            device,
            training_result.classifier_head,
            compute_metrics=True,
            collect_embeddings=True,
        )
        
        if embeddings_data and embeddings_data[1] is not None:
            embeddings, labels, predictions = embeddings_data
            
            # 生成混淆矩阵（仅softmax损失函数）
            if loss_type == "softmax" and predictions is not None:
                class_names = [speaker_id for speaker_id in sorted(train_dataset.speaker_to_idx.keys())]
                _generate_confusion_matrix(
                    labels,
                    predictions,
                    class_names,
                    run_dir,
                )
            
            # 生成TSNE可视化（所有损失函数都支持）
            _visualize_embeddings_tsne(
                embeddings,
                labels,
                train_dataset.speaker_to_idx,
                run_dir,
            )

    print("[生成] 说话人嵌入向量...")
    
    # 从配置中获取目标员工前缀（用于过滤模板）
    target_employee_prefix = str(training_cfg.get("target_employee_prefix", "emp_"))
    print(f"[模板] 目标员工前缀: {target_employee_prefix} (只保存以此前缀开头的说话人模板)")
    
    speaker_embeddings: Dict[str, List[float]] = {}
    cached_vectors: Dict[str, List[np.ndarray]] = {}
    target_count = 0
    filtered_count = 0
    
    for speaker_id, speaker_samples in samples_by_speaker.items():
        # 处理说话人样本（静默处理，减少日志）
        vectors: List[np.ndarray] = []
        for wav in speaker_samples:
            waveform = _prepare_waveform(Path(wav), args.sample_rate)
            vectors.append(
                _compute_embedding_from_waveform(waveform, encoder, resampler, device)
            )
        if vectors:
            cached_vectors[speaker_id] = vectors
            
            # 只保存目标员工的模板（过滤VCTK等负样本）
            if speaker_id.startswith(target_employee_prefix):
            speaker_embeddings[speaker_id] = np.mean(vectors, axis=0).tolist()
                target_count += 1
            else:
                filtered_count += 1
                print(f"[模板] 过滤非目标员工: {speaker_id} (VCTK负样本，不保存模板)")
    
    print(f"[模板] 已保存 {target_count} 个目标员工模板，过滤 {filtered_count} 个非目标员工")

    model_path = Path(args.output)
    _write_json(speaker_embeddings, model_path)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[保存] 说话人嵌入向量: {model_path.name} ({size_mb:.2f} MB)")
        print(f"[保存] 包含 {len(speaker_embeddings)} 个目标员工模板: {list(speaker_embeddings.keys())}")

    report = _calibrate_model(model_path, samples, args.sample_rate, encoder, resampler, device)
    if training_result.history:
        report["training_history"] = training_result.history
        report["final_training_metrics"] = training_result.history[-1]
    if meta:
        report["config"] = meta
    report["fine_tuned_embedding_model"] = str(run_dir / "fine_tuned_embedding_model.pt")
    if training_result.classifier_head is not None:
        report["classification_head_path"] = str(run_dir / "classification_head.pt")
    report["dataset_root"] = str(dataset_dir)
    report["artifacts_directory"] = str(run_dir)
    report["run_timestamp"] = run_timestamp
    _write_json(report, calibration_path)
    print(f"[校准] 推荐阈值: {report['recommended_threshold']:.2f} | 报告: {calibration_path.name}")

    if template_path and not args.skip_template_store:
        template_store = _build_template_store(
            samples,
            args.sample_rate,
            encoder,
            resampler,
            device,
            cached_embeddings=cached_vectors,
        )
        if training_result.history:
            template_store["training_history"] = training_result.history
        template_store["fine_tuned_embedding_model"] = str(run_dir / "fine_tuned_embedding_model.pt")
        if training_result.classifier_head is not None:
            template_store["classification_head_path"] = str(run_dir / "classification_head.pt")
        if meta:
            template_store["config"] = meta
        template_store["dataset_root"] = str(dataset_dir)
        template_store["artifacts_directory"] = str(run_dir)
        template_store["run_timestamp"] = run_timestamp
        _write_json(template_store, template_path)
        print(f"[保存] 模板存储: {template_path.name}")

    if not args.skip_onnx_export:
        # 确保encoder在CPU上后再导出ONNX
        print("[导出] 准备导出ONNX模型...")
        # 获取原始encoder（处理DataParallel包装）
        original_encoder = encoder.module if hasattr(encoder, "module") else encoder
        # 创建一个新的encoder实例在CPU上用于导出
        cpu_encoder = _load_encoder("cpu", encoder_root)
        # 如果训练了模型，加载训练后的权重
        fine_tuned_path = run_dir / "fine_tuned_embedding_model.pt"
        if fine_tuned_path.exists():
            try:
                state_dict = torch.load(fine_tuned_path, map_location="cpu")
                cpu_encoder.mods.embedding_model.load_state_dict(state_dict)
                print("[导出] 已加载微调后的权重")
            except Exception as e:
                print(f"[警告] 加载微调权重失败，使用预训练权重: {e}")
        
        if _export_encoder_to_onnx(encoder_root, Path(args.onnx_output), sample_rate=args.sample_rate, classifier=cpu_encoder):
            print(f"[导出] ONNX模型: {Path(args.onnx_output).name}")

    logger.info("=" * 80)
    logger.info("[完成] 训练完成")
    logger.info(f"输出目录: {run_dir}")
    logger.info(f"说话人嵌入向量: {model_path.name}")
    logger.info(f"包含 {len(speaker_embeddings)} 个目标员工模板")
    if training_result.history:
        final_metrics = training_result.history[-1]
        logger.info(f"最终训练指标: Loss={final_metrics.get('train_loss', 0):.4f}, Acc={final_metrics.get('train_acc', 0)*100:.2f}%")
        if 'val_loss' in final_metrics:
            logger.info(f"最终验证指标: Loss={final_metrics.get('val_loss', 0):.4f}, Acc={final_metrics.get('val_acc', 0)*100:.2f}%")
    logger.info("=" * 80)

    print(f"[完成] 训练完成 | 输出目录: {run_dir}")


if __name__ == "__main__":
    main()
