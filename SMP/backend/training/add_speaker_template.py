#!/usr/bin/env python3
"""
添加新员工模板到现有的 speaker_embeddings.json 文件

用法:
    # 添加单个音频文件
    python add_speaker_template.py --speaker-id emp_003 --audio new_employee.wav

    # 添加文件夹中的所有音频文件
    python add_speaker_template.py --speaker-id emp_003 --audio-dir new_employee_recordings/

    # 指定模型文件路径
    python add_speaker_template.py --speaker-id emp_003 --audio-dir recordings/ --model models/speechbrain_model_XXX/speaker_embeddings.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# 导入训练脚本中的工具函数
SPEECHBRAIN_DEFAULT_DIR = Path(__file__).resolve().parent / "model" / "speechbrain"
REQUIRED_ENCODER_FILES = {
    "hyperparams.yaml",
    "classifier.ckpt",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.txt",
}


def _ensure_encoder_assets(root: Path) -> None:
    """确保encoder资源文件存在"""
    if not root.exists():
        raise FileNotFoundError(
            f"SpeechBrain encoder assets not found at {root}. "
            f"Place the downloaded files in this folder or pass --encoder-root."
        )
    missing = [name for name in REQUIRED_ENCODER_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"SpeechBrain encoder root {root} is missing required files: {missing}."
        )


def _load_encoder(device: str = "cpu", encoder_root: Optional[Path] = None) -> EncoderClassifier:
    """加载SpeechBrain encoder"""
    root = Path(encoder_root) if encoder_root else SPEECHBRAIN_DEFAULT_DIR
    root = root.expanduser().resolve()
    _ensure_encoder_assets(root)
    
    sid = EncoderClassifier.from_hparams(
        source=str(root),
        savedir=str(root),
        run_opts={"device": device},
    )
    sid.eval()
    return sid


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """确保音频是单声道"""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _resample_audio(audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    """重采样音频"""
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
    """准备音频波形"""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio = _ensure_mono(audio)
    audio = _resample_audio(audio, sr, sample_rate)
    return audio.astype(np.float32)


def _create_resampler(sample_rate: int) -> Optional[torchaudio.transforms.Resample]:
    """创建重采样器（如果需要）"""
    if sample_rate == 16000:
        return None
    return torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)


def _compute_embedding_from_waveform(
    waveform: np.ndarray,
    encoder: EncoderClassifier,
    resampler: Optional[torchaudio.transforms.Resample],
    device: torch.device,
) -> np.ndarray:
    """从波形计算嵌入向量"""
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    if resampler is not None:
        tensor = resampler(tensor)
    tensor = tensor.to(device)
    with torch.no_grad():
        embedding = encoder.encode_batch(tensor)
    embedding = embedding.squeeze().cpu().numpy()
    if embedding.ndim > 1:
        embedding = embedding.reshape(-1)
    return embedding.astype(np.float32)


def _find_latest_model() -> Optional[Path]:
    """查找最新的 speaker_embeddings.json 文件"""
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    if not models_dir.exists():
        return None
    
    pattern = "speechbrain_model_*/speaker_embeddings.json"
    matches = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _collect_audio_files(input_path: Path) -> List[Path]:
    """收集音频文件（支持单个文件或文件夹）"""
    audio_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() == ".wav":
            audio_files.append(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}，请使用WAV文件")
    elif input_path.is_dir():
        # 递归查找所有WAV文件
        audio_files = sorted(input_path.rglob("*.wav"))
        if not audio_files:
            raise ValueError(f"目录中没有找到WAV文件: {input_path}")
    else:
        raise ValueError(f"路径不存在: {input_path}")
    
    return audio_files


def add_speaker_template(
    speaker_id: str,
    audio_files: List[Path],
    model_path: Path,
    encoder_root: Optional[Path] = None,
    sample_rate: int = 16000,
    device: str = "cpu",
) -> None:
    """
    添加新员工模板到现有的 speaker_embeddings.json 文件
    
    Args:
        speaker_id: 新员工ID（如 "emp_003"）
        audio_files: 新员工的音频文件列表
        model_path: speaker_embeddings.json 文件路径
        encoder_root: SpeechBrain encoder路径（可选）
        sample_rate: 采样率（默认16000）
        device: 设备（默认"cpu"）
    """
    print("=" * 60)
    print(f"添加新员工模板: {speaker_id}")
    print("=" * 60)
    
    # 验证员工ID格式
    if not speaker_id.startswith("emp_"):
        print(f"⚠️  警告: 员工ID '{speaker_id}' 不以 'emp_' 开头，建议使用 'emp_XXX' 格式")
    
    # 检查模型文件是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载现有模板
    print(f"\n[1/4] 加载现有模板: {model_path}")
    with open(model_path, 'r', encoding='utf-8') as f:
        speaker_embeddings = json.load(f)
    
    # 检查是否已存在
    if speaker_id in speaker_embeddings:
        response = input(f"⚠️  员工 '{speaker_id}' 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("已取消操作")
            return
        print(f"将覆盖现有模板: {speaker_id}")
    
    print(f"当前模板数量: {len(speaker_embeddings)}")
    print(f"当前员工: {list(speaker_embeddings.keys())}")
    
    # 加载encoder
    print(f"\n[2/4] 加载SpeechBrain encoder...")
    encoder = _load_encoder(device, encoder_root)
    resampler = _create_resampler(sample_rate)
    device_obj = torch.device(device)
    
    # 提取嵌入向量
    print(f"\n[3/4] 从 {len(audio_files)} 个音频文件提取嵌入向量...")
    embeddings = []
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"  处理 [{i}/{len(audio_files)}]: {audio_file.name}")
            waveform = _prepare_waveform(audio_file, sample_rate)
            embedding = _compute_embedding_from_waveform(waveform, encoder, resampler, device_obj)
            embeddings.append(embedding)
        except Exception as e:
            print(f"  ⚠️  跳过文件 {audio_file.name}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("没有成功提取任何嵌入向量，请检查音频文件")
    
    # 计算平均嵌入向量（模板）
    print(f"\n[4/4] 计算模板（平均嵌入向量）...")
    template = np.mean(embeddings, axis=0).tolist()
    print(f"  使用 {len(embeddings)} 个嵌入向量")
    print(f"  模板维度: {len(template)}")
    
    # 添加到模型文件
    speaker_embeddings[speaker_id] = template
    
    # 保存更新后的模型文件
    print(f"\n保存更新后的模型文件...")
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_embeddings, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print(f"✅ 成功添加新员工模板: {speaker_id}")
    print(f"   模型文件: {model_path}")
    print(f"   模板数量: {len(speaker_embeddings)}")
    print(f"   所有员工: {list(speaker_embeddings.keys())}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="添加新员工模板到现有的 speaker_embeddings.json 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加单个音频文件
  python add_speaker_template.py --speaker-id emp_003 --audio new_employee.wav

  # 添加文件夹中的所有音频文件
  python add_speaker_template.py --speaker-id emp_003 --audio-dir new_employee_recordings/

  # 指定模型文件路径
  python add_speaker_template.py --speaker-id emp_003 --audio-dir recordings/ \\
      --model models/speechbrain_model_XXX/speaker_embeddings.json
        """
    )
    
    # 员工ID（必需）
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="新员工ID（如 emp_003）"
    )
    
    # 音频输入（二选一）
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--audio",
        type=Path,
        help="单个音频文件路径"
    )
    audio_group.add_argument(
        "--audio-dir",
        type=Path,
        dest="audio_dir",
        help="包含音频文件的文件夹路径（会递归查找所有WAV文件）"
    )
    
    # 模型文件（可选，自动查找最新的）
    parser.add_argument(
        "--model",
        type=Path,
        help="speaker_embeddings.json 文件路径（如果不指定，会自动查找最新的）"
    )
    
    # 其他选项
    parser.add_argument(
        "--encoder-root",
        type=Path,
        help="SpeechBrain encoder路径（默认: model/speechbrain）"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="采样率（默认: 16000）"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="设备（默认: cpu）"
    )
    
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 确定音频文件
    if args.audio:
        audio_files = [args.audio]
    else:
        audio_files = _collect_audio_files(args.audio_dir)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 确定模型文件路径
    if args.model:
        model_path = args.model.resolve()
    else:
        model_path = _find_latest_model()
        if model_path is None:
            raise FileNotFoundError(
                "未找到 speaker_embeddings.json 文件。"
                "请使用 --model 参数指定文件路径，或先运行训练脚本生成模型文件。"
            )
        print(f"自动找到最新的模型文件: {model_path}")
    
    # 添加模板
    add_speaker_template(
        speaker_id=args.speaker_id,
        audio_files=audio_files,
        model_path=model_path,
        encoder_root=args.encoder_root,
        sample_rate=args.sample_rate,
        device=args.device,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

