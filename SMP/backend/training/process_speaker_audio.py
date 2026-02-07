#!/usr/bin/env python3
"""
将原始音频文件处理为Speaker-ID-Lite模型所需的10-20秒声纹片段
要求：
- 音频时长：10-20秒/段
- 样本数量：3-20段/人
- 格式：16kHz/16bit/单声道WAV
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="处理原始音频为Speaker-ID-Lite声纹数据集")
    parser.add_argument("--input_dir", default="/home/swufe/Project/zhoulonghao/data/emp_002.wav", help="原始音频文件目录")
    parser.add_argument("--output_dir", default="/home/swufe/Project/zhoulonghao/code/software/training/raw_data/speaker/data", help="输出目录")
    parser.add_argument("--sample_rate", type=int, default=16000, help="目标采样率")
    parser.add_argument("--bit_depth", type=int, default=16, help="目标位深")
    parser.add_argument("--channels", type=int, default=1, help="目标声道数")
    parser.add_argument("--min_duration", type=int, default=10, help="每段最小时长(秒)")
    parser.add_argument("--max_duration", type=int, default=20, help="每段最大时长(秒)")
    parser.add_argument("--min_segments", type=int, default=5, help="每人最小片段数量")
    parser.add_argument("--max_segments", type=int, default=20, help="每人最大片段数量")
    parser.add_argument("--silence_threshold", type=float, default=0.01, help="静音阈值")
    return parser.parse_args()


def convert_audio(input_path, sample_rate, bit_depth, channels):
    """转换音频格式"""
    # 使用dtype="float32"直接读取为float32格式
    data, sr = sf.read(input_path, dtype="float32")
    
    original_duration = len(data) / sr
    print(f"  原始采样率: {sr} Hz, 原始时长: {original_duration:.2f}秒, 采样点数: {len(data)}")
    
    # 转换声道数
    if len(data.shape) > 1 and channels == 1:
        data = data.mean(axis=1)
    
    # 如果数据已经是int16/int32格式（soundfile可能返回），转换为float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # 转换采样率（在归一化之前，避免影响重采样质量）
    if sr != sample_rate:
        print(f"  进行重采样: {sr} Hz -> {sample_rate} Hz")
        # 使用线性插值重采样，与项目中其他文件保持一致
        # 计算原始音频的时长（秒）
        duration = len(data) / sr
        # 计算目标采样点数（保持相同时长）
        target_len = int(np.round(duration * sample_rate))
        if target_len > 0:
            # 创建原始采样点的时间轴（每个采样点对应的时间戳）
            x_original = np.arange(len(data)) / sr
            # 创建目标采样点的时间轴（每个采样点对应的时间戳）
            x_target = np.arange(target_len) / sample_rate
            # 使用线性插值重采样
            data = np.interp(x_target, x_original, data)
            print(f"  重采样完成: 目标采样点数: {target_len}, 目标时长: {len(data)/sample_rate:.2f}秒")
        else:
            print(f"  警告: 重采样后目标长度 <= 0，跳过重采样")
    else:
        print(f"  采样率已匹配，无需重采样")
    
    # 归一化到-1.0到1.0范围（在所有处理完成后）
    data_max = np.abs(data).max()
    if data_max > 1.0:
        data = data / data_max
    elif data_max == 0.0:
        # 如果音频全为零，返回原样
        pass
    
    # 保持float32格式
    final_duration = len(data) / sample_rate
    print(f"  最终采样率: {sample_rate} Hz, 最终时长: {final_duration:.2f}秒, 采样点数: {len(data)}")
    return data.astype(np.float32), sample_rate


def remove_silence(audio, sample_rate, silence_threshold):
    """去除静音部分"""
    # 确保音频是float32格式
    if audio.dtype != np.float32:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
    
    # 使用更智能的静音检测：基于RMS能量，而不是简单的幅度阈值
    # 计算窗口RMS（避免误删低音量但有用的音频）
    window_size = int(0.02 * sample_rate)  # 20ms窗口
    if window_size < 1:
        window_size = 1
    
    # 计算每个窗口的RMS
    rms_values = []
    for i in range(0, len(audio), window_size):
        window = audio[i:i+window_size]
        if len(window) > 0:
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
    
    if len(rms_values) == 0:
        return audio.astype(np.float32)
    
    # 使用RMS阈值，比简单的幅度阈值更准确
    rms_threshold = silence_threshold * 0.5  # RMS阈值通常比幅度阈值小
    rms_array = np.array(rms_values)
    
    # 找到非静音窗口
    non_silent_windows = rms_array > rms_threshold
    
    # 如果所有窗口都是静音，返回原样（避免完全删除音频）
    if not np.any(non_silent_windows):
        return audio.astype(np.float32)
    
    # 找到第一个和最后一个非静音窗口
    non_silent_indices = np.where(non_silent_windows)[0]
    if len(non_silent_indices) == 0:
        return audio.astype(np.float32)
    
    first_non_silent = non_silent_indices[0] * window_size
    last_non_silent = (non_silent_indices[-1] + 1) * window_size
    
    # 保留非静音部分，并添加一些前后缓冲（避免切掉有用的音频）
    buffer_samples = int(0.1 * sample_rate)  # 100ms缓冲
    start_idx = max(0, first_non_silent - buffer_samples)
    end_idx = min(len(audio), last_non_silent + buffer_samples)
    
    audio = audio[start_idx:end_idx]
    
    return audio.astype(np.float32)


def segment_audio(audio, sample_rate, min_duration, max_duration, min_segments, max_segments):
    """将音频分割为10-20秒的片段"""
    min_samples = int(min_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)
    
    # 计算每段的最佳长度
    total_samples = len(audio)
    if total_samples < min_samples:
        return []
    
    # 计算可能的段数
    possible_segments = total_samples // min_samples
    target_segments = min(max_segments, max(min_segments, possible_segments))
    
    # 计算每段长度
    segment_length = total_samples // target_segments
    segment_length = min(max_samples, max(min_samples, segment_length))
    
    # 分割音频
    segments = []
    for i in range(target_segments):
        start = i * segment_length
        end = start + segment_length
        if end > total_samples:
            break
        segments.append(audio[start:end])
    
    return segments


def save_segments(segments, speaker_name, session_root, sample_rate, bit_depth):
    """保存音频片段到以原始文件名命名的文件夹"""
    speaker_dir = session_root / speaker_name
    speaker_dir.mkdir(exist_ok=True)

    saved_files = []
    for idx, segment in enumerate(segments, start=1):
        segment_path = speaker_dir / f"{speaker_name}_{idx:02d}.wav"
        
        # 确保segment是float32格式
        if segment.dtype != np.float32:
            segment = segment.astype(np.float32)
        
        # 只对超出范围的数据进行归一化，避免过度归一化导致音量变化
        segment_max = np.abs(segment).max()
        if segment_max > 1.0:
            # 如果超出范围，归一化
            segment = segment / segment_max
        elif segment_max > 0.0 and segment_max < 0.01:
            # 如果音频太安静，轻微放大（但不超过1.0）
            # 这有助于保持音频的原始动态范围
            pass  # 保持原样，不放大
        
        # 使用float32格式保存，soundfile会自动处理格式转换
        # 指定subtype为PCM_16或PCM_32，soundfile会自动将float32转换为对应的整数格式
        sf.write(segment_path, segment, sample_rate, subtype=f"PCM_{bit_depth}")
        saved_files.append(segment_path)

    return saved_files, speaker_dir


def write_speaker_config(session_root: Path, speaker_dir: Path, speaker_name: str, pretrained_model: str) -> Path:
    """在与音频文件夹相同的目录下生成配置文件"""
    config = {
        "model": {
            "pretrained_path": pretrained_model,
            "output_path": str(speaker_dir / "speaker_id_hotpot.model"),
            "layers_to_train": "voiceprint_feature",
            "freeze_core_layers": True,
        },
        "training": {
            "speaker_dirs": [str(speaker_dir)],
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 5e-6,
            "segment_duration": 15,
        },
        "validation": {
            "verification_threshold": 0.85,
            "equal_error_rate_target": 0.05,
        },
    }

    config_path = session_root / f"{speaker_name}_config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return config_path


def process_speaker_audio(input_path, output_dir, sample_rate, bit_depth, channels, 
                          min_duration, max_duration, min_segments, max_segments, silence_threshold):
    """处理所有说话人音频"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_name = input_path.stem if input_path.is_file() else input_path.name
    session_root = output_dir / session_name
    session_root.mkdir(parents=True, exist_ok=True)
    
    # 获取所有WAV文件列表
    all_saved_files = []
    
    # 处理单个文件或目录
    wav_files = []
    print(f"输入路径: {input_path}")
    print(f"是否为文件: {input_path.is_file()}")
    print(f"是否为目录: {input_path.is_dir()}")
    print(f"文件后缀: {input_path.suffix}")
    
    if input_path.is_file():
        if input_path.suffix.lower() == ".wav":
            # 处理单个WAV文件
            speaker_name = input_path.stem
            wav_files = [(speaker_name, [input_path])]
        else:
            print(f"警告: {input_path} 不是WAV文件，跳过处理")
            return all_saved_files
    elif input_path.is_dir():
        # 处理目录中的所有WAV文件
        direct_wavs = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
        print(f"找到 {len(direct_wavs)} 个直接WAV文件")
        
        if direct_wavs:
            # 目录中直接包含WAV文件，为每个WAV文件创建一个说话人
            wav_files = []
            for wav_file in direct_wavs:
                # 使用文件名前缀作为说话人名称
                speaker_name = wav_file.stem
                wav_files.append((speaker_name, [wav_file]))
        else:
            # 检查是否有子目录
            subdirs = [d for d in input_path.iterdir() if d.is_dir()]
            print(f"找到 {len(subdirs)} 个子目录")
            
            for subdir in subdirs:
                # 获取子目录中的WAV文件
                subdir_wavs = list(subdir.glob("*.wav")) + list(subdir.glob("*.WAV"))
                if subdir_wavs:
                    # 使用子目录名作为说话人名称
                    speaker_name = subdir.name
                    wav_files.append((speaker_name, subdir_wavs))
                    print(f"- 子目录 {speaker_name}: {len(subdir_wavs)} 个WAV文件")
    else:
        print(f"错误: {input_path} 不是有效的文件或目录")
        return all_saved_files
    
    for speaker_name, speaker_wavs in wav_files:
        print(f"\n处理说话人: {speaker_name}")
        print(f"包含 {len(speaker_wavs)} 个音频文件")
        
        # 合并所有该说话人的音频
        all_audio = []
        for wav_file in speaker_wavs:
            print(f"- 处理文件: {wav_file.name}")
            
            # 转换音频格式
            audio, sr = convert_audio(wav_file, sample_rate, bit_depth, channels)
            
            # 去除静音
            audio = remove_silence(audio, sample_rate, silence_threshold)
            
            if len(audio) > 0:
                all_audio.append(audio)
            else:
                print(f"  警告: {wav_file.name} 去除静音后为空，跳过")
        
        if not all_audio:
            print(f"警告: 说话人 {speaker_name} 没有有效音频，跳过")
            continue
        
        # 合并所有音频
        merged_audio = np.concatenate(all_audio)
        print(f"合并后总时长: {len(merged_audio)/sample_rate:.2f}秒")
        
        # 分割音频为10-20秒片段
        segments = segment_audio(merged_audio, sample_rate, min_duration, max_duration, min_segments, max_segments)
        if len(segments) > max_segments:
            segments = segments[:max_segments]
        
        if not segments:
            print(f"警告: 说话人 {speaker_name} 无法分割为足够的片段")
            continue
        
        # 保存片段
        saved_files, speaker_dir = save_segments(segments, speaker_name, session_root, sample_rate, bit_depth)
        all_saved_files.extend(saved_files)
        # 构建模型相对路径
        project_root = Path(__file__).resolve().parent
        default_model = project_root / "software" / "training" / "model" / "speaker" / "speaker_id_lite.onnx"
        # 直接使用绝对路径，避免相对路径可能带来的问题
        pretrained_model = str(default_model)
        
        config_path = write_speaker_config(
            session_root,
            speaker_dir,
            speaker_name,
            pretrained_model,
        )

        print(f"成功生成 {len(segments)} 个音频片段，保存到: {speaker_dir}")
        print(f"生成配置文件: {config_path}")
    
    return all_saved_files


def main():
    args = parse_args()
    
    saved_files = process_speaker_audio(
        args.input_dir,
        args.output_dir,
        args.sample_rate,
        args.bit_depth,
        args.channels,
        args.min_duration,
        args.max_duration,
        args.min_segments,
        args.max_segments,
        args.silence_threshold
    )
    
    print(f"\n处理完成！")
    print(f"总共生成 {len(saved_files)} 个声纹片段")
    print(f"输出目录: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
