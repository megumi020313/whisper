"""音频处理工具"""
from __future__ import annotations

import io
import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any

from backend.core.exceptions import AudioFormatError
from backend.utils.logger import get_logger
from backend.utils.constants import (
    STANDARD_SAMPLE_RATE,
    MIN_SNR_DB,
)

logger = get_logger()

# 背景噪音缓存（动态加噪增强注册）
NOISE_FILE_PATH = Path(__file__).resolve().parents[2] / "data" / "noise" / "restaurant_bg.wav"
_BG_NOISE_BUFFER = None


def check_audio_energy(audio_numpy: np.ndarray, threshold_db: float = -25.0) -> bool:
    """
    检查音频RMS能量是否超过阈值
    
    Args:
        audio_numpy: 音频数据数组
        threshold_db: 能量阈值（dB），默认-25.0，低于此值视为静音/远端噪声
    
    Returns:
        True表示能量足够，False表示静音
    """
    if audio_numpy.size == 0:
        return False
    rms = np.sqrt(np.mean(np.square(audio_numpy)))
    db = 20 * np.log10(rms + 1e-9)
    return db >= threshold_db


def calculate_snr(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    计算音频信噪比（SNR）
    
    使用简化算法：
    1. 将音频分为多个帧
    2. 计算每帧的能量
    3. 将能量最低的20%帧视为噪声
    4. 将能量最高的50%帧视为信号
    5. SNR = 10 * log10(信号功率 / 噪声功率)
    
    Args:
        audio: 音频数据数组
        sample_rate: 采样率
        
    Returns:
        信噪比（dB），范围通常在-10到40之间
    """
    if audio.size == 0:
        return -np.inf
    
    # 分帧：每帧20ms
    frame_length = int(0.02 * sample_rate)
    if len(audio) < frame_length:
        # 音频太短，直接计算整体RMS
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms < 1e-9:
            return -np.inf
        return 20.0  # 返回一个默认值
    
    # 计算每帧的能量
    num_frames = len(audio) // frame_length
    frames = audio[:num_frames * frame_length].reshape(num_frames, frame_length)
    frame_energy = np.mean(np.square(frames), axis=1)
    
    # 排序能量
    sorted_energy = np.sort(frame_energy)
    
    # 噪声估计：能量最低的20%
    noise_threshold_idx = max(1, int(num_frames * 0.2))
    noise_energy = np.mean(sorted_energy[:noise_threshold_idx])
    
    # 信号估计：能量最高的50%
    signal_threshold_idx = int(num_frames * 0.5)
    signal_energy = np.mean(sorted_energy[signal_threshold_idx:])
    
    # 计算SNR
    if noise_energy < 1e-9:
        return 40.0  # 噪声极低，返回高SNR
    
    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


def _load_noise_buffer() -> None:
    """
    将背景噪音加载至内存，缺失时回退为空数组。
    
    该函数使用全局变量缓存背景噪音，避免重复加载。
    如果噪音文件不存在或加载失败，会创建一个空数组作为回退。
    """
    global _BG_NOISE_BUFFER
    if _BG_NOISE_BUFFER is not None:
        return

    try:
        if NOISE_FILE_PATH.exists():
            data, sr = sf.read(NOISE_FILE_PATH, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            _BG_NOISE_BUFFER = data
            logger.info(f"Background noise loaded from {NOISE_FILE_PATH}, {len(data) / sr:.2f}s")
        else:
            logger.warning(f"Noise file not found at {NOISE_FILE_PATH}")
            _BG_NOISE_BUFFER = np.zeros(16000, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - 防御性回退
        logger.error(f"Failed to load noise buffer: {exc}")
        _BG_NOISE_BUFFER = np.zeros(16000, dtype=np.float32)


def augment_with_noise(clean_audio: np.ndarray, min_factor: float = 0.1, max_factor: float = 0.3) -> np.ndarray:
    """
    动态加噪：将音频与背景噪音按随机比例混合，增强鲁棒性
    
    Args:
        clean_audio: 纯净音频数据
        min_factor: 最小噪音因子，默认0.1（近场场景降低混合比例）
        max_factor: 最大噪音因子，默认0.3
    
    Returns:
        增强后的音频数据（加噪后的混合音频）
    """
    if clean_audio is None or clean_audio.size == 0:
        return clean_audio

    _load_noise_buffer()
    noise_buf = _BG_NOISE_BUFFER
    if noise_buf is None or noise_buf.size == 0:
        return clean_audio

    noise_len = len(noise_buf)
    audio_len = len(clean_audio)

    if noise_len >= audio_len:
        start = random.randint(0, max(0, noise_len - audio_len))
        noise_segment = noise_buf[start:start + audio_len]
    else:
        tile_count = int(np.ceil(audio_len / noise_len))
        noise_segment = np.tile(noise_buf, tile_count)[:audio_len]

    noise_factor = random.uniform(min_factor, max_factor)

    rms_clean = np.sqrt(np.mean(clean_audio ** 2)) + 1e-9
    rms_noise = np.sqrt(np.mean(noise_segment ** 2)) + 1e-9

    target_noise_rms = rms_clean * 0.8
    noise_segment = noise_segment * (target_noise_rms / rms_noise)

    mixed = (1.0 - noise_factor) * clean_audio + noise_factor * noise_segment
    return mixed.astype(np.float32)


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        audio_path: 音频文件路径
        sample_rate: 目标采样率
        mono: 是否转换为单声道
        
    Returns:
        (audio_data, sample_rate) 元组
        
    Raises:
        AudioFormatError: 音频加载失败
    """
    try:
        audio, sr = librosa.load(
            audio_path, 
            sr=sample_rate, 
            mono=mono
        )
        return audio, sr
    except Exception as e:
        raise AudioFormatError(
            f"Failed to load audio file: {str(e)}",
            {"path": str(audio_path)}
        )


def save_audio(
    audio: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int = 16000
) -> None:
    """
    保存音频文件
    
    Args:
        audio: 音频数据
        output_path: 输出路径
        sample_rate: 采样率
        
    Raises:
        AudioFormatError: 音频保存失败
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
    except Exception as e:
        raise AudioFormatError(
            f"Failed to save audio file: {str(e)}",
            {"path": str(output_path)}
        )


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    重采样音频
    
    Args:
        audio: 音频数据
        orig_sr: 原始采样率
        target_sr: 目标采样率
        
    Returns:
        重采样后的音频数据
        
    Raises:
        AudioFormatError: 重采样失败
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        raise AudioFormatError(
            f"Failed to resample audio: {str(e)}",
            {"orig_sr": orig_sr, "target_sr": target_sr}
        )


def validate_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_duration: Optional[float] = None,
    min_duration: float = 0.5
) -> bool:
    """
    验证音频格式
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        max_duration: 最大时长（秒）
        min_duration: 最小时长（秒）
        
    Returns:
        验证是否通过
        
    Raises:
        AudioFormatError: 音频格式不符合要求
    """
    if not isinstance(audio, np.ndarray):
        raise AudioFormatError(
            f"Audio must be numpy.ndarray, got {type(audio)}",
            {"type": str(type(audio))}
        )
    
    if audio.ndim > 1:
        raise AudioFormatError(
            f"Audio must be mono (1D), got {audio.ndim}D",
            {"ndim": audio.ndim}
        )
    
    if audio.size == 0:
        raise AudioFormatError("Audio is empty")
    
    # 计算时长
    duration = len(audio) / sample_rate
    
    if duration < min_duration:
        raise AudioFormatError(
            f"Audio too short: {duration:.2f}s < {min_duration}s",
            {"duration": duration, "min_duration": min_duration}
        )
    
    if max_duration and duration > max_duration:
        raise AudioFormatError(
            f"Audio too long: {duration:.2f}s > {max_duration}s",
            {"duration": duration, "max_duration": max_duration}
        )
    
    return True


def extract_audio_from_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    从字节流提取音频数据
    
    支持多种格式：WebM, Ogg, MP3, M4A, WAV等
    优先使用 soundfile，失败则使用 ffmpeg
    
    Args:
        audio_bytes: 音频字节流
        sample_rate: 目标采样率
        
    Returns:
        音频数据数组
        
    Raises:
        AudioFormatError: 音频解析失败
    """
    import io
    import tempfile
    import subprocess
    import shutil
    from pathlib import Path
    
    # 方法1: 尝试使用 soundfile 直接读取（支持WAV、FLAC、OGG等）
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 转换为单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # 重采样到目标采样率
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        
        # 确保是float32格式
        audio = audio.astype(np.float32)
        
        return audio
        
    except Exception as sf_error:
        # 方法2: 使用临时文件 + ffmpeg转换（支持所有格式，包括WebM）
        try:
            # 检查 ffmpeg 是否可用
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                # ffmpeg不可用，尝试用librosa
                try:
                    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
                    return audio
                except Exception as librosa_error:
                    raise AudioFormatError(
                        f"Failed to extract audio from bytes: soundfile={str(sf_error)}, librosa={str(librosa_error)}, ffmpeg not found"
                    )
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # 使用 ffmpeg 转换
                cmd = [
                    ffmpeg_path,
                    '-i', temp_input_path,
                    '-ar', str(sample_rate),  # 采样率
                    '-ac', '1',               # 单声道
                    '-sample_fmt', 's16',     # 16bit
                    '-y',                      # 覆盖输出文件
                    temp_output_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    raise Exception(f"ffmpeg转换失败: {result.stderr}")
                
                # 读取转换后的音频
                audio, sr = sf.read(temp_output_path)
                
                # 转换为单声道（以防万一）
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                
                # 确保是float32格式
                audio = audio.astype(np.float32)
                
                return audio
                
            finally:
                # 清理临时文件
                Path(temp_input_path).unlink(missing_ok=True)
                Path(temp_output_path).unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            raise AudioFormatError("音频转换超时（超过30秒）")
        except Exception as ffmpeg_error:
            raise AudioFormatError(
                f"Failed to extract audio from bytes: soundfile={str(sf_error)}, ffmpeg={str(ffmpeg_error)}"
            )


def decode_opus_stream(opus_bytes: bytes, expected_sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    使用 soundfile 解码 Ogg/Opus 字节流，返回 float32 PCM。
    
    Args:
        opus_bytes: Ogg/Opus 格式的音频字节流
        expected_sample_rate: 期望的采样率，默认16000Hz
        
    Returns:
        解码后的音频数据（float32格式），如果解码失败返回None
        
    Raises:
        ValueError: 当采样率不匹配时
    """
    try:
        data, samplerate = sf.read(io.BytesIO(opus_bytes), dtype="float32")

        if samplerate != expected_sample_rate:
            raise ValueError(f"Sample rate mismatch: {samplerate}, expected {expected_sample_rate}")

        if data.ndim > 1:
            data = data.mean(axis=1)

        return data

    except Exception as e:
        print(f"Opus decode failed: {e}")
        return None


def calculate_precise_intervals(
    raw_windows: List[Dict[str, Any]],
    target_uid: Optional[str],
    total_duration: float,
    stride: float = 0.1,
    min_gap: float = 0.3,
    score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    将滑窗识别结果映射为基于步长的精细时间段。

    注意：本函数参数较多（6个），建议使用关键字参数调用以提高可读性。

    Args:
        raw_windows: 滑动窗口识别结果列表，需包含 time_start/time_end/user_id。
        target_uid: 目标用户ID（通常为最终判决的用户）。
        total_duration: 音频总时长（秒）。
        stride: 时间分辨率（秒），建议与滑窗步长一致，例如0.1。
        min_gap: 允许填补的最小空隙（秒），用于掩码平滑，默认0.3。
        score: 可选，整体得分/相似度，用于标注结果。

    Returns:
        精细化片段列表，字段包含 time_start/time_end/duration/user_id/avg_similarity/window_count/recognized/source。
    """
    if not raw_windows or not target_uid or target_uid == "unknown":
        return []

    # 避免总时长被窗口尾部截断
    max_end = max((w.get("time_end", 0.0) for w in raw_windows), default=0.0)
    duration = max(float(total_duration), float(max_end))

    if stride <= 0:
        stride = 0.1

    num_steps = max(1, int(np.ceil(duration / stride)))
    mask = np.zeros(num_steps, dtype=bool)

    # 映射窗口到步长掩码（覆盖窗口区间，保证保守包含）
    for win in raw_windows:
        if win.get("user_id") != target_uid:
            continue
        start = float(win.get("time_start", 0.0))
        end = float(win.get("time_end", start))
        start_idx = max(0, int(start / stride))
        end_idx = int(np.ceil(end / stride))
        end_idx = min(num_steps, max(start_idx + 1, end_idx))
        mask[start_idx:end_idx] = True

    # 填补小空隙，减少抖动
    gap_steps = max(1, int(min_gap / stride))
    for i in range(1, len(mask) - 1):
        if mask[i]:
            continue
        if mask[i - 1] and np.any(mask[i + 1:i + 1 + gap_steps]):
            mask[i] = True

    # 将掩码转回时间区间
    intervals = []
    active = False
    start_idx = 0

    for idx, val in enumerate(mask):
        if val and not active:
            active = True
            start_idx = idx
        elif not val and active:
            active = False
            end_idx = idx
            intervals.append((start_idx * stride, end_idx * stride))

    if active:
        intervals.append((start_idx * stride, len(mask) * stride))

    precise_segments = []
    default_score = float(score) if score is not None else None

    for seg_start, seg_end in intervals:
        duration_sec = seg_end - seg_start
        if duration_sec < 0.2:  # 丢弃极短片段
            continue

        # 计算区间内窗口的相似度与数量
        overlap_windows = [
            w for w in raw_windows
            if w.get("user_id") == target_uid
            and w.get("time_end", 0.0) > seg_start
            and w.get("time_start", 0.0) < seg_end
        ]

        sims = [w.get("similarity", default_score or 0.0) for w in overlap_windows]
        window_count = len(overlap_windows)

        if not sims:
            sims = [default_score or 0.0]
        avg_sim = float(np.mean(sims)) if sims else None
        min_sim = float(np.min(sims)) if sims else None
        max_sim = float(np.max(sims)) if sims else None

        precise_segments.append({
            "time_start": round(seg_start, 3),
            "time_end": round(seg_end, 3),
            "duration": round(duration_sec, 3),
            "user_id": target_uid,
            "recognized": True,
            "avg_similarity": avg_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
            "window_count": window_count,
            "source": "timeline_mask"
        })

    # 编号
    for idx, seg in enumerate(precise_segments, 1):
        seg["segment_id"] = idx

    return precise_segments


def calculate_session_intervals(
    raw_segments: List[Dict[str, Any]],
    target_uid: str,
    max_silence_gap: float = 15.0,
    min_interruption_duration: float = 1.0,
    noise_floor: float = 0.20,
    interruption_rms_ratio: float = 0.6
) -> List[Dict[str, Any]]:
    """
    V7.0 近场优先会话分段：噪音底噪 + 近场能量比校验。
    
    注意：本函数参数较多（6个），建议使用关键字参数调用以提高可读性。
    
    Args:
        raw_segments: 原始片段列表，需包含 start/end/user/score/rms 字段。
        target_uid: 目标用户ID。
        max_silence_gap: 最大静音间隙（秒），超过此值会断开会话，默认15.0。
        min_interruption_duration: 最小插话时长（秒），用于判断是否打断会话，默认1.0。
        noise_floor: 噪音底噪阈值，低于此值的片段视为噪音，默认0.20。
        interruption_rms_ratio: 插话与目标的能量比下限，用于过滤远场干扰，默认0.6。
        
    Returns:
        会话片段列表，字段包含 start/end/user/score/average_score/stats/window_count。
    """
    if not raw_segments or not target_uid or target_uid == "unknown":
        return []

    raw_segments.sort(key=lambda x: x['start'])

    target_segments = [s for s in raw_segments if s.get('user') == target_uid]
    if not target_segments:
        return []

    # 计算目标基准能量（取前10个目标窗口的均值）
    sample_count = min(len(target_segments), 10)
    target_base_rms = float(np.mean([s.get('rms', 0.0) for s in target_segments[:sample_count]]))
    if target_base_rms <= 0:
        target_base_rms = 1e-3

    sessions = []
    current_session = {
        "start": float(target_segments[0]['start']),
        "end": float(target_segments[0]['end']),
        "scores": [float(target_segments[0].get('score', 0.0))],
        "window_count": 1
    }

    for i in range(1, len(target_segments)):
        curr = target_segments[i]
        prev_end = current_session['end']
        gap_duration = float(curr['start']) - prev_end

        should_break = False

        # 超时断开
        if gap_duration > max_silence_gap:
            should_break = True

        # 有效插话断开：只统计分数超过噪音底噪的非目标片段
        if not should_break and gap_duration > 0:
            interruption_len = 0.0

            for seg in raw_segments:
                if seg['end'] <= prev_end:
                    continue
                if seg['start'] >= curr['start']:
                    break

                if seg.get('user') == target_uid:
                    continue

                seg_score = float(seg.get('score', 0.0))
                seg_rms = float(seg.get('rms', 0.0))
                if seg_score <= noise_floor:
                    continue  # 低于噪音底噪，忽略

                if seg_rms <= target_base_rms * interruption_rms_ratio:
                    continue  # 能量不足，视为远场

                overlap_start = max(prev_end, seg['start'])
                overlap_end = min(curr['start'], seg['end'])
                overlap = max(0.0, overlap_end - overlap_start)
                interruption_len += overlap

            if interruption_len > min_interruption_duration:
                should_break = True

        if should_break:
            sessions.append(current_session)
            current_session = {
                "start": float(curr['start']),
                "end": float(curr['end']),
                "scores": [float(curr.get('score', 0.0))],
                "window_count": 1
            }
        else:
            current_session['end'] = max(current_session['end'], float(curr['end']))
            current_session['scores'].append(float(curr.get('score', 0.0)))
            current_session['window_count'] += 1

    sessions.append(current_session)

    final_results = []
    for sess in sessions:
        scores = sess.get('scores', []) or [0.0]
        avg_score = float(np.mean(scores)) if scores else 0.0
        final_results.append({
            "start": round(sess['start'], 2),
            "end": round(sess['end'], 2),
            "user": target_uid,
            "score": round(avg_score, 4),
            "average_score": round(avg_score, 4),
            "stats": {
                "min_score": round(float(np.min(scores)), 4),
                "max_score": round(float(np.max(scores)), 4),
                "avg_score": round(avg_score, 4),
                "window_count": sess.get('window_count', 0)
            },
            "window_count": sess.get('window_count', 0)
        })

    return final_results


def map_score_to_percentage(z_score: float, high_threshold: float = 3.5, low_threshold: float = 1.8) -> float:
    """将 Z-Score 映射到 0-100 的数值（人性化显示）。V12 映射规则。"""
    z = float(z_score)

    if z < low_threshold:
        # 在低阈值下使用较平滑线性映射，保证极低值映射到接近 0
        min_z = -2.0
        if z < min_z:
            z = min_z
        percent = 60 * (z - min_z) / (low_threshold - min_z)
        return float(max(0.0, min(60.0, percent)))

    if z > high_threshold:
        diff = z - high_threshold
        percent = 90 + 9.9 * (1 - 1 / (1 + 0.5 * diff))
        return float(min(99.9, percent))

    ratio = (z - low_threshold) / (high_threshold - low_threshold)
    percent = 60 + 30 * ratio
    return float(percent)


def calculate_diarization_intervals(
    raw_segments: List[Dict[str, Any]],
    total_duration: float,
    stride: float = 0.1,
    max_silence_gap: float = 15.0,
    identification_threshold: float = 1.8
) -> List[Dict[str, Any]]:
    """
    V12.0 多人全库识别 + 强连贯性算法

    逻辑：
    1. 遍历 raw_segments，根据分数判定为 'registered user' 或 'unknown'（使用 identification_threshold）。
    2. 将标签投票到时间网格。
    3. 强力平滑：如果前后是同一人且间隙小于 fill_gap_steps，则填补静音；合并同类段并去掉极短误报。
    
    Args:
        raw_segments: 原始片段列表，需包含 start/end/user/score 字段。
        total_duration: 音频总时长（秒）。
        stride: 时间分辨率（秒），默认0.1。
        max_silence_gap: 最大静音间隙（秒），用于合并相邻同类段，默认15.0。
        identification_threshold: 识别阈值（Z-Score），超过此值才认为是注册用户，默认1.8。
        
    Returns:
        说话人分段列表，字段包含 start/end/user/type/score/raw_z_score/window_count。
    """
    if not raw_segments:
        return []

    num_steps = int(np.ceil(total_duration / stride)) + 1
    if num_steps <= 0:
        return []

    # 初始化网格：默认 silence
    grid = ["silence"] * num_steps
    grid_scores = np.zeros(num_steps, dtype=float)

    # 填充投票网格（使用 raw_segments 中的 user 与 score）
    for seg in raw_segments:
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', 0.0))
        center_time = start + (end - start) / 2.0
        idx = int(center_time / stride)
        
        if idx < 0 or idx >= num_steps:
            continue
        
        user = seg.get('user')
        score = float(seg.get('score', 0.0))

        # 关键修复：在此处判定身份
        if score >= identification_threshold and user not in (None, 'unknown'):
            grid[idx] = user
            grid_scores[idx] = score
        else:
            # 不足阈值视为 unknown
            grid[idx] = 'unknown'
            grid_scores[idx] = score

    # 强力填补：如果前后都是同一人且间隙在 fill_gap_steps 内，则填补中间为该人
    fill_gap_steps = int(1.5 / stride)  # 默认1.5s
    for i in range(1, num_steps - 1):
        if grid[i] == 'silence':
            left = None
            for l in range(i - 1, max(-1, i - fill_gap_steps - 1), -1):
                if grid[l] != 'silence':
                    left = grid[l]
                    break
            right = None
            for r in range(i + 1, min(num_steps, i + fill_gap_steps + 1)):
                if grid[r] != 'silence':
                    right = grid[r]
                    break
            if left and right and left == right and left != 'unknown':
                grid[i] = left

    # 删除短片段（误报）：如果某人仅出现 <0.3s，则抹为 silence
    i = 0
    while i < num_steps:
        val = grid[i]
        j = i
        while j < num_steps and grid[j] == val:
            j += 1
        length = j - i
        if val != 'silence' and length < max(1, int(0.3 / stride)):
            for k in range(i, j):
                grid[k] = 'silence'
                grid_scores[k] = 0.0
        i = j

    # 转换为片段
    final_results = []
    current = grid[0]
    start_idx = 0
    current_scores = [grid_scores[0]] if grid_scores[0] != 0 else []
    for i in range(1, num_steps):
        val = grid[i]
        if val != current:
            _append_segment(final_results, current, start_idx, i, current_scores, stride)
            current = val
            start_idx = i
            current_scores = []
        if grid_scores[i] != 0:
            current_scores.append(grid_scores[i])
    _append_segment(final_results, current, start_idx, num_steps, current_scores, stride)

    # 合并相邻同类段并根据 max_silence_gap 决定是否合并
    merged = []
    if final_results:
        curr = final_results[0]
        for nxt in final_results[1:]:
            if curr['user'] == nxt['user']:
                gap = nxt['start'] - curr['end']
                should_merge = False
                if curr['user'] == 'silence':
                    should_merge = True
                elif curr['user'] == 'unknown':
                    should_merge = gap <= 2.0
                else:
                    should_merge = gap <= max_silence_gap

                if should_merge:
                    curr['end'] = nxt['end']
                    curr['scores'].extend(nxt['scores'])
                    curr['window_count'] += nxt['window_count']
                    continue
            merged.append(curr)
            curr = nxt
        merged.append(curr)

    # 格式化输出：map score -> percent，unknown/silence 显示 0
    output = []
    for seg in merged:
        scores = seg.get('scores', []) or []
        avg_z = float(np.mean(scores)) if scores else 0.0
        user = seg.get('user')
        if user == 'silence':
            pct = 0.0
            type_str = 'silence'
        elif user == 'unknown':
            pct = 0.0
            type_str = 'other'
        else:
            pct = map_score_to_percentage(avg_z, high_threshold=3.5, low_threshold=1.8)
            type_str = 'target'
        output.append({
            'start': round(seg['start'], 2),
            'end': round(seg['end'], 2),
            'user': user,
            'type': type_str,
            'score': round(pct, 1),
            'raw_z_score': round(avg_z, 2),
            'window_count': seg['window_count']
        })

    return output


def _append_segment(
    results: List[Dict[str, Any]], 
    user: str, 
    start_idx: int, 
    end_idx: int, 
    scores: List[float], 
    stride: float
) -> None:
    """
    将片段追加到结果列表（内部辅助函数）
    
    Args:
        results: 结果列表，会被修改
        user: 用户ID或标签
        start_idx: 起始索引
        end_idx: 结束索引
        scores: 分数列表
        stride: 时间步长（秒）
    """
    results.append({
        'start': start_idx * stride,
        'end': end_idx * stride,
        'user': user,
        'scores': list(scores) if scores else [],
        'window_count': max(0, end_idx - start_idx)
    })
