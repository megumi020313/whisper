"""VAD 隔离测试脚本

【功能说明】
用于诊断 Silero VAD 是否正确切分音频片段，验证：
- VAD模型加载
- 语音活动检测
- 音频片段切分
- VAD参数配置（阈值、最小语音时长、最小静音时长）

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_vad_isolation.py

【测试音频】
需要在代码中指定音频文件路径（audio_path参数）
可选：指定VAD阈值（threshold参数）

【预期输出】
- VAD配置参数
- 音频加载信息
- 检测到的语音片段列表（起止时间、时长）
- VAD检测统计（总片段数、总时长、覆盖率）
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.vad import SileroVAD
from backend.utils.audio_utils import load_audio
from backend.common.config import get_config


def test_vad_isolation(audio_path: str, threshold: float = None):
    """
    隔离测试 VAD 模块
    
    Args:
        audio_path: 音频文件路径
        threshold: VAD 阈值（可选，不指定则使用配置文件值）
    """
    print("=" * 80)
    print("VAD 隔离测试")
    print("=" * 80)
    
    # 1. 加载配置
    config = get_config()
    vad_threshold = threshold if threshold is not None else config.vad_threshold
    
    print(f"\n【配置参数】")
    print(f"  VAD 阈值: {vad_threshold}")
    print(f"  最小语音时长: {config.min_speech_duration_ms}ms")
    print(f"  最小静音时长: {config.min_silence_duration_ms}ms")
    print(f"  语音填充: {config.speech_pad_ms}ms")
    
    # 2. 加载音频
    print(f"\n【加载音频】")
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"  ❌ 音频文件不存在: {audio_path}")
        return False
    
    audio, sr = load_audio(audio_path, sample_rate=16000)
    duration = len(audio) / sr
    print(f"  ✓ 音频加载成功")
    print(f"  - 文件: {audio_file.name}")
    print(f"  - 采样率: {sr} Hz")
    print(f"  - 时长: {duration:.2f}s")
    print(f"  - 样本数: {len(audio)}")
    
    # 3. 初始化 VAD
    print(f"\n【初始化 VAD】")
    vad = SileroVAD()
    print(f"  ✓ VAD 模型加载成功")
    
    # 4. 执行 VAD 检测
    print(f"\n【执行 VAD 检测】")
    print(f"  使用阈值: {vad_threshold}")
    
    speech_timestamps = vad.get_speech_timestamps(
        audio=audio,
        sample_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=config.min_speech_duration_ms,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms
    )
    
    # 5. 分析结果
    print(f"\n【VAD 检测结果】")
    print(f"  检测到的语音片段数: {len(speech_timestamps)}")
    
    if len(speech_timestamps) == 0:
        print(f"  ❌ 未检测到任何语音片段！")
        print(f"  建议: 降低 VAD 阈值（当前: {vad_threshold}）")
        return False
    
    if len(speech_timestamps) == 1:
        seg = speech_timestamps[0]
        start_time = seg['start'] / sr
        end_time = seg['end'] / sr
        seg_duration = end_time - start_time
        
        print(f"  ⚠️  只检测到 1 个片段（可能是 VAD 失效）")
        print(f"  - 起止时间: {start_time:.3f}s - {end_time:.3f}s")
        print(f"  - 时长: {seg_duration:.3f}s")
        print(f"  - 覆盖率: {seg_duration/duration*100:.1f}%")
        
        if seg_duration / duration > 0.95:
            print(f"\n  ❌ VAD 失效！片段覆盖了几乎整个音频")
            print(f"  问题: VAD 阈值过低，无法找到静音切分点")
            print(f"  建议: 提高 VAD 阈值（当前: {vad_threshold} → 建议: 0.4-0.5）")
            return False
    
    # 6. 详细片段信息
    print(f"\n【片段详情】")
    total_speech_duration = 0
    
    for i, seg in enumerate(speech_timestamps, 1):
        start_time = seg['start'] / sr
        end_time = seg['end'] / sr
        seg_duration = end_time - start_time
        total_speech_duration += seg_duration
        
        # 计算与前一片段的间隔
        if i > 1:
            prev_seg = speech_timestamps[i-2]
            prev_end = prev_seg['end'] / sr
            gap = start_time - prev_end
            gap_str = f" (间隔: {gap:.3f}s)"
        else:
            gap_str = ""
        
        print(f"  片段 {i:2d}: {start_time:7.3f}s - {end_time:7.3f}s (时长: {seg_duration:.3f}s){gap_str}")
        
        # 只显示前20个片段，避免输出过长
        if i == 20 and len(speech_timestamps) > 20:
            print(f"  ... (省略 {len(speech_timestamps) - 20} 个片段)")
            break
    
    # 7. 统计信息
    print(f"\n【统计信息】")
    print(f"  总片段数: {len(speech_timestamps)}")
    print(f"  总语音时长: {total_speech_duration:.2f}s")
    print(f"  音频总时长: {duration:.2f}s")
    print(f"  语音占比: {total_speech_duration/duration*100:.1f}%")
    
    if len(speech_timestamps) > 1:
        durations = [(seg['end'] - seg['start']) / sr for seg in speech_timestamps]
        avg_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        
        print(f"  平均片段时长: {avg_duration:.3f}s")
        print(f"  最短片段: {min_duration:.3f}s")
        print(f"  最长片段: {max_duration:.3f}s")
    
    # 8. 评估结果
    print(f"\n【评估结果】")
    if len(speech_timestamps) >= 5:
        print(f"  ✅ VAD 工作正常！检测到 {len(speech_timestamps)} 个片段")
        print(f"  智能融合逻辑可以正常工作")
        return True
    elif len(speech_timestamps) >= 2:
        print(f"  ⚠️  片段数较少（{len(speech_timestamps)} 个）")
        print(f"  可能需要微调 VAD 参数")
        return True
    else:
        print(f"  ❌ VAD 失效！只有 {len(speech_timestamps)} 个片段")
        return False
    
    print("=" * 80)


def test_multiple_thresholds(audio_path: str):
    """测试多个 VAD 阈值，找到最佳值"""
    print("\n" + "=" * 80)
    print("多阈值测试")
    print("=" * 80)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    results = []
    
    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}")
        print("-" * 80)
        
        # 加载音频
        audio, sr = load_audio(audio_path, sample_rate=16000)
        
        # 初始化 VAD
        vad = SileroVAD()
        config = get_config()
        
        # 执行检测
        speech_timestamps = vad.get_speech_timestamps(
            audio=audio,
            sample_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=config.min_speech_duration_ms,
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms
        )
        
        num_segments = len(speech_timestamps)
        results.append((threshold, num_segments))
        
        print(f"  结果: {num_segments} 个片段")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(f"{'阈值':<10} {'片段数':<10} {'评估'}")
    print("-" * 80)
    
    for threshold, num_segments in results:
        if num_segments == 1:
            status = "❌ 失效"
        elif num_segments < 5:
            status = "⚠️  偏少"
        else:
            status = "✅ 正常"
        
        print(f"{threshold:<10.2f} {num_segments:<10} {status}")
    
    # 推荐阈值
    print("\n【推荐阈值】")
    valid_results = [(t, n) for t, n in results if n >= 5]
    if valid_results:
        # 选择片段数适中的阈值
        recommended = min(valid_results, key=lambda x: abs(x[1] - 15))
        print(f"  推荐使用阈值: {recommended[0]} (片段数: {recommended[1]})")
    else:
        print(f"  ⚠️  所有阈值都未产生足够的片段")
        print(f"  建议检查音频质量或调整其他 VAD 参数")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VAD 隔离测试")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--threshold", type=float, help="VAD 阈值（可选）")
    parser.add_argument("--multi", action="store_true", help="测试多个阈值")
    
    args = parser.parse_args()
    
    try:
        if args.multi:
            test_multiple_thresholds(args.audio_path)
        else:
            test_vad_isolation(args.audio_path, args.threshold)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

