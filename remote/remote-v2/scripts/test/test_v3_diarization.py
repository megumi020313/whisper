"""V3.0 对话智能引擎测试脚本

【功能说明】
测试新的并行感知 + 对话智能引擎流水线，验证：
- 声纹识别流水线
- 对话智能引擎（DiarizationEngine）
- 并行感知算法
- 会话片段合并和平滑

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_v3_diarization.py

【测试音频】
默认使用: data/test/zlh.wav
可通过修改 test_audio_path 变量指定其他音频文件

【预期输出】
- 声纹识别结果（用户ID、置信度、时间戳）
- 对话智能引擎处理结果（会话片段、插话检测）
- 性能统计（处理时间、实时率）
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf
from datetime import datetime, timezone

from backend.pipeline.inference_pipeline import InferencePipeline
from backend.modules.diarization import DiarizationEngine
from backend.utils.logger import get_logger

logger = get_logger()


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"{'─'*80}\n")


def test_v3_pipeline():
    """测试V3.0完整流水线"""
    print_separator("V3.0 对话智能引擎测试")
    
    # 1. 初始化组件
    print("📦 初始化组件...")
    pipeline = InferencePipeline(enable_asr=True)
    diarization_engine = DiarizationEngine(
        time_threshold=1.5,
        similarity_threshold=0.80,
        low_confidence_threshold=2.5,
        max_merge_duration=10.0
    )
    print("✅ 组件初始化完成\n")
    
    # 2. 加载测试音频
    test_audio_path = project_root / "data" / "test" / "zlh.wav"
    
    if not test_audio_path.exists():
        logger.error(f"测试音频不存在: {test_audio_path}")
        return
    
    print(f"📂 加载测试音频: {test_audio_path}")
    audio, sample_rate = sf.read(test_audio_path)
    
    # 转换为单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # 转换为float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    duration = len(audio) / sample_rate
    print(f"✅ 音频加载完成: {duration:.2f}秒, {sample_rate}Hz\n")
    
    # 3. 执行V3.0并行感知
    print_separator("阶段1: 并行感知")
    
    start_time = datetime.now(timezone.utc)
    
    try:
        parallel_result = pipeline.recognize_parallel_v3(
            audio=audio,
            sample_rate=sample_rate,
            start_time=start_time,
            language="zh",
            beam_size=5
        )
    except Exception as e:
        logger.error(f"并行感知失败: {e}", exc_info=True)
        return
    
    sv_results = parallel_result['sv_results']
    asr_results = parallel_result['asr_results']
    
    print(f"✅ 声纹识别: {len(sv_results)} 个片段")
    print(f"✅ ASR识别: {len(asr_results)} 个词\n")
    
    # 显示SV结果
    print("【声纹识别结果】")
    for i, sv in enumerate(sv_results[:10], 1):  # 只显示前10个
        print(
            f"  {i}. [{sv['start']:.2f}s - {sv['end']:.2f}s] "
            f"说话人: {sv['speaker']}, "
            f"Z-score: {sv['z_score']:.2f}"
        )
    if len(sv_results) > 10:
        print(f"  ... 还有 {len(sv_results) - 10} 个片段")
    print()
    
    # 显示ASR结果
    print("【ASR识别结果（前20个词）】")
    for i, word in enumerate(asr_results[:20], 1):
        print(
            f"  {i}. '{word['word']}' "
            f"[{word['start']:.2f}s - {word['end']:.2f}s] "
            f"confidence: {word['confidence']:.2f}"
        )
    if len(asr_results) > 20:
        print(f"  ... 还有 {len(asr_results) - 20} 个词")
    print()
    
    # 4. 执行对话智能处理
    print_separator("阶段1.5: 对话智能引擎")
    
    try:
        final_transcript = diarization_engine.process(
            sv_results=sv_results,
            asr_results=asr_results
        )
    except Exception as e:
        logger.error(f"对话智能处理失败: {e}", exc_info=True)
        return
    
    print(f"✅ 对话智能处理完成: {len(final_transcript)} 个对话片段\n")
    
    # 显示最终对话流
    print("【高质量对话流】")
    for i, segment in enumerate(final_transcript, 1):
        print(
            f"\n{i}. 说话人: {segment['speaker']}"
        )
        print(
            f"   时间: [{segment['start']:.2f}s - {segment['end']:.2f}s] "
            f"时长: {segment['duration']:.2f}s"
        )
        print(
            f"   文本: {segment['transcript']}"
        )
        print(
            f"   Z-score: {segment['avg_z_score']:.2f}, "
            f"词数: {segment['word_count']}"
        )
    print()
    
    # 5. 统计分析
    print_separator("统计分析")
    
    # 统计说话人
    speaker_stats = {}
    for segment in final_transcript:
        speaker = segment['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'count': 0,
                'total_duration': 0.0,
                'total_words': 0,
                'avg_z_score': []
            }
        
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_duration'] += segment['duration']
        speaker_stats[speaker]['total_words'] += segment['word_count']
        speaker_stats[speaker]['avg_z_score'].append(segment['avg_z_score'])
    
    print("【说话人统计】")
    for speaker, stats in speaker_stats.items():
        avg_z = np.mean(stats['avg_z_score']) if stats['avg_z_score'] else 0.0
        print(
            f"  {speaker}: "
            f"{stats['count']} 个片段, "
            f"{stats['total_duration']:.2f}秒, "
            f"{stats['total_words']} 个词, "
            f"平均Z-score: {avg_z:.2f}"
        )
    print()
    
    # 对比分析
    print("【对比分析】")
    print(f"  原始SV片段数: {len(sv_results)}")
    print(f"  原始ASR词数: {len(asr_results)}")
    print(f"  最终对话片段数: {len(final_transcript)}")
    print(f"  压缩率: {len(final_transcript) / len(sv_results) * 100:.1f}%")
    print()
    
    print_separator("测试完成")
    print("✅ V3.0 对话智能引擎测试成功！")
    print()


def test_diarization_engine_only():
    """单独测试对话智能引擎（使用模拟数据）"""
    print_separator("对话智能引擎单元测试")
    
    # 创建模拟数据
    sv_results = [
        {'start': 0.0, 'end': 2.0, 'speaker': 'zlh', 'z_score': 5.2, 'embedding': [0.1] * 192, 'confidence': 0.95},
        {'start': 2.1, 'end': 3.5, 'speaker': 'cc', 'z_score': 4.8, 'embedding': [0.2] * 192, 'confidence': 0.92},
        {'start': 3.6, 'end': 5.0, 'speaker': 'zlh', 'z_score': 5.5, 'embedding': [0.1] * 192, 'confidence': 0.96},
    ]
    
    asr_results = [
        {'word': '现在', 'start': 0.0, 'end': 0.5, 'confidence': 0.98},
        {'word': '在', 'start': 0.5, 'end': 0.8, 'confidence': 0.95},
        {'word': '测试', 'start': 0.8, 'end': 1.2, 'confidence': 0.97},
        {'word': '一个', 'start': 1.2, 'end': 1.5, 'confidence': 0.96},
        {'word': '项目', 'start': 1.5, 'end': 2.0, 'confidence': 0.98},
        {'word': '就是', 'start': 2.1, 'end': 2.5, 'confidence': 0.94},
        {'word': '扮演', 'start': 2.5, 'end': 2.9, 'confidence': 0.96},
        {'word': '一个', 'start': 2.9, 'end': 3.2, 'confidence': 0.95},
        {'word': '客人', 'start': 3.2, 'end': 3.5, 'confidence': 0.97},
        {'word': '然后', 'start': 3.6, 'end': 4.0, 'confidence': 0.96},
        {'word': '我是', 'start': 4.0, 'end': 4.3, 'confidence': 0.95},
        {'word': '本人', 'start': 4.3, 'end': 4.7, 'confidence': 0.98},
    ]
    
    print("📦 模拟数据:")
    print(f"  SV片段: {len(sv_results)} 个")
    print(f"  ASR词: {len(asr_results)} 个\n")
    
    # 初始化引擎
    engine = DiarizationEngine(
        time_threshold=1.5,
        similarity_threshold=0.80,
        low_confidence_threshold=2.5
    )
    
    # 处理
    result = engine.process(sv_results, asr_results)
    
    print(f"✅ 处理完成: {len(result)} 个对话片段\n")
    
    # 显示结果
    print("【对话流结果】")
    for i, segment in enumerate(result, 1):
        print(
            f"{i}. [{segment['start']:.1f}s - {segment['end']:.1f}s] "
            f"{segment['speaker']}: {segment['transcript']} "
            f"(Z={segment['avg_z_score']:.2f}, {segment['word_count']}词)"
        )
    print()
    
    print_separator("单元测试完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V3.0 对话智能引擎测试")
    parser.add_argument(
        '--mode',
        choices=['full', 'unit'],
        default='full',
        help='测试模式: full=完整流水线, unit=单元测试'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        test_v3_pipeline()
    else:
        test_diarization_engine_only()

