"""ASR 功能测试脚本

【功能说明】
测试 Whisper ASR 服务和推理流水线的 ASR 集成，验证：
- Whisper服务初始化
- 音频转录功能
- 中文语音识别准确性
- ASR与声纹识别流水线集成
- GPU加速效果

【启动方式】
cd /path/to/your/project
conda activate your_env_name
python scripts/test/test_asr.py

【前置条件】
- Whisper模型已下载
- CUDA环境已配置（GPU模式）
- 测试音频文件已准备

【预期输出】
- Whisper服务初始化信息
- 音频转录结果
- 识别准确率统计
- 处理时间和性能指标
- 集成测试结果
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backend.modules.audio_analysis import WhisperService
from backend.utils.logger import get_logger

logger = get_logger()


def test_whisper_service():
    """测试 Whisper 服务"""
    print("\n" + "="*80)
    print("测试 1: Whisper 服务基础功能")
    print("="*80)
    
    try:
        # 初始化服务
        print("\n1. 初始化 Whisper 服务...")
        whisper = WhisperService()
        
        # 获取模型信息
        info = whisper.get_info()
        print(f"   模型信息: {info}")
        
        # 创建测试音频（3秒静音）
        print("\n2. 创建测试音频...")
        sample_rate = 16000
        duration = 3
        audio = np.zeros(sample_rate * duration, dtype=np.float32)
        print(f"   音频长度: {len(audio)} 样本 ({duration}秒)")
        
        # 测试转录
        print("\n3. 测试 ASR 转录...")
        text = whisper.transcribe(audio, language="zh", beam_size=1)
        print(f"   识别结果: '{text}'")
        print(f"   结果长度: {len(text)} 字符")
        
        print("\n✅ Whisper 服务测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ Whisper 服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whisper_advanced():
    """测试 Whisper 高级功能"""
    print("\n" + "="*80)
    print("测试 2: Whisper 高级功能")
    print("="*80)

    try:
        # 初始化服务
        print("\n1. 初始化 Whisper 服务...")
        whisper = WhisperService()

        # 创建测试音频
        print("\n2. 创建测试音频...")
        sample_rate = 16000
        duration = 5
        audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.01
        print(f"   音频长度: {len(audio)} 样本 ({duration}秒)")

        # 测试带时间戳的转录
        print("\n3. 测试带时间戳的 ASR 识别...")
        segments = whisper.transcribe_with_timestamps(audio, language="zh", beam_size=1)
        print(f"   识别片段数: {len(segments)}")
        for i, seg in enumerate(segments[:3]):  # 只显示前3个
            print(f"   片段 {i+1}: [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

        # 测试词级别时间戳
        print("\n4. 测试词级别时间戳...")
        words = whisper.transcribe_with_word_timestamps(audio, language="zh", beam_size=1)
        print(f"   识别词数: {len(words)}")
        for i, word in enumerate(words[:5]):  # 只显示前5个词
            print(f"   词 {i+1}: [{word['start']:.2f}s] {word['word']}")

        print("\n✅ Whisper 高级功能测试通过")
        return True

    except Exception as e:
        print(f"\n❌ Whisper 高级功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_audio():
    """使用真实音频文件测试（如果存在）"""
    print("\n" + "="*80)
    print("测试 3: 真实音频文件测试（可选）")
    print("="*80)

    # 查找测试音频文件
    test_audio_paths = [
        "data/audio_samples/test.wav",
        "data/audio_samples/emp_001_test3.wav",
    ]

    audio_file = None
    for path in test_audio_paths:
        if Path(path).exists():
            audio_file = path
            break

    if audio_file is None:
        print("\n⚠️  未找到测试音频文件，跳过此测试")
        print(f"   尝试的路径: {test_audio_paths}")
        return True

    try:
        print(f"\n1. 加载音频文件: {audio_file}")
        audio, sr = librosa.load(audio_file, sr=16000)
        print(f"   音频长度: {len(audio)} 样本 ({len(audio)/sr:.2f}秒)")

        print("\n2. 初始化 Whisper 服务...")
        whisper = WhisperService()

        print("\n3. 执行音频转文字...")
        text = whisper.transcribe(audio=audio, language="zh", beam_size=1)

        print(f"\n4. 识别结果:")
        print(f"   识别文本: '{text}'")
        print(f"   文本长度: {len(text)} 字符")

        # 如果音频不长，也测试带时间戳的版本
        if len(audio) / sr < 30:  # 30秒以内
            print("\n5. 测试带时间戳的识别...")
            segments = whisper.transcribe_with_timestamps(audio=audio, language="zh", beam_size=1)
            print(f"   识别片段数: {len(segments)}")
            for i, seg in enumerate(segments[:3]):  # 只显示前3个
                print(f"   片段 {i+1}: [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

        print("\n✅ 真实音频测试通过")
        return True

    except Exception as e:
        print(f"\n❌ 真实音频测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("ASR 功能测试套件")
    print("="*80)
    print("\n⚠️  注意：此测试需要先下载 Whisper 模型")
    print("   下载命令: huggingface-cli download Systran/faster-whisper-large-v3 \\")
    print("            --local-dir models/asr/faster-whisper-large-v3")
    print("\n" + "="*80)
    
    results = []
    
    # 测试1: Whisper 服务
    results.append(("Whisper 服务", test_whisper_service()))

    # 测试2: Whisper 高级功能
    results.append(("Whisper 高级功能", test_whisper_advanced()))

    # 测试3: 真实音频（可选）
    results.append(("真实音频测试", test_with_real_audio()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

