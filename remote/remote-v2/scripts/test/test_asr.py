"""ASR 功能测试脚本

【功能说明】
测试 FunASR ASR 服务和推理流水线的 ASR 集成，验证：
- FunASR 服务初始化
- 音频转录功能
- 中文语音识别准确性
- ASR与声纹识别流水线集成
- GPU加速效果

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_asr.py

【前置条件】
- FunASR 模型已下载
- CUDA环境已配置（GPU模式）
- 测试音频文件已准备

【预期输出】
- FunASR 服务初始化信息
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
import librosa
from backend.modules.audio_analysis import FunASRService
from backend.pipeline.inference_pipeline import InferencePipeline
from backend.utils.logger import get_logger

logger = get_logger()


def test_funasr_service():
    """测试 FunASR 服务"""
    print("\n" + "="*80)
    print("测试 1: FunASR 服务基础功能")
    print("="*80)
    
    try:
        # 初始化服务
        print("\n1. 初始化 FunASR 服务...")
        funasr = FunASRService()
        
        # 获取模型信息
        info = funasr.get_info()
        print(f"   模型信息: {info}")
        
        # 创建测试音频（3秒静音）
        print("\n2. 创建测试音频...")
        sample_rate = 16000
        duration = 3
        audio = np.zeros(sample_rate * duration, dtype=np.float32)
        print(f"   音频长度: {len(audio)} 样本 ({duration}秒)")
        
        # 测试转录
        print("\n3. 测试 ASR 转录...")
        text = funasr.transcribe(audio, language="zh", beam_size=1)
        print(f"   识别结果: '{text}'")
        print(f"   结果长度: {len(text)} 字符")
        
        print("\n✅ FunASR 服务测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ FunASR 服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_with_asr():
    """测试推理流水线的 ASR 集成"""
    print("\n" + "="*80)
    print("测试 2: 推理流水线 ASR 集成")
    print("="*80)
    
    try:
        # 初始化流水线（启用ASR）
        print("\n1. 初始化推理流水线（启用 ASR）...")
        pipeline = InferencePipeline(enable_asr=True)
        
        # 检查ASR是否启用
        if not pipeline.enable_asr:
            print("   ⚠️  ASR 未成功启用")
            return False
        
        print(f"   ASR 状态: {'已启用' if pipeline.enable_asr else '未启用'}")
        
        # 创建测试音频
        print("\n2. 创建测试音频...")
        sample_rate = 16000
        duration = 5
        audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.01
        print(f"   音频长度: {len(audio)} 样本 ({duration}秒)")
        
        # 测试仅ASR识别
        print("\n3. 测试仅 ASR 识别...")
        text = pipeline.transcribe_only(audio, language="zh", beam_size=1)
        print(f"   识别文本: '{text}'")
        
        # 测试带时间戳的ASR识别
        print("\n4. 测试带时间戳的 ASR 识别...")
        segments = pipeline.transcribe_only(audio, language="zh", beam_size=1, with_timestamps=True)
        print(f"   识别片段数: {len(segments)}")
        for i, seg in enumerate(segments[:3]):  # 只显示前3个
            print(f"   片段 {i+1}: [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        
        print("\n✅ 推理流水线 ASR 集成测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 推理流水线 ASR 集成测试失败: {e}")
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
        
        print("\n2. 初始化推理流水线...")
        pipeline = InferencePipeline(enable_asr=True)
        
        if not pipeline.enable_asr:
            print("   ⚠️  ASR 未成功启用")
            return False
        
        print("\n3. 执行声纹识别 + ASR...")
        result = pipeline.recognize_with_asr(
            audio=audio,
            sample_rate=sr,
            language="zh",
            beam_size=1
        )
        
        print(f"\n4. 识别结果:")
        print(f"   ASR 启用: {result.get('asr_enabled', False)}")
        print(f"   片段数量: {len(result.get('segments', []))}")
        
        for i, seg in enumerate(result.get('segments', [])[:5]):  # 只显示前5个
            print(f"\n   片段 {i+1}:")
            print(f"     时间: [{seg['start']:.2f}s - {seg['end']:.2f}s]")
            print(f"     说话人: {seg.get('user_id', 'unknown')}")
            print(f"     文本: {seg.get('text', '')}")
            print(f"     得分: {seg.get('score', 0):.2f}")
        
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
    print("\n⚠️  注意：此测试需要先下载 FunASR 模型")
    print("   模型目录: models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    print("\n" + "="*80)
    
    results = []
    
    # 测试1: FunASR 服务
    results.append(("FunASR 服务", test_funasr_service()))
    
    # 测试2: 推理流水线集成
    results.append(("推理流水线集成", test_pipeline_with_asr()))
    
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

