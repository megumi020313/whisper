"""已移除

V2.0 旧流程测试脚本已移除（recognize 已废弃）。
    

raise RuntimeError("V2.0 测试脚本已移除")
    Args:
        audio_path: 音频文件路径
    
    Returns:
        (audio, sample_rate) 元组
    """
    try:
        audio, sr = sf.read(audio_path)
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # 转换为 float32
        audio = audio.astype(np.float32)
        
        return audio, sr
    except Exception as e:
        logger.error(f"加载音频失败: {e}")
        raise


def test_vad_optimization():
    """测试 VAD 优化（min_silence_duration_ms 从 300 提升到 800）"""
    print("\n" + "="*60)
    print("🧪 测试 VAD 优化")
    print("="*60)
    
    try:
        from backend.core.config import get_config
        config = get_config()
        
        print(f"\n📊 VAD 配置:")
        print(f"   - min_silence_duration_ms: {config.min_silence_duration_ms}")
        print(f"   - min_speech_duration_ms: {config.min_speech_duration_ms}")
        print(f"   - threshold: {config.vad_threshold}")
        
        if config.min_silence_duration_ms == 800:
            print("\n✅ VAD 配置已优化（800ms）")
        else:
            print(f"\n⚠️  VAD 配置未优化（当前: {config.min_silence_duration_ms}ms，期望: 800ms）")
        
        return config.min_silence_duration_ms == 800
        
    except Exception as e:
        print(f"\n❌ VAD 配置检查失败: {e}")
        return False


def test_intelligent_merge():
    """测试智能会话融合算法"""
    print("\n" + "="*60)
    print("🧪 测试智能会话融合算法")
    print("="*60)
    
    try:
        # 创建推理流水线
        pipeline = InferencePipeline()
        
        # 检查是否有智能融合方法
        if hasattr(pipeline, '_intelligent_merge_segments'):
            print("\n✅ 智能融合方法已实现")
        else:
            print("\n❌ 智能融合方法未找到")
            return False
        
        # 加载测试音频（使用 zlh.wav，通常包含真实语音）
        test_audio_path = project_root / "data" / "test" / "zlh.wav"
        print(f"\n📊 加载测试音频: {test_audio_path}")
        audio, sr = load_test_audio(test_audio_path)
        print(f"   - 音频时长: {len(audio) / sr:.2f}s")
        print(f"   - 采样率: {sr} Hz")
        
        # 重采样到 16kHz（如果需要）
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            print(f"   - 已重采样到 16000 Hz")
        
        # 执行 VAD 切分
        print("\n📊 执行 VAD 切分...")
        speech_timestamps = pipeline.vad.get_speech_timestamps(
            audio=audio,
            sample_rate=sr,
            threshold=pipeline.config.vad_threshold,
            min_speech_duration_ms=pipeline.config.min_speech_duration_ms,
            min_silence_duration_ms=pipeline.config.min_silence_duration_ms,
            speech_pad_ms=pipeline.config.speech_pad_ms
        )
        
        print(f"   - VAD 检测到 {len(speech_timestamps)} 个片段")
        for i, seg in enumerate(speech_timestamps[:10], 1):  # 只显示前10个
            duration = (seg['end'] - seg['start']) / sr
            print(f"     {i}. [{seg['start']/sr:.2f}s - {seg['end']/sr:.2f}s] 时长: {duration:.2f}s")
        if len(speech_timestamps) > 10:
            print(f"     ... 还有 {len(speech_timestamps) - 10} 个片段")
        
        # 执行智能融合
        print("\n📊 执行智能融合...")
        merged_segments = pipeline._intelligent_merge_segments(
            initial_segments=speech_timestamps,
            audio=audio,
            sample_rate=sr,
            time_threshold=1.5,
            similarity_threshold=0.80,
            max_merge_duration=10.0
        )
        
        print(f"   - 融合后 {len(merged_segments)} 个片段")
        for i, seg in enumerate(merged_segments[:10], 1):  # 只显示前10个
            duration = (seg['end'] - seg['start']) / sr
            merged_count = seg.get('merged_count', 1)
            print(f"     {i}. [{seg['start']/sr:.2f}s - {seg['end']/sr:.2f}s] 时长: {duration:.2f}s (合并了 {merged_count} 个片段)")
        if len(merged_segments) > 10:
            print(f"     ... 还有 {len(merged_segments) - 10} 个片段")
        
        # 验证融合效果
        print("\n📊 验证融合效果:")
        avg_duration_before = np.mean([(s['end'] - s['start']) / sr for s in speech_timestamps])
        avg_duration_after = np.mean([(s['end'] - s['start']) / sr for s in merged_segments])
        
        print(f"   - 融合前平均时长: {avg_duration_before:.2f}s")
        print(f"   - 融合后平均时长: {avg_duration_after:.2f}s")
        print(f"   - 时长提升: {(avg_duration_after / avg_duration_before - 1) * 100:.1f}%")
        
        if avg_duration_after > avg_duration_before:
            print("\n✅ 智能融合有效，片段平均时长提升")
            return True
        else:
            print("\n⚠️  智能融合效果不明显")
            return False
        
    except Exception as e:
        print(f"\n❌ 智能融合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整的识别流水线"""
    print("\n" + "="*60)
    print("🧪 测试完整识别流水线")
    print("="*60)
    
    try:
        # 创建推理流水线
        pipeline = InferencePipeline()
        
        # 加载测试音频（使用 zlh.wav，通常包含真实语音）
        test_audio_path = project_root / "data" / "test" / "zlh.wav"
        print(f"\n📊 加载测试音频: {test_audio_path}")
        audio, sr = load_test_audio(test_audio_path)
        print(f"   - 音频时长: {len(audio) / sr:.2f}s")
        
        # 重采样到 16kHz（如果需要）
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # 执行识别
        print("\n📊 执行声纹识别...")
        result = pipeline.recognize(
            audio=audio,
            sample_rate=sr,
            threshold=None,
            use_vad_batch=True
        )
        
        # 显示结果
        print(f"\n📊 识别结果:")
        print(f"   - 成功: {result.get('success', False)}")
        print(f"   - 模式: {result.get('mode', 'unknown')}")
        print(f"   - 检测到说话人数: {result.get('total_speakers', 0)}")
        print(f"   - VAD 片段数: {result.get('vad_regions', 0)}")
        print(f"   - 融合后片段数: {result.get('merged_regions', 0)}")
        print(f"   - 总窗口数: {result.get('total_windows', 0)}")
        
        # 显示片段信息
        segments = result.get('segments', [])
        if segments:
            print(f"\n📊 识别片段 ({len(segments)} 个):")
            for i, seg in enumerate(segments[:5], 1):  # 只显示前5个
                print(f"     {i}. [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] "
                      f"说话人: {seg.get('user_id', 'unknown')} "
                      f"Z-score: {seg.get('raw_z_score', 0):.2f}")
        
        # 验证是否使用了 V2.0 模式
        if result.get('mode') == 'vad_batch_v20':
            print("\n✅ 使用 V2.0 优化模式")
        else:
            print(f"\n⚠️  未使用 V2.0 模式（当前: {result.get('mode')}）")
        
        # 验证融合效果
        vad_regions = result.get('vad_regions', 0)
        merged_regions = result.get('merged_regions', 0)
        
        if merged_regions > 0 and merged_regions < vad_regions:
            reduction = (1 - merged_regions / vad_regions) * 100
            print(f"✅ 智能融合生效: VAD {vad_regions} → 融合 {merged_regions} (减少 {reduction:.1f}%)")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n❌ 完整流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("🚀 V2.0 声纹识别质量优化测试")
    print("="*60)
    
    results = {}
    
    # 测试1: VAD 优化
    results['vad'] = test_vad_optimization()
    
    # 测试2: 智能融合
    results['merge'] = test_intelligent_merge()
    
    # 测试3: 完整流水线
    results['pipeline'] = test_full_pipeline()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    print(f"\n✅ VAD 优化: {'通过' if results['vad'] else '失败'}")
    print(f"✅ 智能融合: {'通过' if results['merge'] else '失败'}")
    print(f"✅ 完整流水线: {'通过' if results['pipeline'] else '失败'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！V2.0 优化已成功实施")
        print("\n💡 预期效果:")
        print("   - Z-score 提升: 4.0-5.0 → 6.0-9.0")
        print("   - 片段平均时长提升: 1.5-2.5s → 3.0-5.0s")
        print("   - 识别准确率提升: 5-10%")
    else:
        print("\n⚠️  部分测试失败，请检查实施情况")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

