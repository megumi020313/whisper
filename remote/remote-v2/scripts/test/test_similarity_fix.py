"""已移除

旧相似度修复测试脚本已移除（识别流程已更新）。
"""

raise RuntimeError("旧相似度修复测试脚本已移除")
        audio_path: 音频文件路径
    """
    print("=" * 80)
    print("测试相似度计算修复")
    print("=" * 80)
    
    # 1. 加载音频
    print(f"\n【步骤1：加载音频】")
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"  ❌ 音频文件不存在: {audio_path}")
        return False
    
    audio, sr = load_audio(audio_path, sample_rate=16000)
    duration = len(audio) / sr
    print(f"  ✓ 音频加载成功")
    print(f"  - 文件: {audio_file.name}")
    print(f"  - 时长: {duration:.2f}s")
    
    # 2. 初始化流水线
    print(f"\n【步骤2：初始化流水线】")
    pipeline = InferencePipeline()
    print(f"  ✓ 流水线初始化成功")
    
    # 3. 执行识别（触发智能融合）
    print(f"\n【步骤3：执行识别】")
    print(f"  正在识别...")
    
    result = pipeline.recognize(
        audio=audio,
        sample_rate=sr,
        audio_filename=audio_file.name
    )
    
    print(f"  ✓ 识别完成")
    print(f"  - 识别成功: {result.get('success', False)}")
    print(f"  - 片段数: {len(result.get('segments', []))}")
    
    # 4. 检查融合日志
    print(f"\n【步骤4：检查融合日志】")
    log_dir = Path("/home/swufe/Project/zhoulonghao/remote/remote-v2/logs/Speaker Recognition")
    
    if not log_dir.exists():
        print(f"  ⚠️  日志目录不存在")
        return False
    
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        print(f"  ⚠️  未找到日志文件")
        return False
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"  ✓ 找到最新日志: {latest_log.name}")
    
    # 5. 分析日志内容
    print(f"\n【步骤5：分析日志】")
    with open(latest_log, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # 检查是否有相似度计算失败
    if "声纹相似度: 计算失败" in log_content:
        print(f"  ❌ 相似度计算仍然失败！")
        print(f"  请检查错误信息：")
        for line in log_content.split('\n'):
            if "计算失败" in line:
                print(f"    {line}")
        return False
    
    # 检查是否有成功的相似度计算
    similarity_lines = [line for line in log_content.split('\n') if "声纹相似度:" in line and "计算失败" not in line]
    
    if not similarity_lines:
        print(f"  ⚠️  未找到相似度计算记录")
        return False
    
    print(f"  ✓ 相似度计算成功！")
    print(f"  - 计算次数: {len(similarity_lines)}")
    
    # 提取相似度值
    similarities = []
    for line in similarity_lines:
        try:
            # 提取相似度值，格式：  声纹相似度: 0.8523 (阈值: 0.8)
            parts = line.split(":")
            if len(parts) >= 2:
                value_part = parts[1].split("(")[0].strip()
                similarity = float(value_part)
                similarities.append(similarity)
        except:
            continue
    
    if similarities:
        print(f"  - 相似度范围: {min(similarities):.4f} - {max(similarities):.4f}")
        print(f"  - 平均相似度: {np.mean(similarities):.4f}")
    
    # 6. 检查融合效果
    print(f"\n【步骤6：检查融合效果】")
    
    # 从日志中提取片段数
    initial_segments = None
    merged_segments = None
    
    for line in log_content.split('\n'):
        if "初始片段数:" in line:
            try:
                initial_segments = int(line.split(":")[-1].strip())
            except:
                pass
        if "融合后片段数:" in line:
            try:
                merged_segments = int(line.split(":")[-1].strip())
            except:
                pass
    
    if initial_segments is not None and merged_segments is not None:
        print(f"  初始片段数: {initial_segments}")
        print(f"  融合后片段数: {merged_segments}")
        
        if merged_segments < initial_segments:
            reduction = initial_segments - merged_segments
            reduction_pct = reduction / initial_segments * 100
            print(f"  ✅ 融合成功！减少了 {reduction} 个片段 ({reduction_pct:.1f}%)")
            return True
        elif merged_segments == initial_segments:
            print(f"  ⚠️  未发生融合（片段数未减少）")
            print(f"  可能原因：")
            print(f"    1. 相似度都低于阈值（需要降低阈值）")
            print(f"    2. 时间间隔都超过阈值（需要增加时间阈值）")
            return False
        else:
            print(f"  ❌ 异常：融合后片段数增加了")
            return False
    else:
        print(f"  ⚠️  无法从日志中提取片段数信息")
        return False
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试相似度计算修复")
    parser.add_argument("audio_path", help="音频文件路径")
    
    args = parser.parse_args()
    
    try:
        success = test_similarity_fix(args.audio_path)
        if success:
            print("\n✅ 测试通过！相似度计算修复成功。")
        else:
            print("\n⚠️  测试未完全通过，请查看上述信息。")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

