#!/usr/bin/env python3
"""
方案B：从5000个音频中划分
- Cohort: 4000个音频（约650人）
- Targets: 1000个音频（约181人），每人2-6个音频用于测试
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 配置参数
SOURCE_DIR = "data/raw_data_grouped"
OUTPUT_DIR = "data/calibration"
COHORT_COUNT = 4000  # Cohort音频数量
TARGET_SPEAKER_COUNT = 181  # Targets说话人数量（剩余的人）

random.seed(42)  # 固定随机种子，保证可复现

def collect_speakers(source_dir):
    """收集所有说话人及其音频文件"""
    source_path = Path(source_dir)
    speaker_dict = {}
    
    for speaker_dir in source_path.iterdir():
        if speaker_dir.is_dir():
            wav_files = list(speaker_dir.glob("*.wav"))
            if wav_files:
                speaker_dict[speaker_dir.name] = wav_files
    
    return speaker_dict

def main():
    print("="*60)
    print("方案B：4000 Cohort + 1000 Targets")
    print("="*60)
    
    # 收集所有说话人
    print("\n正在扫描说话人...")
    speaker_dict = collect_speakers(SOURCE_DIR)
    all_speakers = list(speaker_dict.keys())
    random.shuffle(all_speakers)
    
    total_files = sum(len(files) for files in speaker_dict.values())
    print(f"✅ 找到 {len(all_speakers)} 个说话人，共 {total_files} 个音频文件")
    
    # 划分说话人：Cohort vs Targets
    cohort_speakers = []
    target_speakers = []
    
    # 先选择Targets说话人（181人，保证每人至少2个音频）
    print(f"\n正在选择 Targets 说话人（目标: {TARGET_SPEAKER_COUNT}人）...")
    for spk in all_speakers:
        if len(target_speakers) >= TARGET_SPEAKER_COUNT:
            break
        if len(speaker_dict[spk]) >= 2:  # 至少2个音频才能做测试
            target_speakers.append(spk)
    
    # 剩余的作为Cohort候选
    cohort_candidates = [spk for spk in all_speakers if spk not in target_speakers]
    
    print(f"✅ Targets 说话人: {len(target_speakers)} 人")
    print(f"✅ Cohort 候选: {len(cohort_candidates)} 人")
    
    # 从Cohort候选中选择音频，凑够4000个
    print(f"\n正在选择 Cohort 音频（目标: {COHORT_COUNT}个）...")
    cohort_files = []
    for spk in cohort_candidates:
        files = speaker_dict[spk]
        cohort_files.extend(files)
        if len(cohort_files) >= COHORT_COUNT:
            break
    
    # 如果超过4000，随机截取
    if len(cohort_files) > COHORT_COUNT:
        cohort_files = random.sample(cohort_files, COHORT_COUNT)
    
    print(f"✅ Cohort 音频: {len(cohort_files)} 个")
    
    # 统计Cohort实际说话人数
    cohort_speaker_ids = set()
    for f in cohort_files:
        # 从文件名提取speaker_id (例如: 14_3466_14_3466_xxx.wav -> 14_3466)
        parts = f.name.split('_')
        if len(parts) >= 2:
            cohort_speaker_ids.add(f"{parts[0]}_{parts[1]}")
    
    print(f"✅ Cohort 实际说话人数: {len(cohort_speaker_ids)} 人")
    
    # 准备Targets音频
    target_files = []
    for spk in target_speakers:
        target_files.extend(speaker_dict[spk])
    
    print(f"✅ Targets 音频: {len(target_files)} 个")
    
    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    cohort_dir = output_path / "cohort_wavs"
    targets_dir = output_path / "targets"
    
    cohort_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制Cohort文件
    print(f"\n正在复制 Cohort 音频...")
    for src_file in tqdm(cohort_files, desc="Cohort"):
        dst_file = cohort_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    
    # 复制Targets文件
    print(f"\n正在复制 Targets 音频...")
    for src_file in tqdm(target_files, desc="Targets"):
        dst_file = targets_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    
    # 生成 trials_list.txt
    print(f"\n正在生成 trials_list.txt...")
    trials_file = output_path / "trials_list.txt"
    
    with open(trials_file, 'w', encoding='utf-8') as f:
        # 为每个Targets说话人生成正负样本对
        for spk in tqdm(target_speakers, desc="生成trials"):
            spk_files = speaker_dict[spk]
            if len(spk_files) < 2:
                continue
            
            # 正样本：同一说话人的不同音频配对
            for i in range(len(spk_files)):
                for j in range(i + 1, len(spk_files)):
                    f.write(f"1 {spk_files[i].name} {spk_files[j].name}\n")
            
            # 负样本：与其他说话人配对（每人随机选5个）
            other_speakers = [s for s in target_speakers if s != spk]
            sampled_others = random.sample(other_speakers, min(5, len(other_speakers)))
            
            for other_spk in sampled_others:
                other_files = speaker_dict[other_spk]
                for spk_file in spk_files[:2]:  # 只用前2个文件
                    for other_file in other_files[:2]:
                        f.write(f"0 {spk_file.name} {other_file.name}\n")
    
    # 统计trials
    with open(trials_file, 'r') as f:
        lines = f.readlines()
        positive_count = sum(1 for line in lines if line.startswith('1'))
        negative_count = sum(1 for line in lines if line.startswith('0'))
    
    # 最终统计
    print("\n" + "="*60)
    print("✅ 数据准备完成！")
    print("="*60)
    print(f"\n【Cohort】")
    print(f"  - 音频数量: {len(cohort_files)}")
    print(f"  - 说话人数: {len(cohort_speaker_ids)}")
    print(f"  - 保存路径: {cohort_dir}")
    
    print(f"\n【Targets】")
    print(f"  - 音频数量: {len(target_files)}")
    print(f"  - 说话人数: {len(target_speakers)}")
    print(f"  - 平均每人: {len(target_files) / len(target_speakers):.1f} 个音频")
    print(f"  - 保存路径: {targets_dir}")
    
    print(f"\n【Trials】")
    print(f"  - 正样本对: {positive_count}")
    print(f"  - 负样本对: {negative_count}")
    print(f"  - 总样本对: {positive_count + negative_count}")
    print(f"  - 保存路径: {trials_file}")
    
    print("\n" + "="*60)
    print("下一步操作：")
    print("1. 运行 build_cohort.py 生成 cohort.npy")
    print("2. 更新 model_config.yaml:")
    print(f"   - cohort_size: {len(cohort_files)}")
    print(f"   - top_k: {int(len(cohort_files) * 0.1)}-{int(len(cohort_files) * 0.15)}")
    print("3. 运行 calibrate_threshold.py 计算阈值")
    print("="*60)

if __name__ == "__main__":
    main()

