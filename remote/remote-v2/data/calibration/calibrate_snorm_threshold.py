#!/usr/bin/env python3
"""
S-Norm Z-Score阈值校准脚本

功能：
1. 使用S-Norm计算正负样本对的Z-Score分数
2. 基于统计学方法计算high_threshold和low_threshold
3. 生成分布图和详细报告

输出：
- high_threshold: 高置信度阈值（用于success判断）
- low_threshold: 识别阈值（用于unknown拒识）
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Optional

project_root = "/home/swufe/Project/zhoulonghao/remote/remote-v2"
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型和存储模块
from backend.models.speaker.eres2netv2 import ERes2NetV2
from backend.core.vector_storage import VectorStorage

# ================= 配置区域 =================
TRIALS_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/trials_list.txt"
TARGET_WAV_DIR = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/targets"
MODEL_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/models/eres2netv2/model.ckpt"
OUTPUT_PNG = "snorm_calibration_result.png"
# ===========================================

def run_snorm_calibration():
    """运行S-Norm阈值校准"""
    
    # 1. 初始化模型和存储模块
    print("⏳ 初始化模型与S-Norm引擎...")
    model = ERes2NetV2(MODEL_PATH, device="cuda:0")
    
    # 使用绝对路径初始化VectorStorage
    os.chdir(project_root)
    storage = VectorStorage()  # 自动加载 cohort.npy
    
    # 检查Cohort矩阵是否加载
    if storage.cohort_matrix is None:
        print("❌ 错误：Cohort矩阵未加载")
        print(f"   期望路径: {storage.cohort_path}")
        return
    
    print(f"✅ Cohort矩阵形状: {storage.cohort_matrix.shape}")
    print(f"✅ Top-K: {storage.top_k}")
    
    target_scores = []
    imposter_scores = []
    
    # 2. 读取Trials
    with open(TRIALS_PATH, 'r') as f:
        lines = f.readlines()
    
    print(f"🔎 开始S-Norm评分测试，共 {len(lines)} 组比对...")
    
    # 缓存embedding避免重复计算
    embedding_cache = {}
    
    def get_emb(filename):
        """提取并缓存embedding"""
        if filename not in embedding_cache:
            path = os.path.join(TARGET_WAV_DIR, filename)
            audio, _ = librosa.load(path, sr=16000)
            emb = model.extract_embedding(audio)
            if hasattr(emb, 'detach'):
                emb = emb.detach().cpu().numpy()
            embedding_cache[filename] = emb.flatten()
        return embedding_cache[filename]
    
    # 3. 预计算所有音频的T-Norm参数（真正的S-Norm需要这个）
    print("⏳ 预计算T-Norm参数...")
    
    # 收集所有唯一的音频文件
    unique_files = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            unique_files.add(parts[1])
            unique_files.add(parts[2])
    
    # 为每个音频预计算T-Norm参数
    target_metadata_cache = {}
    for filename in tqdm(unique_files, desc="计算T-Norm参数"):
        try:
            emb = get_emb(filename)
            
            # 计算该embedding与cohort的相似度分布
            scores_with_cohort = np.dot(storage.cohort_matrix, emb)
            k = min(storage.top_k, len(scores_with_cohort))
            top_k_idx = np.argpartition(scores_with_cohort, -k)[-k:]
            top_scores = scores_with_cohort[top_k_idx]
            
            # 保存T-Norm参数
            target_metadata_cache[filename] = {
                'tnorm_mu': float(np.mean(top_scores)),
                'tnorm_sigma': float(np.std(top_scores))
            }
        except Exception as e:
            print(f"Error computing T-Norm for {filename}: {e}")
    
    print(f"✅ T-Norm参数计算完成: {len(target_metadata_cache)} 个音频")
    
    # 4. 计算真正的S-Norm分数（Z-Norm + T-Norm）
    for line in tqdm(lines, desc="S-Norm评分"):
        parts = line.strip().split()
        if len(parts) != 3:
            continue
            
        label, file_a, file_b = parts
        
        try:
            emb_a = get_emb(file_a)
            emb_b = get_emb(file_b)
            
            # 使用真正的S-Norm计算（包含T-Norm参数）
            # 注意：这里使用file_a的T-Norm参数（将file_a作为enrollment）
            metadata_a = target_metadata_cache.get(file_a)
            
            if metadata_a:
                score = storage.compute_snorm_score(emb_a, emb_b, target_metadata=metadata_a)
            else:
                # 如果T-Norm参数不存在，降级到Z-Norm
                score = storage.compute_snorm_score(emb_a, emb_b, target_metadata=None)
            
            if label == '1':
                target_scores.append(score)
            else:
                imposter_scores.append(score)
                
        except Exception as e:
            print(f"Error processing {file_a}-{file_b}: {e}")
    
    # 4. 统计分析
    target_scores = np.array(target_scores)
    imposter_scores = np.array(imposter_scores)
    
    print(f"\n✅ 评分完成！")
    print(f"  - 正样本数量: {len(target_scores)}")
    print(f"  - 负样本数量: {len(imposter_scores)}")
    print(f"  - 正样本Z-Score范围: [{target_scores.min():.2f}, {target_scores.max():.2f}]")
    print(f"  - 负样本Z-Score范围: [{imposter_scores.min():.2f}, {imposter_scores.max():.2f}]")
    
    # 5. 计算阈值
    print("\n" + "="*60)
    print("📊 S-Norm Z-Score 阈值校准报告")
    print("="*60)
    
    # 方法1：基于FAR计算low_threshold（识别阈值）
    # FAR 1.0% -> 99%的负样本被正确拒绝
    low_threshold_far1 = np.percentile(imposter_scores, 99.0)
    frr_at_low1 = np.mean(target_scores < low_threshold_far1) * 100
    
    # FAR 0.1% -> 99.9%的负样本被正确拒绝（更严格）
    low_threshold_far01 = np.percentile(imposter_scores, 99.9)
    frr_at_low01 = np.mean(target_scores < low_threshold_far01) * 100
    
    print(f"\n【low_threshold 候选值】（识别阈值，用于unknown拒识）")
    print(f"  方案A (FAR 1.0%):  low_threshold = {low_threshold_far1:.4f}")
    print(f"          → FRR = {frr_at_low1:.2f}%")
    print(f"  方案B (FAR 0.1%): low_threshold = {low_threshold_far01:.4f} ✅ 推荐")
    print(f"          → FRR = {frr_at_low01:.2f}%")
    
    # 方法2：基于正样本分布计算high_threshold（高置信度阈值）
    # 使用正样本的高分位数（例如75%分位数）作为高置信度阈值
    high_threshold_p75 = np.percentile(target_scores, 75)
    high_threshold_p90 = np.percentile(target_scores, 90)
    high_threshold_p95 = np.percentile(target_scores, 95)
    
    # 计算在high_threshold下的FAR
    far_at_high_p75 = np.mean(imposter_scores >= high_threshold_p75) * 100
    far_at_high_p90 = np.mean(imposter_scores >= high_threshold_p90) * 100
    far_at_high_p95 = np.mean(imposter_scores >= high_threshold_p95) * 100
    
    print(f"\n【high_threshold 候选值】（高置信度阈值，用于success判断）")
    print(f"  方案A (P75): high_threshold = {high_threshold_p75:.4f}")
    print(f"          → FAR = {far_at_high_p75:.3f}%")
    print(f"  方案B (P90): high_threshold = {high_threshold_p90:.4f} ✅ 推荐")
    print(f"          → FAR = {far_at_high_p90:.3f}%")
    print(f"  方案C (P95): high_threshold = {high_threshold_p95:.4f}")
    print(f"          → FAR = {far_at_high_p95:.3f}%")
    
    # 6. 计算EER（Equal Error Rate）
    thresholds = np.linspace(imposter_scores.min(), target_scores.max(), 1000)
    far_list = []
    frr_list = []
    
    for t in thresholds:
        far = np.mean(imposter_scores >= t) * 100
        frr = np.mean(target_scores < t) * 100
        far_list.append(far)
        frr_list.append(frr)
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    eer_idx = np.argmin(np.abs(far_array - frr_array))
    eer_threshold = thresholds[eer_idx]
    eer_value = (far_array[eer_idx] + frr_array[eer_idx]) / 2
    
    print(f"\n【EER（等错误率）】")
    print(f"  EER阈值: {eer_threshold:.4f}")
    print(f"  EER值: {eer_value:.2f}%")
    
    # 7. 推荐配置
    print("\n" + "="*60)
    print("🎯 推荐配置（config/model_config.yaml）")
    print("="*60)
    print(f"""
speaker:
  high_threshold: {high_threshold_p90:.4f}  # 高置信度阈值（P90，FAR={far_at_high_p90:.3f}%）
  low_threshold: {low_threshold_far01:.4f}   # 识别阈值（FAR 0.1%，FRR={frr_at_low01:.2f}%）
""")
    
    # 8. 绘图
    print("\n🎨 正在生成分布图...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1：分数分布
    ax1 = axes[0]
    sns.kdeplot(target_scores, fill=True, color="green", label="正样本 (Genuine)", bw_adjust=0.5, ax=ax1)
    sns.kdeplot(imposter_scores, fill=True, color="red", label="负样本 (Imposter)", bw_adjust=0.5, ax=ax1)
    
    # 标记阈值
    ax1.axvline(low_threshold_far01, color='blue', linestyle='--', linewidth=2, 
                label=f'low_threshold={low_threshold_far01:.2f} (FAR 0.1%)')
    ax1.axvline(high_threshold_p90, color='purple', linestyle='--', linewidth=2,
                label=f'high_threshold={high_threshold_p90:.2f} (P90)')
    ax1.axvline(eer_threshold, color='orange', linestyle=':', linewidth=2,
                label=f'EER={eer_threshold:.2f}')
    
    ax1.set_title("S-Norm Z-Score 分数分布", fontsize=14, fontweight='bold')
    ax1.set_xlabel("S-Norm Z-Score", fontsize=12)
    ax1.set_ylabel("密度", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 子图2：FAR/FRR曲线
    ax2 = axes[1]
    ax2.plot(thresholds, far_list, 'r-', label='FAR (False Accept Rate)', linewidth=2)
    ax2.plot(thresholds, frr_list, 'g-', label='FRR (False Reject Rate)', linewidth=2)
    ax2.axvline(eer_threshold, color='orange', linestyle=':', linewidth=2, label=f'EER点 ({eer_value:.2f}%)')
    ax2.axvline(low_threshold_far01, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(high_threshold_p90, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_title("FAR/FRR 曲线", fontsize=14, fontweight='bold')
    ax2.set_xlabel("阈值 (S-Norm Z-Score)", fontsize=12)
    ax2.set_ylabel("错误率 (%)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"✅ 分布图已保存至: {OUTPUT_PNG}")
    
    print("\n" + "="*60)
    print("✅ 校准完成！请根据推荐配置更新 config/model_config.yaml")
    print("="*60)

if __name__ == "__main__":
    run_snorm_calibration()

