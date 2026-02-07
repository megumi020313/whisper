"""Cohort泄漏调试脚本

【功能说明】
检测AS-Norm的Cohort矩阵是否存在"数据泄漏"问题，验证：
- Cohort矩阵是否包含注册用户的声纹
- Target向量与Cohort向量的相似度分布
- 高相似度嫌疑向量的来源

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_cohort_leak.py

【前置条件】
- Cohort矩阵已构建（data/vector_db/cohort.npy）
- 保留了Cohort源文件（data/cohort_wavs/*.wav）
- 存在注册用户数据（data/calibration/targets/）

【预期输出】
- Cohort大小和Target信息
- 高相似度（>0.8）嫌疑向量列表
- 嫌疑向量对应的源文件
- 数据泄漏诊断结论
"""
import numpy as np
import os
import sys
import soundfile as sf
import glob

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.speaker.eres2netv2 import ERes2NetV2
from src.core.vector_storage import VectorStorage

def find_leak():
    print("🕵️‍♂️ Starting Cohort Leak Investigation...")
    
    # 1. 加载模型和 Cohort
    model = ERes2NetV2("models/eres2netv2/pretrained_eres2netv2.ckpt", device="cuda:0")
    storage = VectorStorage()
    
    if storage.cohort_matrix is None:
        print("❌ Cohort not loaded.")
        return

    # 2. 加载一个 Target (你自己)
    target_dir = "data/calibration/targets"
    users = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    if not users:
        print("❌ No targets found.")
        return
    
    # 随便找一个 wav
    target_wav = glob.glob(os.path.join(target_dir, users[0], "*.wav"))[0]
    print(f"👤 Target: {target_wav}")
    
    audio, _ = sf.read(target_wav, dtype='float32')
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    target_emb = model.extract_embedding(audio).flatten()
    
    # 3. 遍历 Cohort 源文件 (如果有的话)
    # 如果你保留了 data/cohort_wavs，我们可以一一对应
    cohort_dir = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/cohort_wavs"
    cohort_files = sorted(glob.glob(os.path.join(cohort_dir, "*.wav")))
    
    print(f"📚 Cohort Size: {len(storage.cohort_matrix)}")
    
    # 计算 Target 和 Cohort 所有向量的相似度
    scores = np.dot(storage.cohort_matrix, target_emb)
    
    # 4. 找出高分嫌疑人
    suspicious_indices = np.where(scores > 0.8)[0]
    
    if len(suspicious_indices) == 0:
        print("\n✅ No leaks found. Max similarity is:", np.max(scores))
        print("👉 如果 Max 依然很低 (<0.3)，但 Z-Score 也很低，说明 Std (标准差) 太大了。")
        print(f"   Cohort Mean: {np.mean(scores):.4f}")
        print(f"   Cohort Std:  {np.std(scores):.4f}")
    else:
        print(f"\n⚠️ FOUND {len(suspicious_indices)} SUSPICIOUS FILES (Score > 0.8)!")
        print("这些文件和你的声音太像了，导致 AS-Norm 失效：")
        
        # 尝试打印文件名
        if len(cohort_files) == len(storage.cohort_matrix):
            for idx in suspicious_indices:
                print(f"   [{idx}] Score={scores[idx]:.4f} -> {os.path.basename(cohort_files[idx])}")
        else:
            print("   (无法对应文件名，因为 cohort.npy 和 cohort_wavs 数量不一致)")
            print(f"   Indices: {suspicious_indices}")

if __name__ == "__main__":
    find_leak()