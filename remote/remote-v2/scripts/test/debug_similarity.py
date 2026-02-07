"""相似度计算调试脚本

【功能说明】
调试同一用户不同音频样本的相似度计算，验证：
- 模型是否正常工作
- 原始余弦相似度是否合理
- 音频质量（RMS能量、时长）
- 声纹向量提取是否正确

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_similarity.py

【前置条件】
- 存在注册用户数据（data/calibration/targets/）
- 每个用户至少有2个音频样本

【预期输出】
- 测试音频文件路径
- 音频RMS能量和时长
- 声纹向量形状
- 原始余弦相似度
- 诊断结论：
  - 相似度>0.25：模型正常，问题在AS-Norm
  - 相似度<0.25：模型异常，问题在模型架构或权重
"""
import torch
import soundfile as sf
import numpy as np
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.speaker.eres2netv2 import ERes2NetV2

def debug():
    print("🔍 Starting Debug...")
    
    # 1. 加载模型
    model = ERes2NetV2("models/eres2netv2/model.ckpt", device="cuda:0")
    
    # 2. 找两个同一个人(zlh)的文件
    target_dir = "data/calibration/targets"
    # 自动找第一个用户目录
    users = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    if not users:
        print("❌ No users found in data/calibration/targets")
        return
    
    user = users[0]
    wavs = [os.path.join(target_dir, user, f) for f in os.listdir(os.path.join(target_dir, user)) if f.endswith('.wav')]
    
    if len(wavs) < 2:
        print("❌ Need at least 2 wav files to compare.")
        return
        
    file_a = wavs[0]
    file_b = wavs[1]
    
    print(f"📂 File A: {file_a}")
    print(f"📂 File B: {file_b}")
    
    # 3. 听感检查 (打印时长和能量)
    data_a, sr = sf.read(file_a)
    data_b, sr = sf.read(file_b)
    rms_a = np.sqrt(np.mean(data_a**2))
    rms_b = np.sqrt(np.mean(data_b**2))
    
    print(f"🔊 RMS A: {rms_a:.4f} (Duration: {len(data_a)/sr:.2f}s)")
    print(f"🔊 RMS B: {rms_b:.4f} (Duration: {len(data_b)/sr:.2f}s)")
    
    if rms_a < 0.01 or rms_b < 0.01:
        print("⚠️ WARNING: Audio seems silent! Check your slicing script.")
    
    # 4. 提取特征
    emb_a = model.extract_embedding(data_a)
    emb_b = model.extract_embedding(data_b)
    
    print(f"📐 Embedding Shape: {emb_a.shape}")
    
    # 5. 计算原始余弦相似度
    # 归一化后直接点积
    score = np.dot(emb_a.flatten(), emb_b.flatten())
    
    print(f"\n🎯 Raw Cosine Similarity: {score:.4f}")
    
    if score > 0.25:
        print("✅ Model is working! (Raw score is decent)")
        print("👉 Problem is likely in AS-Norm Cohort data.")
    else:
        print("❌ Model is BROKEN! (Raw score is too low for same person)")
        print("👉 Problem is in Model Architecture or Weights.")

if __name__ == "__main__":
    debug()