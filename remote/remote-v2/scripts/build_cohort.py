"""Cohort矩阵构建工具

【功能说明】
构建AS-Norm所需的Cohort矩阵，用于声纹识别的归一化处理：
- 从音频文件提取声纹向量
- L2归一化处理
- 保存为numpy矩阵文件
- 支持.flac和.wav格式
- 过滤过短音频（<1.5秒）

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/build_cohort.py

【前置条件】
- Cohort音频文件已准备（data/calibration/cohort_wavs/）
- ERes2NetV2模型已加载（models/eres2netv2/model.ckpt）
- 音频文件数量建议：500个以上

【输出文件】
data/vector_db/cohort.npy - Cohort矩阵（形状：[N, 192]）

【注意事项】
- 确保Cohort音频不包含注册用户的声音（避免数据泄漏）
- 音频质量要求：清晰、无噪音、时长>1.5秒
- 建议使用多样化的说话人样本
"""
import glob
import numpy as np
import soundfile as sf
import os
import sys
import torch
import librosa  # 建议增加 librosa 处理重采样

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.speaker.eres2netv2 import ERes2NetV2 

def build_cohort():
    print("🚀 Building Cohort Matrix (V19.0)...")
    
    # 使用绝对路径，确保能找到上一步生成的文件
    WAV_DIR = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/cohort_wavs" 
    OUTPUT_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/vector_db/cohort.npy"
    MODEL_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/models/eres2netv2/model.ckpt"
    
    if not os.path.exists(WAV_DIR):
        print(f"❌ Error: {WAV_DIR} does not exist.")
        return
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 加载模型
    model = ERes2NetV2(MODEL_PATH, device="cuda:0")
    
    vectors = []
    # 修改后缀匹配：支持 .flac 和 .wav
    files = glob.glob(os.path.join(WAV_DIR, "*.flac")) + glob.glob(os.path.join(WAV_DIR, "*.wav"))
    # 移除限制，处理所有文件（支持2000个Cohort）
    print(f"Found {len(files)} files. Processing...")

    for f in files:
        try:
            # 使用 librosa 读取并自动重采样到 16000Hz
            audio, sr = librosa.load(f, sr=16000)
            
            # 过滤太短的音频 (1.5秒以下跳过)
            if len(audio) < 16000 * 1.5: 
                print(f"Skipping {f}: too short")
                continue 
            
            # 提取特征
            # 注意：请确保你的 extract_embedding 内部处理了 numpy 到 torch tensor 的转换
            emb = model.extract_embedding(audio)
            
            # 确保 emb 是 1D array (192,)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            emb = emb.flatten()
            
            vectors.append(emb)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not vectors:
        print("❌ No vectors generated. Check if files are empty or path is wrong.")
        return

    # 堆叠
    cohort_matrix = np.array(vectors) # 形状: (N, 192)
    
    # L2 归一化 (AS-Norm 必须基于单位向量进行余弦相似度计算)
    norms = np.linalg.norm(cohort_matrix, axis=1, keepdims=True)
    cohort_matrix = cohort_matrix / (norms + 1e-9)
    
    np.save(OUTPUT_PATH, cohort_matrix)
    print(f"✅ Cohort saved to {OUTPUT_PATH}, shape: {cohort_matrix.shape}")

if __name__ == "__main__":
    build_cohort()