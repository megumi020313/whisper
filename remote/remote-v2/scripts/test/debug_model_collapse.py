"""模型坍塌调试脚本

【功能说明】
检测ERes2NetV2模型是否存在"模型坍塌"问题，验证：
- 模型是否对不同输入产生不同输出
- 声纹向量的有效性
- BN层统计量是否正确
- 激活函数是否失效

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/debug_model_collapse.py

【测试方法】
生成两个完全不同的音频（随机噪音、不同频率纯音）
提取声纹向量并计算相似度
- 如果相似度>0.9：模型坍塌（输出相同向量）
- 如果相似度<0.1：模型正常

【预期输出】
- 向量范数
- 向量前10位数值
- 相似度分数
- 诊断结论（模型正常/坍塌）
"""
import torch
import numpy as np
import soundfile as sf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.speaker.eres2netv2 import ERes2NetV2

def check_collapse():
    print("🔍 Checking for Model Collapse...")
    model = ERes2NetV2(device="cuda:0")
    # 防止 BN 走训练分支，显式切换到 eval
    if hasattr(model, "model"):
        model.model.eval()
    
    # 1. 生成两个完全不同的随机噪音音频
    # 模拟两个完全不同的人
    audio_a = np.random.uniform(-0.5, 0.5, 16000*3).astype(np.float32) # 3秒白噪 A
    audio_b = np.random.uniform(-0.5, 0.5, 16000*3).astype(np.float32) # 3秒白噪 B

    # 1.1 生成两段频率差异巨大的纯音，避免“随机噪声分布太接近”导致的误判
    t = np.linspace(0, 3, 16000*3, endpoint=False)
    sine_a = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)   # 440 Hz
    sine_b = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)   # 880 Hz
    
    # 2. 提取特征
    emb_a = model.extract_embedding(audio_a).flatten()
    emb_b = model.extract_embedding(audio_b).flatten()

    # 3. 打印向量范数，防止输出全 0 或常数
    print(f"Norm A: {np.linalg.norm(emb_a):.6f}")
    print(f"Norm B: {np.linalg.norm(emb_b):.6f}")
    
    # 3. 打印前 10 位数值
    print(f"\nVector A (First 10): {emb_a[:10]}")
    print(f"Vector B (First 10): {emb_b[:10]}")
    
    # 4. 计算相似度
    score = np.dot(emb_a, emb_b)
    print(f"\n🎯 Similarity between Random Noise A & B: {score:.4f}")
    
    # 5. 判定
    if score > 0.9:
        print("\n❌ 严重故障: 模型坍塌！")
        print("无论输入什么，模型都输出一样的向量。")
        print("原因可能是：权重加载错误、BN层统计量错误、或者激活函数失效。")
    elif score < 0.1:
        print("\n✅ 模型正常: 随机噪音的相似度很低。")
        print("问题可能出在数据预处理环节。")
    else:
        print(f"\n⚠️ 异常: 相似度 {score} 不符合预期 (应接近 0)。")

    # 6. 纯音测试，进一步排除“随机噪声特征过于相似”
    emb_sa = model.extract_embedding(sine_a).flatten()
    emb_sb = model.extract_embedding(sine_b).flatten()
    sine_score = np.dot(emb_sa, emb_sb)
    print(f"\n🎯 Similarity between Sine 440Hz & 880Hz: {sine_score:.4f}")

    # 额外：直接喂随机 FBank 特征，排除前端影响
    with torch.no_grad():
        fb_a = torch.randn(1, 200, 80, device=model.device)
        fb_b = torch.randn(1, 200, 80, device=model.device) * 3.0  # 更大方差
        emb_fa = model.model(fb_a).flatten()
        emb_fb = model.model(fb_b).flatten()
        emb_fa = torch.nn.functional.normalize(emb_fa, p=2, dim=0)
        emb_fb = torch.nn.functional.normalize(emb_fb, p=2, dim=0)
        print(f"\n[FBank Direct] Norm A: {emb_fa.norm().item():.6f}, Norm B: {emb_fb.norm().item():.6f}")
        fb_score = torch.dot(emb_fa, emb_fb).item()
        print(f"🎯 Similarity between random FBanks: {fb_score:.4f}")

if __name__ == "__main__":
    check_collapse()