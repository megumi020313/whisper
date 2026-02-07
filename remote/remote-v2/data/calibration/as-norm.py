import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

project_root = "/home/swufe/Project/zhoulonghao/remote/remote-v2"
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入你已实现的类和模型
from backend.models.speaker.eres2netv2 import ERes2NetV2
from backend.core.vector_storage import VectorStorage # 假设你的 AS-Norm 逻辑在这

# ================= 配置区域 =================
TRIALS_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/trials_list.txt"
TARGET_WAV_DIR = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/targets"
MODEL_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/models/eres2netv2/model.ckpt"
# ===========================================

def run_final_calibration():
    # 1. 初始化模型和带有 AS-Norm 的存储模块
    print("⏳ 初始化模型与 AS-Norm 引擎...")
    model = ERes2NetV2(MODEL_PATH, device="cuda:0")
    # VectorStorage 实例化时会自动加载你的 data/vector_db/cohort.npy
    storage = VectorStorage() 

    target_scores = []
    imposter_scores = []

    # 2. 读取 Trials
    with open(TRIALS_PATH, 'r') as f:
        lines = f.readlines()

    print(f"🔎 开始跑分测试，共 {len(lines)} 组比对...")
    
    # 缓存 embedding 避免重复计算同一文件的特征
    embedding_cache = {}

    def get_emb(filename):
        if filename not in embedding_cache:
            path = os.path.join(TARGET_WAV_DIR, filename)
            audio, _ = librosa.load(path, sr=16000)
            emb = model.extract_embedding(audio)
            if hasattr(emb, 'detach'): emb = emb.detach().cpu().numpy()
            embedding_cache[filename] = emb.flatten()
        return embedding_cache[filename]

    for line in tqdm(lines):
        label, file_a, file_b = line.strip().split()
        
        try:
            emb_a = get_emb(file_a)
            emb_b = get_emb(file_b)
            
            # 使用你已实现的 AS-Norm 计算分数的函数
            # 注意：请确保 compute_as_norm_score 接收的是 numpy 数组
            score = storage.compute_as_norm_score(emb_a, emb_b)
            
            if label == '1':
                target_scores.append(score)
            else:
                imposter_scores.append(score)
        except Exception as e:
            print(f"Error processing {file_a}-{file_b}: {e}")

    # 3. 统计学标定
    target_scores = np.array(target_scores)
    imposter_scores = np.array(imposter_scores)

    # 计算不同 FAR 下的阈值
    # FAR 1% (比较宽松)
    t_far_1 = np.percentile(imposter_scores, 99)
    # FAR 0.1% (推荐，工业级标准)
    t_far_01 = np.percentile(imposter_scores, 99.9)

    # 计算对应的 FRR
    frr_01 = np.mean(target_scores < t_far_01) * 100

    print("\n" + "="*40)
    print(f"📈 最终科学标定报告：")
    print(f"【保守型】FAR 1.0%  -> 建议阈值: {t_far_1:.4f}")
    print(f"【标准型】FAR 0.1%  -> 建议阈值: {t_far_01:.4f}")
    print(f"在此阈值({t_far_01:.2f})下，你的 FRR 为: {frr_01:.2f}%")
    print("="*40)

    # 4. 绘图展示
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文展示
    plt.figure(figsize=(12, 6))
    sns.kdeplot(target_scores, fill=True, color="green", label="正样本 (Target)", bw_adjust=0.5)
    sns.kdeplot(imposter_scores, fill=True, color="red", label="负样本 (Imposter)", bw_adjust=0.5)
    plt.axvline(t_far_01, color='blue', linestyle='--', label=f'建议阈值 (FAR 0.1%)')
    
    plt.title("声纹识别 AS-Norm 分数分布图")
    plt.xlabel("AS-Norm 归一化得分 (Z-Score)")
    plt.ylabel("分布密度")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("calibration_result.png")
    print("🎨 分布图已保存至: calibration_result.png")
    plt.show()

if __name__ == "__main__":
    run_final_calibration()