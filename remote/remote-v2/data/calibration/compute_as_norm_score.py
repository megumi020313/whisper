import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 强制指向项目根目录
PROJECT_ROOT = "/home/swufe/Project/zhoulonghao/remote/remote-v2"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.speaker.eres2netv2 import ERes2NetV2
from src.core.vector_storage import VectorStorage

# ================= 配置区域 =================
TRIALS_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/trials_list.txt"
TARGET_WAV_DIR = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/calibration/targets"
MODEL_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/models/eres2netv2/model.ckpt"
# ===========================================

def run_score_calibration():
    # 1. 初始化模型与存储库
    print("⏳ 正在加载模型和 AS-Norm Cohort 库...")
    model = ERes2NetV2(MODEL_PATH, device="cuda:0")
    storage = VectorStorage() # 确保其内部已正确 load 了 cohort.npy

    target_scores = []
    imposter_scores = []
    
    # 2. 读取测试清单
    with open(TRIALS_PATH, 'r') as f:
        lines = f.readlines()

    # 缓存 Embedding，避免重复加载同一文件
    cache = {}

    def get_embedding(filename):
        if filename not in cache:
            path = os.path.join(TARGET_WAV_DIR, filename)
            # 16k采样率重采样
            audio, _ = librosa.load(path, sr=16000)
            emb = model.extract_embedding(audio)
            if hasattr(emb, 'detach'): emb = emb.detach().cpu().numpy()
            cache[filename] = emb.flatten()
        return cache[filename]

    print(f"🔎 正在计算 {len(lines)} 组比对分数...")
    for line in tqdm(lines):
        label, f1, f2 = line.strip().split()
        try:
            emb1 = get_embedding(f1)
            emb2 = get_embedding(f2)
            
            # 调用你实现的 AS-Norm 核心算法
            score = storage.compute_as_norm_score(emb1, emb2)
            
            if label == '1':
                target_scores.append(score)
            else:
                imposter_scores.append(score)
        except Exception as e:
            print(f"跳过错误对 {f1}-{f2}: {e}")

    target_scores = np.array(target_scores)
    imposter_scores = np.array(imposter_scores)

    # 3. 核心决策矩阵：寻找最优平衡点
    print("\n" + "="*55)
    print(f"{'阈值 (Threshold)':<15} | {'FAR (误报率)':<15} | {'FRR (拒真率)':<15}")
    print("-" * 55)
    
    # 扫描不同阈值，看性能表现
    best_eer_threshold = 0
    min_diff = 100
    
    for t in np.arange(2.0, 8.5, 0.5):
        far = np.mean(imposter_scores >= t) * 100
        frr = np.mean(target_scores < t) * 100
        print(f"{t:<15.2f} | {far:<15.2f}% | {frr:<15.2f}%")
        
        # 寻找等错误率 (EER) 附近的点
        if abs(far - frr) < min_diff:
            min_diff = abs(far - frr)
            best_eer_threshold = t

    print("-" * 55)
    print(f"💡 建议最优平衡阈值 (EER附近): {best_eer_threshold:.2f}")
    print("="*55)

    # 4. 可视化分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(target_scores, color="green", label="Target (自己人)", kde=True, stat="density")
    sns.histplot(imposter_scores, color="red", label="Imposter (陌生人)", kde=True, stat="density")
    plt.axvline(best_eer_threshold, color='blue', linestyle='--', label=f'EER Threshold {best_eer_threshold}')
    plt.title("AS-Norm Score Distribution (ERes2NetV2)")
    plt.xlabel("Norm Score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("as_norm_distribution.png")
    print("\n📊 分布直方图已保存至项目根目录: as_norm_distribution.png")

if __name__ == "__main__":
    run_score_calibration()