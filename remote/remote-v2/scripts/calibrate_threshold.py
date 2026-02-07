"""阈值校准工具

【功能说明】
校准声纹识别系统的AS-Norm阈值，通过统计分析确定最佳阈值：
- 计算Target用户的Z-score分布（真阳性）
- 计算Imposter用户的Z-score分布（真阴性）
- 统计FAR（误识率）和FRR（拒识率）
- 推荐最佳high_threshold和low_threshold

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/calibrate_threshold.py

【前置条件】
- Cohort矩阵已构建（data/vector_db/cohort.npy）
- Target用户音频已准备（data/calibration/targets/）
- Imposter用户音频已准备（data/calibration/imposters/）

【输出结果】
- Target Z-score统计（均值、标准差、分位数）
- Imposter Z-score统计
- 不同阈值下的FAR和FRR
- 推荐阈值配置

【建议】
- Target样本数量：每人3-5个音频
- Imposter样本数量：50个以上
- 平衡FAR和FRR，选择合适的阈值
"""
import numpy as np
import glob
import os
import sys
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.speaker.eres2netv2 import ERes2NetV2
from src.core.vector_storage import VectorStorage

def load_audio(path):
    try:
        data, _ = sf.read(path, dtype='float32')
        if len(data.shape) > 1: data = data.mean(axis=1)
        return data
    except:
        return None

def calibrate():
    print("🚀 Starting Calibration...")
    
    # 1. 初始化
    model = ERes2NetV2("models/eres2netv2/model.ckpt", device="cuda:0")
    storage = VectorStorage() # 自动加载 Cohort
    
    if storage.cohort_matrix is None:
        print("❌ Cohort missing! Run build_cohort.py first.")
        return

    target_base = "data/calibration/targets/"
    imposter_dir = "data/calibration/imposters/"
    
    users = [d for d in os.listdir(target_base) if os.path.isdir(os.path.join(target_base, d))]
    imposter_files = glob.glob(os.path.join(imposter_dir, "*.wav"))
    
    if not users or not imposter_files:
        print("❌ Data missing. Need targets and imposters.")
        return

    # 2. 预加载 Imposters
    print(f"Loading {len(imposter_files)} imposters...")
    imposter_embs = []
    for f in imposter_files:
        audio = load_audio(f)
        if audio is not None and len(audio) > 1600:
            emb = model.extract_embedding(audio)
            imposter_embs.append(emb)

    pos_scores = []
    neg_scores = []

    # 3. 跑分
    for user in users:
        user_dir = os.path.join(target_base, user)
        files = glob.glob(os.path.join(user_dir, "*.wav"))
        if len(files) < 2: continue
        
        user_embs = []
        for f in files:
            audio = load_audio(f)
            if audio is not None and len(audio) > 1600:
                emb = model.extract_embedding(audio)
                user_embs.append(emb)
        
        # 正样本 (Self vs Self)
        for i in range(len(user_embs)):
            for j in range(i+1, len(user_embs)):
                s = storage.compute_as_norm_score(user_embs[i], user_embs[j])
                pos_scores.append(s)
        
        # 负样本 (Self vs Imposter)
        for u_emb in user_embs:
            for i_emb in imposter_embs:
                s = storage.compute_as_norm_score(u_emb, i_emb)
                neg_scores.append(s)

    # 4. 统计
    if not pos_scores:
        print("❌ No scores.")
        return

    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    
    p_5 = np.percentile(pos, 5)
    n_99 = np.percentile(neg, 99)
    
    print(f"\n✅ Positive (Same): Mean={np.mean(pos):.2f}, Worst 5%={p_5:.2f}")
    print(f"❌ Negative (Diff): Mean={np.mean(neg):.2f}, Best 1%={n_99:.2f}")
    
    rec_high = max(4.0, p_5 - 0.5)
    rec_low = max(1.5, n_99 + 0.5)
    
    print(f"\n💡 Recommended:\nhigh_threshold: {rec_high:.2f}\nlow_threshold: {rec_low:.2f}")

if __name__ == "__main__":
    calibrate()