"""定向麦克风校准数据生成工具

【功能说明】
为定向麦克风场景生成加噪增强的校准数据：
- 加载背景噪音文件
- 对Target用户音频进行噪音增强
- 生成多个噪音比例的增强样本
- 用于提升定向麦克风场景的识别鲁棒性

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/gen_directional_calibration.py

【前置条件】
- 背景噪音文件已准备（data/noise/restaurant_bg.wav.wav）
- Target用户音频已准备（data/calibration/targets/）

【输出文件】
在每个用户目录下生成增强音频：
- {原文件名}_aug_{factor}.wav

【增强策略】
- 噪音比例：0.1, 0.2, 0.3
- 能量匹配：噪音能量与原音频能量匹配
- 随机采样：从背景噪音中随机选取片段

【应用场景】
- 定向麦克风环境
- 嘈杂环境识别
- 鲁棒性测试
"""
import os
import sys
import glob
import soundfile as sf
import numpy as np
import random

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置路径
NOISE_FILE_PATH = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/noise/restaurant_bg.wav.wav"
TARGET_ROOT = "data/calibration/targets"

BG_NOISE_BUFFER = None

def _load_noise():
    global BG_NOISE_BUFFER
    if os.path.exists(NOISE_FILE_PATH):
        data, sr = sf.read(NOISE_FILE_PATH, dtype='float32')
        if len(data.shape) > 1: data = np.mean(data, axis=1)
        BG_NOISE_BUFFER = data
        print(f"✅ 背景噪音已加载")
    else:
        print(f"❌ 错误: 找不到 {NOISE_FILE_PATH}")
        sys.exit(1)

def augment(clean_audio, factor):
    noise_len = len(BG_NOISE_BUFFER)
    audio_len = len(clean_audio)
    if noise_len == 0: return clean_audio

    if noise_len < audio_len:
        tile = int(np.ceil(audio_len / noise_len))
        noise_segment = np.tile(BG_NOISE_BUFFER, tile)[:audio_len]
    else:
        start = random.randint(0, noise_len - audio_len)
        noise_segment = BG_NOISE_BUFFER[start : start + audio_len]
        
    # 能量匹配
    rms_clean = np.sqrt(np.mean(clean_audio**2))
    rms_noise = np.sqrt(np.mean(noise_segment**2))
    if rms_noise > 0:
        noise_segment = noise_segment * (rms_clean / rms_noise)
        
    # 混合
    return (1.0 - factor) * clean_audio + factor * noise_segment

def main():
    _load_noise()
    
    users = [d for d in os.listdir(TARGET_ROOT) if os.path.isdir(os.path.join(TARGET_ROOT, d))]
    print(f"🔍 为 {len(users)} 个用户生成合成数据...")
    
    for user in users:
        user_dir = os.path.join(TARGET_ROOT, user)
        # 只处理原始切片 (不包含 synth_ 的)
        wavs = [f for f in glob.glob(os.path.join(user_dir, "*.wav")) if "synth_" not in os.path.basename(f)]
        
        for wav_path in wavs:
            try:
                clean_audio, sr = sf.read(wav_path, dtype='float32')
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                
                # 1. 低噪版 (模拟定向麦克风正常工作, 10% 噪音)
                noisy_low = augment(clean_audio, factor=0.1)
                sf.write(os.path.join(user_dir, f"synth_low_{base_name}.wav"), noisy_low, sr)
                
                # 2. 高噪版 (模拟极度嘈杂, 25% 噪音)
                noisy_high = augment(clean_audio, factor=0.25)
                sf.write(os.path.join(user_dir, f"synth_high_{base_name}.wav"), noisy_high, sr)
                
            except Exception as e:
                print(f"Error {wav_path}: {e}")

    print("✅ 合成完成。")

if __name__ == "__main__":
    main()