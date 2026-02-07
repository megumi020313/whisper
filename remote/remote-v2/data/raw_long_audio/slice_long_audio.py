import os
import glob
import librosa
import soundfile as sf
import numpy as np
import sys

def slice_long_audio():
    # 配置
    INPUT_DIR = "/home/swufe/Project/zhoulonghao/remote/remote-v2/data/raw_long_audio" # 请把你的长音频放在这里
    OUTPUT_BASE = "data/calibration/targets"
    TARGET_SR = 16000
    SLICE_DURATION = 4.0  # 切片时长 (秒)
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"❌ 请将你的长音频(如 zlh.wav) 放入 {INPUT_DIR} 文件夹，然后重新运行此脚本。")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not files:
        print(f"❌ {INPUT_DIR} 文件夹是空的。")
        return

    print(f"🔍 发现 {len(files)} 个长音频，开始切片...")

    for f in files:
        filename = os.path.basename(f)
        user_id = os.path.splitext(filename)[0] # 文件名即用户名
        
        user_out_dir = os.path.join(OUTPUT_BASE, user_id)
        os.makedirs(user_out_dir, exist_ok=True)
        
        try:
            print(f"Processing: {user_id} ...")
            y, sr = librosa.load(f, sr=TARGET_SR, mono=True)
            
            # 去除静音 (VAD)
            intervals = librosa.effects.split(y, top_db=25)
            y_speech = np.concatenate([y[start:end] for start, end in intervals])
            
            # 切片
            samples_per_slice = int(SLICE_DURATION * TARGET_SR)
            total_samples = len(y_speech)
            
            count = 0
            for start in range(0, total_samples, samples_per_slice):
                end = start + samples_per_slice
                if end > total_samples: break
                
                chunk = y_speech[start:end]
                
                # 保存
                out_name = f"{user_id}_{count:03d}.wav"
                sf.write(os.path.join(user_out_dir, out_name), chunk, TARGET_SR)
                count += 1
            
            print(f"✅ {user_id}: 生成了 {count} 个切片")
            
        except Exception as e:
            print(f"❌ 处理 {f} 失败: {e}")

if __name__ == "__main__":
    slice_long_audio()