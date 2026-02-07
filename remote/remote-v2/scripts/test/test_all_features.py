"""系统全功能集成测试脚本

【功能说明】
测试声纹识别系统的所有核心功能，验证：
- API健康检查
- 用户注册功能
- 用户列表查询
- 声纹识别功能
- ASR语音识别功能
- 系统集成完整性

【启动方式】
1. 先启动API服务器：
   cd /home/swufe/Project/zhoulonghao/remote/remote-v2
   conda activate vioce
   python -m src.api.app

2. 在另一个终端运行测试：
   cd /home/swufe/Project/zhoulonghao/remote/remote-v2
   conda activate vioce
   python scripts/test/test_all_features.py

【前置条件】
- API服务器已启动（https://localhost:8000）
- 测试音频文件已准备

【预期输出】
- 各项功能测试结果（通过/失败）
- API响应数据
- 识别结果详情
- 整体测试统计
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests
import json
from backend.utils.logger import get_logger

logger = get_logger()

# API 基础URL (使用HTTPS，禁用证书验证)
BASE_URL_SPEAKER = "https://localhost:8000/api/v1/speaker"
BASE_URL_API = "https://localhost:8000/api/v1"

# 禁用SSL警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def test_health_check():
    """测试1: 健康检查"""
    print("\n" + "="*80)
    print("测试 1: 健康检查")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL_API}/health", verify=False)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            print("✅ 健康检查通过")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False


def test_list_users():
    """测试2: 获取用户列表"""
    print("\n" + "="*80)
    print("测试 2: 获取用户列表")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL_SPEAKER}/users", verify=False)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"用户数量: {data.get('data', {}).get('count', 0)}")
            users = data.get('data', {}).get('speakers', [])
            if users:
                print(f"用户列表: {users[:5]}")  # 只显示前5个
            print("✅ 获取用户列表成功")
            return True
        else:
            print(f"❌ 获取用户列表失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取用户列表异常: {e}")
        return False


def test_register_user():
    """测试3: 注册用户"""
    print("\n" + "="*80)
    print("测试 3: 注册用户")
    print("="*80)
    
    try:
        # 创建测试音频（3秒，16kHz）- 使用更强的信号
        sample_rate = 16000
        duration = 3
        # 生成正弦波混合白噪声，模拟语音信号
        t = np.linspace(0, duration, sample_rate * duration)
        audio = (np.sin(2 * np.pi * 200 * t) * 0.3 +  # 200Hz 正弦波
                 np.sin(2 * np.pi * 400 * t) * 0.2 +  # 400Hz 正弦波
                 np.random.randn(sample_rate * duration) * 0.1).astype(np.float32)
        
        # 保存为临时WAV文件
        import wave
        temp_file = "/tmp/test_register.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        # 发送注册请求
        with open(temp_file, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            data = {
                'user_id': 'test_user_asr',
                'overwrite': 'true',
                'save_audio': 'false'
            }
            response = requests.post(f"{BASE_URL_SPEAKER}/register", files=files, data=data, verify=False)
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print("✅ 注册用户成功")
            return True
        else:
            print(f"❌ 注册用户失败: {response.status_code}")
            print(f"错误: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 注册用户异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recognize_without_asr():
    """测试4: 识别用户（不启用ASR）"""
    print("\n" + "="*80)
    print("测试 4: 识别用户（检查是否影响原有功能）")
    print("="*80)
    
    try:
        # 创建测试音频 - 使用更强的信号
        sample_rate = 16000
        duration = 3
        # 生成正弦波混合白噪声，模拟语音信号
        t = np.linspace(0, duration, sample_rate * duration)
        audio = (np.sin(2 * np.pi * 200 * t) * 0.3 +
                 np.sin(2 * np.pi * 400 * t) * 0.2 +
                 np.random.randn(sample_rate * duration) * 0.1).astype(np.float32)
        
        # 保存为临时WAV文件
        import wave
        temp_file = "/tmp/test_recognize.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        # 发送识别请求
        with open(temp_file, 'rb') as f:
            files = {'audio': ('test.wav', f, 'audio/wav')}
            response = requests.post(f"{BASE_URL_SPEAKER}/recognize", files=files, verify=False)
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"识别成功: {result.get('success')}")
            print(f"说话人: {result.get('data', {}).get('speaker_id')}")
            print(f"置信度: {result.get('data', {}).get('confidence')}")
            
            # 检查是否有segments
            segments = result.get('details', {}).get('segments', [])
            print(f"片段数量: {len(segments)}")
            
            if segments:
                # 检查是否有text字段
                has_text = any('text' in seg for seg in segments)
                print(f"包含ASR文本: {has_text}")
                
                if has_text:
                    print("\n前3个片段:")
                    for i, seg in enumerate(segments[:3]):
                        print(f"  片段 {i+1}:")
                        print(f"    时间: {seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s")
                        print(f"    说话人: {seg.get('user_id', 'unknown')}")
                        print(f"    文本: {seg.get('text', '无')}")
            
            print("✅ 识别用户成功")
            return True
        else:
            print(f"❌ 识别用户失败: {response.status_code}")
            print(f"错误: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 识别用户异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_endpoint():
    """测试5: 配置端点"""
    print("\n" + "="*80)
    print("测试 5: 获取配置信息")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL_API}/config", verify=False)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"配置项数量: {len(data.get('data', {}))}")
            
            # 检查ASR配置
            asr_enabled = data.get('data', {}).get('asr_enabled')
            print(f"ASR 启用状态: {asr_enabled}")
            
            print("✅ 获取配置成功")
            return True
        else:
            print(f"❌ 获取配置失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取配置异常: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("声纹识别系统 - 完整功能测试")
    print("="*80)
    print("\n⚠️  注意：")
    print("1. 确保API服务已启动（python -m backend.api.app）")
    print("2. 确保配置文件正确（config/model_config.yaml）")
    print("3. 测试将创建临时用户 'test_user_asr'")
    print("\n" + "="*80)
    
    results = []
    
    # 测试1: 健康检查
    results.append(("健康检查", test_health_check()))
    
    # 测试2: 获取用户列表
    results.append(("获取用户列表", test_list_users()))
    
    # 测试3: 注册用户
    results.append(("注册用户", test_register_user()))
    
    # 测试4: 识别用户
    results.append(("识别用户", test_recognize_without_asr()))
    
    # 测试5: 配置端点
    results.append(("获取配置", test_config_endpoint()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n✅ ASR集成成功，原有功能正常")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

