"""CUDA/cuDNN 配置测试脚本

【功能说明】
验证 CUDA/cuDNN 库是否正确配置，检查：
- CUDA可用性和版本
- cuDNN可用性和版本
- GPU设备信息
- FunASR模型GPU运行能力
- 简单推理测试

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_cuda_setup.py

【前置条件】
- NVIDIA驱动已安装
- CUDA Toolkit已安装
- cuDNN库已配置

【预期输出】
- CUDA版本和设备信息
- cuDNN版本和配置状态
- GPU内存信息
- FunASR模型加载和推理测试结果
- 所有测试通过/失败状态
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import numpy as np


def test_cuda_available():
    """测试 CUDA 是否可用"""
    print("\n" + "="*80)
    print("测试 1: CUDA 可用性")
    print("="*80)
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("❌ CUDA 不可用")
        return False


def test_cudnn_available():
    """测试 cuDNN 是否可用"""
    print("\n" + "="*80)
    print("测试 2: cuDNN 可用性")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过 cuDNN 测试")
        return False
    
    try:
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"cuDNN 可用: {cudnn_available}")
        
        if cudnn_available:
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN 版本: {cudnn_version}")
            print(f"cuDNN 启用: {torch.backends.cudnn.enabled}")
            return True
        else:
            print("❌ cuDNN 不可用")
            return False
    except Exception as e:
        print(f"❌ cuDNN 测试失败: {e}")
        return False


def test_library_paths():
    """测试库路径配置"""
    print("\n" + "="*80)
    print("测试 3: 动态库路径")
    print("="*80)
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"LD_LIBRARY_PATH:")
    if ld_library_path:
        for path in ld_library_path.split(':'):
            if path:
                print(f"  - {path}")
    else:
        print("  (未设置)")
    
    # 检查 nvidia 库
    try:
        import nvidia.cudnn
        cudnn_path = os.path.dirname(nvidia.cudnn.__file__)
        print(f"\nnvidia.cudnn 路径: {cudnn_path}")
        
        cudnn_lib = os.path.join(cudnn_path, 'lib')
        if os.path.exists(cudnn_lib):
            print(f"✅ cuDNN 库目录存在: {cudnn_lib}")
            # 列出库文件
            lib_files = [f for f in os.listdir(cudnn_lib) if f.startswith('libcudnn')]
            print(f"   找到 {len(lib_files)} 个 cuDNN 库文件")
            for f in lib_files[:3]:  # 只显示前3个
                print(f"   - {f}")
        else:
            print(f"❌ cuDNN 库目录不存在")
            
    except ImportError:
        print("⚠️  nvidia.cudnn 未安装")
        print("   安装命令: pip install nvidia-cudnn-cu12")
    
    return True


def test_funasr_cuda():
    """测试 FunASR 模型在 CUDA 上的运行"""
    print("\n" + "="*80)
    print("测试 4: FunASR 模型 CUDA 推理")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过 FunASR CUDA 测试")
        return False
    
    try:
        from backend.modules.audio_analysis import FunASRService

        print("正在加载 FunASR 模型（CUDA）...")
        asr = FunASRService(device="cuda")
        print("✅ 模型加载成功")

        # 创建测试音频
        print("\n测试推理...")
        audio = np.random.randn(16000).astype(np.float32)

        text = asr.transcribe(audio, language="zh")
        print("✅ 推理成功")
        print(f"   识别文本长度: {len(text)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FunASR CUDA 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("CUDA/cuDNN 配置测试")
    print("="*80)
    
    results = []
    
    # 测试1: CUDA
    results.append(("CUDA 可用性", test_cuda_available()))
    
    # 测试2: cuDNN
    results.append(("cuDNN 可用性", test_cudnn_available()))
    
    # 测试3: 库路径
    results.append(("库路径配置", test_library_paths()))
    
    # 测试4: FunASR CUDA
    results.append(("FunASR CUDA", test_funasr_cuda()))
    
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
        print("🎉 所有测试通过！GPU 加速 ASR 已就绪")
        print("\n现在可以启动 API 服务:")
        print("  python -m backend.api.app")
        print("或使用脚本:")
        print("  bash scripts/start_api.sh")
    else:
        print("⚠️  部分测试失败")
        print("\n如果 cuDNN 测试失败，请尝试:")
        print("  1. pip install nvidia-cudnn-cu12")
        print("  2. 重新运行此测试")
        print("  3. 如果仍然失败，使用 CPU 模式（config/model_config.yaml: device: cpu）")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

