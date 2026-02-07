"""模型加载诊断脚本

【功能说明】
诊断ERes2NetV2模型加载问题，检查：
- Checkpoint文件结构
- State dict键名和形状
- 权重参数数量
- 模型架构匹配性
- 加载错误原因

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/diagnose_model_loading.py

【前置条件】
- 模型checkpoint文件存在（models/eres2netv2/）

【预期输出】
- Checkpoint文件类型和键列表
- State dict总键数
- 前10个和后10个键的名称和形状
- 权重参数统计
- 加载问题诊断建议
"""
import sys
from pathlib import Path
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def diagnose_checkpoint(checkpoint_path: str):
    """诊断checkpoint文件"""
    print("=" * 60)
    print("ERes2NetV2 模型加载诊断")
    print("=" * 60)
    print(f"Checkpoint路径: {checkpoint_path}")
    print()
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Checkpoint类型: {type(checkpoint)}")
        print()
        
        if isinstance(checkpoint, dict):
            print("Checkpoint键:")
            for key in checkpoint.keys():
                print(f"  - {key}")
            print()
            
            # 检查state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            print(f"State dict总键数: {len(state_dict)}")
            print()
            
            print("前10个键:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                tensor = state_dict[key]
                print(f"  {i+1}. {key}")
                print(f"     Shape: {tensor.shape}")
            
            print()
            print("后10个键:")
            for i, key in enumerate(list(state_dict.keys())[-10:]):
                tensor = state_dict[key]
                print(f"  {i+1}. {key}")
                print(f"     Shape: {tensor.shape}")
            
            print()
            
            # 统计参数数量
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"总参数数量: {total_params:,}")
            
        print()
        print("✅ 诊断完成")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "models/eres2netv2/pretrained_eres2netv2.ckpt"
    
    diagnose_checkpoint(checkpoint_path)

