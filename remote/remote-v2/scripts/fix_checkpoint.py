"""权重清洗脚本 - 从 Fusion 版本提取标准版权重

【功能说明】
从Fusion版本的checkpoint中提取标准版ERes2NetV2权重：
- 移除fuse_models分支（保留主干）
- 移除DDP训练的module.前缀
- 清洗不兼容的权重参数
- 生成干净的标准版权重文件

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/fix_checkpoint.py

【输入文件】
models/eres2netv2/pretrained_eres2netv2.ckpt - 原始Fusion版checkpoint

【输出文件】
models/eres2netv2/eres2netv2_clean.pth - 清洗后的标准版权重

【清洗策略】
- 提取fuse_models.0.为主干参数
- 丢弃fuse_models.1, fuse_models.2等辅助分支
- 移除module.前缀（DDP残留）
- 保持参数名称和形状不变

【使用场景】
- 模型权重不兼容
- 从多模型融合版本提取单模型
- 权重文件格式转换
"""
import torch
import collections
from pathlib import Path

def fix_checkpoint():
    # 1. 配置路径
    input_ckpt = "models/eres2netv2/pretrained_eres2netv2.ckpt"
    output_ckpt = "models/eres2netv2/eres2netv2_clean.pth"
    
    print(f"🔍 加载原始 checkpoint: {input_ckpt}")
    checkpoint = torch.load(input_ckpt, map_location="cpu")
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"📦 原始键数量: {len(state_dict)}")
    print(f"📦 原始参数总数: {sum(v.numel() for v in state_dict.values()):,}")
    
    new_state_dict = collections.OrderedDict()
    removed_keys = []
    
    # 2. 智能清洗逻辑
    for k, v in state_dict.items():
        name = k
        
        # 情况 A: Fusion 模型 (含有 fuse_models)
        if "fuse_models" in name:
            if "fuse_models.0." in name:
                # 去掉前缀，保留主干参数
                name = name.replace("fuse_models.0.", "")
                print(f"✂️  提取主干: {k} -> {name}")
            else:
                # 丢弃 fuse_models.1, fuse_models.2 等辅助分支
                removed_keys.append(k)
                continue
        
        # 情况 B: 移除 module. 前缀 (DDP 训练残留)
        if name.startswith("module."):
            name = name.replace("module.", "")
        
        new_state_dict[name] = v
    
    print(f"\n📊 清洗结果:")
    print(f"  清洗后键数量: {len(new_state_dict)}")
    print(f"  清洗后参数总数: {sum(v.numel() for v in new_state_dict.values()):,}")
    print(f"  移除的键数量: {len(removed_keys)}")
    
    if removed_keys:
        print(f"\n🗑️  移除的键 (前10个):")
        for key in removed_keys[:10]:
            print(f"    - {key}")
    
    # 检查是否有 fuse_models
    has_fuse = any("fuse_models" in k for k in state_dict.keys())
    print(f"\n🔍 检查 fuse_models: {'存在' if has_fuse else '不存在'}")
    
    # 3. 保存清洗后的权重
    output_path = Path(output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_ckpt)
    
    print(f"\n✅ 清洗后的 checkpoint 已保存到: {output_ckpt}")
    print(f"💡 提示: 如果原始 checkpoint 没有 fuse_models，则清洗前后几乎相同")

if __name__ == "__main__":
    fix_checkpoint()

