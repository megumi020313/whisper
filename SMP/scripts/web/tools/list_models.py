#!/usr/bin/env python3
"""
列出所有可用的声纹识别模型及其性能信息
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training import TRAINING_DIR

MODELS_DIR = TRAINING_DIR / "models"


def list_models():
    """列出所有可用模型及其性能"""
    print("=" * 80)
    print("声纹识别模型列表")
    print("=" * 80)
    
    if not MODELS_DIR.exists():
        print("❌ 模型目录不存在")
        return
    
    # 查找所有模型
    model_dirs = sorted(MODELS_DIR.glob("speechbrain_model_*"), key=lambda p: p.name)
    
    if not model_dirs:
        print("❌ 未找到任何模型")
        return
    
    print(f"\n找到 {len(model_dirs)} 个模型:\n")
    
    models_info = []
    
    for i, model_dir in enumerate(model_dirs, 1):
        model_name = model_dir.name
        embeddings_file = model_dir / "speaker_embeddings.json"
        performance_file = model_dir / "performance_report.json"
        
        # 基本信息
        info = {
            "name": model_name,
            "path": str(model_dir),
            "has_embeddings": embeddings_file.exists(),
            "has_performance": performance_file.exists(),
        }
        
        # 读取性能信息
        if performance_file.exists():
            try:
                with open(performance_file, 'r') as f:
                    perf_data = json.load(f)
                    info["best_accuracy"] = perf_data.get("best_performance", {}).get("best_val_accuracy")
                    info["best_epoch"] = perf_data.get("best_performance", {}).get("best_val_epoch")
                    info["total_epochs"] = perf_data.get("training_summary", {}).get("total_epochs")
            except Exception as e:
                info["performance_error"] = str(e)
        
        # 读取说话人信息
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    embeddings = json.load(f)
                    info["num_speakers"] = len(embeddings)
                    info["speakers"] = list(embeddings.keys())
            except Exception as e:
                info["embeddings_error"] = str(e)
        
        # 获取文件修改时间
        if embeddings_file.exists():
            mtime = embeddings_file.stat().st_mtime
            info["modified_time"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        models_info.append(info)
        
        # 显示信息
        print(f"[{i}] {model_name}")
        print(f"    路径: {model_dir}")
        print(f"    状态: {'✅ 可用' if info['has_embeddings'] else '❌ 缺少embeddings文件'}")
        
        if info.get("best_accuracy"):
            accuracy = info["best_accuracy"]
            print(f"    验证准确率: {accuracy:.2f}% (第{info.get('best_epoch')}轮，共{info.get('total_epochs')}轮)")
        
        if info.get("num_speakers"):
            print(f"    注册用户数: {info['num_speakers']}")
            print(f"    用户列表: {', '.join(info['speakers'][:5])}" + 
                  (f"... (共{info['num_speakers']}个)" if info['num_speakers'] > 5 else ""))
        
        if info.get("modified_time"):
            print(f"    更新时间: {info['modified_time']}")
        
        print()
    
    # 找出最佳模型
    best_model = max(
        (m for m in models_info if m.get("best_accuracy")),
        key=lambda m: m["best_accuracy"],
        default=None
    )
    
    if best_model:
        print("=" * 80)
        print(f"🏆 推荐模型: {best_model['name']}")
        print(f"   验证准确率: {best_model['best_accuracy']:.2f}%")
        print(f"   注册用户数: {best_model.get('num_speakers', 0)}")
        print("=" * 80)
    
    # 显示配置建议
    print("\n📝 配置建议:")
    print("   在 config.py 中设置:")
    if best_model:
        print(f'   SELECTED_MODEL = "{best_model["name"]}"')
    print("\n   或设置为 None 使用最新模型:")
    print(f'   SELECTED_MODEL = None')
    print()


if __name__ == "__main__":
    try:
        list_models()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

