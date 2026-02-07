#!/bin/bash
# Remote-V2 Web 前端启动脚本

cd /home/swufe/Project/zhoulonghao/remote/remote-v2

# ============================================================
# 激活 conda 环境
# ============================================================
# 初始化 conda（如果还没初始化）
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$CONDA_PREFIX/../etc/profile.d/conda.sh" ]; then
    source "$CONDA_PREFIX/../etc/profile.d/conda.sh"
fi

# 激活 vioce 环境
conda activate vioce 2>/dev/null || echo "⚠️  conda 环境激活失败，使用当前环境"

# 激活 conda 环境
source /home/swufe/miniconda3/etc/profile.d/conda.sh
conda activate vioce

echo "🚀 启动 Remote-V2 Web 前端..."
echo "📁 工作目录: $(pwd)"
echo "🐍 Python 版本: $(python --version)"
echo "⏰ 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 启动 Web 前端
python web/app.py

echo "✅ Web 前端已停止"

