#!/bin/bash
# Remote-V2 API 服务启动脚本（支持自动重启）

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

echo "🚀 启动 Remote-V2 API 服务..."
echo "📁 工作目录: $(pwd)"
echo "🐍 Python 版本: $(python --version)"
echo "⏰ 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================
# 配置动态库路径（支持 CUDA/cuDNN）
# ============================================================
echo "🔧 配置动态库路径..."

# 1. 自动获取 nvidia 相关库的路径
CUDNN_LIB_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
CUBLAS_LIB_PATH=$(python -c "import nvidia.cublas; import os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib'))" 2>/dev/null)

# 2. 将路径加入 LD_LIBRARY_PATH
if [ -n "$CUDNN_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_LIB_PATH:$LD_LIBRARY_PATH
    echo "   ✅ cuDNN 库路径: $CUDNN_LIB_PATH"
else
    echo "   ⚠️  未找到 cuDNN 库（如需 GPU 加速 ASR，请安装 nvidia-cudnn-cu12）"
fi

if [ -n "$CUBLAS_LIB_PATH" ]; then
    export LD_LIBRARY_PATH=$CUBLAS_LIB_PATH:$LD_LIBRARY_PATH
    echo "   ✅ cuBLAS 库路径: $CUBLAS_LIB_PATH"
fi

# 3. 将 Conda 自身的库路径也加上（双重保险）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "   ✅ Conda 库路径: $CONDA_PREFIX/lib"

echo ""

# 无限循环，支持自动重启
while true; do
    echo "▶️  启动 API 服务..."
    python -m backend.api.app
    
    EXIT_CODE=$?
    echo ""
    echo "⚠️  服务已停止 (退出码: $EXIT_CODE)"
    echo "⏰ 停止时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 如果是正常退出（Ctrl+C），则不重启
    if [ $EXIT_CODE -eq 130 ]; then
        echo "👋 检测到用户中断，退出..."
        break
    fi
    
    # 等待 3 秒后重启
    echo "⏳ 3 秒后自动重启..."
    sleep 3
    echo ""
done

echo "✅ 服务已完全停止"

