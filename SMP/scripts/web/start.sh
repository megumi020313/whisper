#!/bin/bash
# 启动网页版声纹识别系统

echo "=========================================="
echo "  网页版声纹识别系统 - 启动脚本"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误：未找到Python"
    exit 1
fi

echo "✅ Python版本: $(python --version)"

# 检查依赖
echo ""
echo "检查依赖..."
pip list | grep -i flask > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask未安装，正在安装..."
    pip install -r requirements.txt
fi

# 启动服务
echo ""
echo "=========================================="
echo "  启动Flask服务..."
echo "=========================================="
echo ""
echo "服务地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

python app.py

