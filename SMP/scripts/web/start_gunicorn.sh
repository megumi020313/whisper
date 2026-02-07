#!/bin/bash

# 定义变量
APP_DIR="/home/swufe/Project/zhoulonghao/smp/code/software/web"
LOG_FILE="$APP_DIR/gunicorn.log"
CONDA_ENV_NAME="vioce"

echo "正在停止旧进程..."
pkill -f "gunicorn.*app:app"
pkill -f "python.*app.py"
sleep 2

echo "正在激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ 激活conda环境失败"
    exit 1
fi

echo "正在启动Gunicorn HTTPS服务..."
cd "$APP_DIR" || exit

# 使用gunicorn启动（4个worker进程，超时时间120秒）
nohup gunicorn \
    -w 4 \
    -b 0.0.0.0:51003 \
    --certfile=cert.pem \
    --keyfile=key.pem \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app:app > "$LOG_FILE" 2>&1 &

echo "等待服务启动..."
sleep 5

echo "=========================================="
echo "服务启动完成！"
echo "=========================================="

if pgrep -f "gunicorn.*app:app" > /dev/null; then
    echo "✅ Gunicorn HTTPS服务已启动！"
    echo "📌 访问地址: https://10.8.21.33:51003"
    echo "⚠️  浏览器会显示证书警告，点击\"继续访问\"即可"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
else
    echo "❌ 服务启动失败，请检查日志: $LOG_FILE"
fi

