#!/bin/bash
# 重启声纹识别系统（HTTPS模式）

echo "正在停止旧进程..."
pkill -f "python.*app.py"
sleep 2

echo "正在激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vioce

echo "正在启动HTTPS服务..."
cd /home/swufe/Project/zhoulonghao/code/software/web
nohup python app.py > server.log 2>&1 &

echo "等待服务启动..."
sleep 3

# 显示日志
echo "=========================================="
echo "服务启动完成！"
echo "=========================================="
tail -n 20 server.log

echo ""
echo "✅ HTTPS服务已启动！"
echo "📌 访问地址: https://10.8.21.50:51003"
echo "⚠️  浏览器会显示证书警告，点击\"继续访问\"即可"
echo ""
echo "查看日志: tail -f /home/swufe/Project/zhoulonghao/code/software/web/server.log"

