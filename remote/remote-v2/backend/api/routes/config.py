"""配置管理路由"""
import os
import sys
import signal
import yaml
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.utils.logger import get_logger
from backend.core.config import get_config

logger = get_logger(__name__)
router = APIRouter()

# 配置文件路径
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "model_config.yaml"


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    content: str
    restart: bool = False


@router.get("/config")
async def get_config_file():
    """获取当前配置文件内容"""
    try:
        if not CONFIG_PATH.exists():
            raise HTTPException(status_code=404, detail="配置文件不存在")
        
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"读取配置文件: {CONFIG_PATH}")
        return {
            "success": True,
            "content": content,
            "path": str(CONFIG_PATH)
        }
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取配置文件失败: {str(e)}")


@router.get("/config/file-settings")
async def get_file_settings():
    """获取文件上传配置（文件大小限制和允许的格式）"""
    try:
        cfg = get_config()
        return {
            "success": True,
            "data": {
                "max_audio_size": cfg.max_audio_size,
                "max_audio_size_mb": cfg.max_audio_size / (1024 * 1024),
                "allowed_extensions": list(cfg.allowed_extensions)
            }
        }
    except Exception as e:
        logger.error(f"获取文件配置失败: {e}")
        # 返回默认值
        return {
            "success": True,
            "data": {
                "max_audio_size": 524288000,  # 500MB
                "max_audio_size_mb": 500,
                "allowed_extensions": [".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".flac"]
            }
        }


@router.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """更新配置文件"""
    try:
        # 验证 YAML 格式
        try:
            yaml.safe_load(request.content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"YAML 格式错误: {str(e)}")
        
        # 备份原配置文件
        backup_path = CONFIG_PATH.with_suffix('.yaml.bak')
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            logger.info(f"备份配置文件到: {backup_path}")
        
        # 写入新配置
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(request.content)
        
        logger.info(f"配置文件已更新: {CONFIG_PATH}")
        
        response = {
            "success": True,
            "message": "配置文件已保存",
            "path": str(CONFIG_PATH)
        }
        
        # 如果需要重启服务
        if request.restart:
            logger.warning("收到重启服务请求，准备重启...")
            response["message"] = "配置文件已保存，服务即将重启..."
            
            # 延迟重启，给响应时间返回
            import threading
            def restart_server():
                import time
                time.sleep(1)  # 给响应时间返回
                logger.warning("正在重启服务...")
                # 使用 sys.exit() 退出，如果使用 uvicorn --reload 会自动重启
                # 或者使用 os.execv 重新启动当前进程
                python = sys.executable
                os.execv(python, [python] + sys.argv)
            
            threading.Thread(target=restart_server, daemon=True).start()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置文件失败: {str(e)}")
