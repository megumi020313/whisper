"""Redis 服务管理工具"""
import subprocess
import time
from typing import Tuple

import redis

from backend.utils.logger import get_logger

logger = get_logger()


class RedisManager:
    """Redis 服务管理器，负责检查和启动 Redis 服务"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
    
    def is_redis_running(self) -> bool:
        """检查 Redis 服务是否正在运行"""
        try:
            r = redis.Redis(host=self.host, port=self.port, db=self.db, socket_connect_timeout=2)
            r.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            return False
    
    def check_redis_installed(self) -> bool:
        """检查 Redis 是否已安装"""
        try:
            result = subprocess.run(
                ["which", "redis-server"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to check redis-server installation: {e}")
            return False
    
    def start_redis_service(self) -> Tuple[bool, str]:
        """尝试启动 Redis 服务
        
        Returns:
            (success, message): 是否成功启动和消息
        """
        # 检查是否已经运行
        if self.is_redis_running():
            return True, "Redis is already running"
        
        # 检查是否已安装
        if not self.check_redis_installed():
            msg = (
                "❌ Redis server is not installed. Please install it first:\n"
                "   Ubuntu/Debian: sudo apt install redis-server\n"
                "   Or use Docker: docker run -d --name redis -p 6379:6379 redis:latest"
            )
            return False, msg
        
        # 尝试启动服务
        logger.info("🔄 Attempting to start Redis service...")
        
        # 方法1：尝试使用 systemctl
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "redis-server"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 等待服务启动
                time.sleep(2)
                if self.is_redis_running():
                    logger.info("✅ Redis service started successfully via systemctl")
                    return True, "Redis service started successfully"
                else:
                    logger.warning("⚠️ systemctl command succeeded but Redis is not responding")
            else:
                logger.warning(f"⚠️ systemctl failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ systemctl command timed out")
        except FileNotFoundError:
            logger.warning("⚠️ systemctl not found")
        except Exception as e:
            logger.warning(f"⚠️ systemctl failed: {e}")
        
        # 方法2：尝试直接启动 redis-server（后台运行）
        try:
            logger.info("🔄 Attempting to start redis-server directly...")
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 等待服务启动
            time.sleep(2)
            if self.is_redis_running():
                logger.info("✅ Redis server started successfully in daemon mode")
                return True, "Redis server started successfully"
        except FileNotFoundError:
            logger.error("❌ redis-server command not found")
        except Exception as e:
            logger.error(f"❌ Failed to start redis-server: {e}")
        
        # 所有方法都失败
        msg = (
            "❌ Failed to start Redis service. Please start it manually:\n"
            "   Method 1: sudo systemctl start redis-server\n"
            "   Method 2: redis-server --daemonize yes\n"
            "   Method 3: docker run -d --name redis -p 6379:6379 redis:latest"
        )
        return False, msg
    
    def ensure_redis_available(self, auto_start: bool = True) -> Tuple[bool, str]:
        """确保 Redis 可用
        
        Args:
            auto_start: 如果 Redis 未运行，是否尝试自动启动
        
        Returns:
            (available, message): Redis 是否可用和消息
        """
        # 检查是否正在运行
        if self.is_redis_running():
            logger.info("✅ Redis is running and available")
            return True, "Redis is available"
        
        # 如果不自动启动，直接返回
        if not auto_start:
            msg = "❌ Redis is not running and auto_start is disabled"
            logger.warning(msg)
            return False, msg
        
        # 尝试启动
        logger.info("⚠️ Redis is not running, attempting to start...")
        return self.start_redis_service()

