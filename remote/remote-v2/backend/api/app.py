"""FastAPI 应用主入口"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ============================================================
# 配置动态库路径（支持 CUDA/cuDNN）- 手动启动时也生效
# ============================================================
def setup_cuda_libraries():
    """配置 CUDA 相关库的动态链接路径"""
    try:
        # 获取 nvidia cuDNN 库路径
        import nvidia.cudnn
        cudnn_lib_path = os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib')
        
        # 获取 nvidia cuBLAS 库路径
        import nvidia.cublas
        cublas_lib_path = os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib')
        
        # 更新 LD_LIBRARY_PATH
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = [cudnn_lib_path, cublas_lib_path]
        
        # 添加 Conda 库路径
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            new_paths.append(os.path.join(conda_prefix, 'lib'))
        
        # 合并路径
        if current_ld_path:
            new_paths.append(current_ld_path)
        
        os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
        
        print(f"✅ CUDA 库路径已配置:")
        print(f"   - cuDNN: {cudnn_lib_path}")
        print(f"   - cuBLAS: {cublas_lib_path}")
        if conda_prefix:
            print(f"   - Conda: {os.path.join(conda_prefix, 'lib')}")
        
        return True
    except ImportError as e:
        print(f"⚠️  未找到 CUDA 库: {e}")
        print(f"   如需 GPU 加速 ASR，请安装: pip install nvidia-cudnn-cu12")
        return False
    except Exception as e:
        print(f"⚠️  配置 CUDA 库路径时出错: {e}")
        return False

# 在导入其他模块前配置库路径
setup_cuda_libraries()

from backend.utils.logger import setup_logger, get_logger
from backend.api.dependencies import get_pipeline
from backend.api.routes import recognition, health, config, smp_proxy
from backend.core.config import get_config


# 设置日志
setup_logger()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting application...")
    
    # 启动时初始化推理流水线
    try:
        pipeline = get_pipeline()
        logger.info("✅ Inference pipeline initialized")
    except Exception as e:
        logger.error(f"Failed to initialize inference pipeline: {e}")
    
    yield
    
    # 关闭时清理资源
    logger.info("Shutting down application...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Remote-V2 声纹识别系统",
    description="基于 ERes2NetV2 和 Silero VAD 的声纹识别系统",
    version="14.2",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/api/v1", tags=["健康检查"])
app.include_router(recognition.router, prefix="/api/v1/speaker", tags=["声纹识别"])
app.include_router(smp_proxy.router, prefix="/api/v1", tags=["SMP Proxy"])
app.include_router(config.router, prefix="/api/v1", tags=["配置管理"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Remote-V2 声纹识别系统",
        "version": "14.2",
        "docs": "/docs"
    }


if __name__ == "__main__":
    logger.info("Starting FastAPI server...")

    # 读取服务器配置
    config_file = Path(__file__).resolve().parent.parent.parent / "config" / "server_config.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            server_config = yaml.safe_load(f)
            api_config = server_config.get('api', {})
            host = api_config.get('host', '0.0.0.0')
            port = api_config.get('port', 8000)
            workers = api_config.get('workers', 1)
            timeout = api_config.get('timeout', 300)
    else:
        logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
        host = '0.0.0.0'
        port = 8000
        workers = 1
        timeout = 300

    logger.info(f"服务器配置: host={host}, port={port}, workers={workers}, timeout={timeout}")

    # SSL 证书路径
    cert_dir = Path(__file__).resolve().parent.parent.parent / "config" / "certs"
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    # 检查证书是否存在
    if cert_file.exists() and key_file.exists():
        logger.info("🔒 HTTPS enabled with self-signed certificate")
        uvicorn.run(
            "backend.api.app:app",
            host=host,
            port=port,
            workers=workers,
            reload=False,
            log_level="info",
            ssl_certfile=str(cert_file),
            ssl_keyfile=str(key_file)
        )
    else:
        logger.info("⚠️  Running with HTTP (no SSL certificate found)")
        uvicorn.run(
            "backend.api.app:app",
            host=host,
            port=port,
            workers=workers,
            reload=False,
            log_level="info"
        )

