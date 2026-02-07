"""健康检查接口"""
from fastapi import APIRouter, Depends

from backend.api.dependencies import get_pipeline
from backend.pipeline.inference_pipeline import InferencePipeline


router = APIRouter()


@router.get("/health")
async def health_check(pipeline: InferencePipeline = Depends(get_pipeline)):
    """
    健康检查接口
    
    Returns:
        系统状态信息
    """
    try:
        # 获取已注册用户数量
        users = pipeline.list_users()
        
        return {
            "status": "ok",
            "model_loaded": True,
            "registered_users": len(users),
            "version": "14.2"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "14.2"
        }

