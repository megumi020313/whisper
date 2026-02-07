from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import httpx
from backend.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/smp/recognize")
async def proxy_smp_recognize(audio: UploadFile = File(...)):
    """
    [Proxy] 转发请求到 SMP 服务 (Port 51003)
    """
    smp_url = "http://127.0.0.1:51003/api/v1/speaker/recognize"
    
    try:
        logger.info(f"Forwarding recognition request to SMP service: {smp_url}")
        
        # 读取文件内容
        content = await audio.read()
        
        async with httpx.AsyncClient() as client:
            # 构造 multipart/form-data 请求
            # 注意：httpx 需要 (filename, content, content_type) 元组
            files = {'audio': (audio.filename, content, audio.content_type)}
            
            # 发送请求到 SMP
            response = await client.post(smp_url, files=files, timeout=30.0)
            
            # 返回 SMP 的响应
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
            
    except httpx.RequestError as exc:
        logger.error(f"SMP Service Connection Error: {exc}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"SMP Service Unavailable",
                "details": f"Could not connect to {smp_url}. Ensure SMP service is running on port 51003."
            },
            status_code=502
        )
    except Exception as exc:
        logger.error(f"SMP Proxy Error: {exc}")
        return JSONResponse(
            content={"success": False, "error": f"Proxy Error: {str(exc)}"},
            status_code=500
        )

