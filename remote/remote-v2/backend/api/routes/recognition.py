"""声纹识别路由"""
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Dict, Optional, List
import numpy as np

from backend.utils.logger import get_logger
from backend.utils.audio_utils import extract_audio_from_bytes, check_audio_energy
from backend.core.config import get_config
from backend.api.dependencies import get_pipeline

router = APIRouter()
logger = get_logger()


def validate_uploaded_file(audio: UploadFile) -> None:
    """
    验证上传的文件大小和格式
    
    Args:
        audio: 上传的文件对象
        
    Raises:
        HTTPException: 当文件大小或格式不符合要求时
    """
    cfg = get_config()
    
    # 检查文件扩展名
    if audio.filename:
        file_ext = Path(audio.filename).suffix.lower()
        if file_ext not in cfg.allowed_extensions:
            allowed_str = ', '.join(cfg.allowed_extensions)
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}，支持的格式: {allowed_str}"
            )
    
    # 注意：文件大小检查需要在读取文件后进行
    # 这里只检查格式，大小检查在读取后执行


@router.post("/register")
async def register_user(
    audio: UploadFile = File(..., description="音频文件 (WAV格式)"),
    user_id: str = Form(..., description="用户ID"),
    overwrite: bool = Form(False, description="是否覆盖已存在的用户"),
    save_audio: bool = Form(False, description="是否保存音频样本")
) -> Dict:
    """
    注册用户声纹（单样本模式）
    
    Args:
        audio: 音频文件
        user_id: 用户ID
        overwrite: 是否覆盖已存在的用户
        save_audio: 是否保存音频样本
        
    Returns:
        注册结果字典，包含success、user_id等字段
        
    Raises:
        HTTPException: 当服务未就绪(503)、音频能量过低、注册失败(400)或服务器错误(500)时
        AudioFormatError: 当音频格式解析失败时
    """
    try:
        cfg = get_config()
        
        # 验证文件格式
        validate_uploaded_file(audio)
        
        # 读取音频数据
        audio_bytes = await audio.read()
        logger.info(f"Received audio file: {len(audio_bytes)} bytes, filename: {audio.filename}")
        
        # 检查文件大小
        if len(audio_bytes) > cfg.max_audio_size:
            max_size_mb = cfg.max_audio_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"文件过大: {len(audio_bytes) / (1024 * 1024):.2f}MB，最大允许: {max_size_mb:.2f}MB"
            )
        
        # 检查音频数据是否为空
        if not audio_bytes or len(audio_bytes) == 0:
            logger.error("Empty audio bytes received")
            raise HTTPException(status_code=400, detail="Empty audio file received")
        
        # 解析音频
        audio_data = extract_audio_from_bytes(audio_bytes, sample_rate=16000)
        logger.info(f"Audio extracted: shape={audio_data.shape}, dtype={audio_data.dtype}, duration={len(audio_data)/16000:.2f}s")
        
        # 检查音频数据是否有效
        if audio_data is None or audio_data.size == 0:
            logger.error("Audio extraction resulted in empty array")
            raise HTTPException(status_code=400, detail="Failed to extract valid audio data")

        # RMS 门限过滤（静音/远端噪声直接拒绝）
        if not check_audio_energy(audio_data, threshold_db=getattr(cfg, "rms_gate_db", -25.0)):
            return {"success": False, "status": "silence", "message": "Audio energy too low"}
        
        # 获取流水线
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # 执行注册
        result = pipeline.register(
            audio=audio_data,
            user_id=user_id,
            sample_rate=16000,
            overwrite=overwrite,
            save_audio=save_audio
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-multi")
async def register_user_multi_sample(
    audio_files: List[UploadFile] = File(..., description="多个音频文件 (WAV格式，建议3个样本)"),
    user_id: str = Form(..., description="用户ID"),
    overwrite: bool = Form(False, description="是否覆盖已存在的用户"),
    save_audio: bool = Form(False, description="是否保存音频样本")
) -> Dict:
    """
    多样本注册用户声纹（推荐用于火锅店等嘈杂环境）
    
    建议录制3个样本：
    - 样本1: 安静环境
    - 样本2: 稍微吵闹环境
    - 样本3: 正常说话环境
    
    系统会计算平均向量，比单样本更稳定可靠
    
    Args:
        audio_files: 音频文件列表
        user_id: 用户ID
        overwrite: 是否覆盖已存在的用户
        save_audio: 是否保存音频样本
        
    Returns:
        注册结果字典，包含success、user_id等字段
        
    Raises:
        HTTPException: 当未提供音频文件(400)、服务未就绪(503)、注册失败(400)或服务器错误(500)时
        AudioFormatError: 当音频格式解析失败时
    """
    try:
        cfg = get_config()
        if not audio_files:
            raise HTTPException(status_code=400, detail="No audio files provided")
        
        # 读取所有音频数据
        audio_data_list = []
        for audio_file in audio_files:
            # 验证文件格式
            validate_uploaded_file(audio_file)
            
            audio_bytes = await audio_file.read()
            
            # 检查文件大小
            if len(audio_bytes) > cfg.max_audio_size:
                max_size_mb = cfg.max_audio_size / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"文件 {audio_file.filename} 过大: {len(audio_bytes) / (1024 * 1024):.2f}MB，最大允许: {max_size_mb:.2f}MB"
                )
            audio_data = extract_audio_from_bytes(audio_bytes, sample_rate=16000)
            if not check_audio_energy(audio_data, threshold_db=getattr(cfg, "rms_gate_db", -25.0)):
                logger.warning("Skipping low-energy sample during multi-register")
                continue
            audio_data_list.append(audio_data)
        
        # 获取流水线
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # 执行多样本注册
        result = pipeline.register_multi_sample(
            audio_list=audio_data_list,
            user_id=user_id,
            sample_rate=16000,
            overwrite=overwrite,
            save_audio=save_audio
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-sample registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize")
async def recognize_user(
    audio: UploadFile = File(..., description="音频文件 (WAV格式)"),
    threshold: Optional[float] = Form(None, description="相似度阈值"),
    device_id: Optional[str] = Form(None, description="设备ID（用于IoTDB存储，默认dev01）")
) -> Dict:
    """
    识别用户身份（支持V3.0对话智能引擎）
    
    Args:
        audio: 音频文件
        threshold: 相似度阈值（可选）
        use_v3: 是否使用V3.0引擎（默认True）
        
    Returns:
        识别结果字典，包含success、recognized、user_id等字段
        
    Raises:
        HTTPException: 当服务未就绪(503)、识别失败(400)或服务器错误(500)时
        AudioFormatError: 当音频格式解析失败时
    """
    try:
        cfg = get_config()
        
        # 验证文件格式
        validate_uploaded_file(audio)
        
        # 读取音频数据
        audio_bytes = await audio.read()
        
        # 检查文件大小
        if len(audio_bytes) > cfg.max_audio_size:
            max_size_mb = cfg.max_audio_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"文件过大: {len(audio_bytes) / (1024 * 1024):.2f}MB，最大允许: {max_size_mb:.2f}MB"
            )
        
        # 捕获请求开始时间（UTC）
        request_start_time = datetime.now(timezone.utc)
        import time
        processing_start = time.time()
        
        # 解析音频
        audio_data = extract_audio_from_bytes(audio_bytes, sample_rate=16000)

        # RMS 门限过滤
        if not check_audio_energy(audio_data, threshold_db=getattr(cfg, "rms_gate_db", -25.0)):
            return {"success": False, "status": "silence", "message": "Audio energy too low"}
        
        # 获取流水线
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # ========== V3.0 对话智能引擎（唯一引擎） ==========
        if not pipeline.enable_asr or not cfg.diarization_enabled:
            raise HTTPException(
                status_code=503, 
                detail="V3.0 engine requires ASR and diarization to be enabled"
            )
        
        import math
        
        logger.info("🚀 使用V3.0对话智能引擎")
        
        # 1. 并行感知 + 对话智能处理（Pipeline内部完成）
        parallel_result = pipeline.recognize_parallel_v3(
            audio=audio_data,
            sample_rate=16000,
            start_time=request_start_time,
            language="zh",
            beam_size=cfg.asr_beam_size,  # 从配置文件读取
            audio_filename=audio.filename  # 传递文件名用于日志
        )
        
        # 2. 获取对话流结果
        final_transcript = parallel_result.get('dialogue_segments')
        
        if not final_transcript:
            raise HTTPException(
                status_code=500,
                detail="Dialogue processing failed"
            )
        
        # 3. 写入IoTDB（如果启用）
        if cfg.iotdb_enabled and final_transcript:
            try:
                from backend.data.iotdb.connector import IoTDBConnector
                
                # 初始化IoTDB连接器（单例模式，可复用）
                iotdb_connector = IoTDBConnector()
                
                # 计算基准时间戳（毫秒）
                base_timestamp_ms = int(request_start_time.timestamp() * 1000)
                
                # 设备ID（从请求参数获取，或使用默认值）
                device_id_value = device_id or "dev01"
                
                # 批量写入对话流
                success = iotdb_connector.insert_dialogue_transcript(
                    device_id=device_id_value,
                    dialogue_segments=final_transcript,
                    base_timestamp=base_timestamp_ms
                )
                
                if success:
                    logger.info(
                        f"✅ 对话流已写入IoTDB: {device_id_value} | "
                        f"{len(final_transcript)} 个片段 | "
                        f"基准时间: {request_start_time.isoformat()}"
                    )
                else:
                    logger.warning("⚠️ IoTDB写入失败（模拟模式或连接失败）")
                    
            except Exception as e:
                logger.error(f"❌ IoTDB写入异常: {e}", exc_info=True)
                # 不中断主流程，只记录错误
        
        # 4. 构建响应
        if final_transcript:
            # 调试：打印 final_transcript 的内容
            logger.info(f"📊 Final Transcript ({len(final_transcript)} 个片段):")
            for i, seg in enumerate(final_transcript[:3]):  # 只打印前3个
                logger.info(f"  片段 {i}: speaker={seg.get('speaker')}, "
                           f"transcript='{seg.get('transcript', '')[:20]}...', "
                           f"avg_z_score={seg.get('avg_z_score')}, "
                           f"word_count={seg.get('word_count')}")
            
            # 找到主要说话人（词数最多的）
            speaker_counts = {}
            speaker_z_scores = {}
            
            for seg in final_transcript:
                speaker = seg['speaker']
                if speaker not in speaker_counts:
                    speaker_counts[speaker] = 0
                    speaker_z_scores[speaker] = []
                
                speaker_counts[speaker] += seg['word_count']
                speaker_z_scores[speaker].append(seg['avg_z_score'])
            
            primary_speaker = max(speaker_counts, key=speaker_counts.get)
            avg_z_score = float(np.mean(speaker_z_scores[primary_speaker]))
            
            # 🔧 转换 segments 字段以兼容前端（前端期望 user/user_id/text，后端返回 speaker/transcript）
            frontend_segments = []
            for seg in final_transcript:
                frontend_seg = {
                    'start': seg['start'],
                    'end': seg['end'],
                    'user': seg['speaker'],  # 前端期望的字段
                    'user_id': seg['speaker'],  # 备用字段
                    'text': seg.get('transcript', ''),  # ✅ 前端期望 text 字段，不是 transcript
                    'transcript': seg.get('transcript', ''),  # 保留 transcript 用于调试
                    'raw_z_score': seg.get('avg_z_score', 0.0),  # 前端期望的原始 Z-score
                    'score': min(100, max(0, seg.get('avg_z_score', 0.0) * 12.5)),  # 转换为百分比 (0-8 -> 0-100)
                    'word_count': seg.get('word_count', 0),
                    'duration': seg['end'] - seg['start']
                }
                frontend_segments.append(frontend_seg)
            
            # 计算处理时间
            processing_time = time.time() - processing_start
            
            result = {
                "success": True,
                "best_user": primary_speaker,
                "best_score": avg_z_score,
                "mode": "v3_diarization",
                "segments": frontend_segments,
                "total_speakers": len(speaker_counts),
                "detected_speakers": list(speaker_counts.keys()),
                "processing_time": round(processing_time, 2)
            }
            
            logger.info(f"✅ V3.0识别完成: {primary_speaker}, Z-score={avg_z_score:.2f}")
        else:
            result = {
                "success": False,
                "error": "No valid segments after diarization"
            }
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Recognition failed"))
        
        # 🔧 将内部格式转换为前端期望的格式
        speaker_id = result.get("best_user") or result.get("primary_speaker") or "unknown"
        z_score = result.get("best_score", 0.0)
        
        import math
        confidence = 100.0 / (1.0 + math.exp(-z_score + 2.5))
        
        frontend_response = {
            "success": True,
            "data": {
                "speaker_id": speaker_id,
                "confidence": round(confidence, 2)
            },
            # 保留详细信息供调试
            "details": {
                "mode": result.get("mode"),
                "best_score": z_score,
                "detected_speakers": result.get("detected_speakers", []),
                "total_speakers": result.get("total_speakers", 0),
                "segments": result.get("segments", []),
                "speaker_stats": result.get("speaker_stats", {}),
                "processing_time": result.get("processing_time", 0)
            }
        }
        
        return frontend_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_users() -> Dict:
    """
    列出所有注册用户（包含详细信息）
    
    Returns:
        用户列表字典，格式为{"success": true, "data": {"speakers": [...], "count": N}}
        
    Raises:
        HTTPException: 当服务未就绪(503)或服务器错误(500)时
    """
    try:
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        user_ids = pipeline.list_users()
        
        # 获取每个用户的详细信息
        users_detail = []
        for user_id in user_ids:
            user_info = pipeline.get_user_info(user_id)
            if user_info:
                metadata = user_info.get("metadata", {})
                users_detail.append({
                    "speaker_id": user_id,
                    "registered_at": metadata.get("registered_at"),
                    "file_size": metadata.get("file_size"),
                    "num_samples": metadata.get("num_samples", 1),
                    "registration_mode": metadata.get("registration_mode", "single-sample")
                })
        
        # 返回格式符合前端期望：{"success": true, "data": {"speakers": [...]}}
        return {
            "success": True,
            "data": {
                "speakers": users_detail,
                "count": len(users_detail)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List users error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user_info(user_id: str) -> Dict:
    """
    获取用户信息
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户信息字典，包含user_id、metadata、embedding_shape等字段
        
    Raises:
        HTTPException: 当服务未就绪(503)、用户不存在(404)或服务器错误(500)时
    """
    try:
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        user_info = pipeline.get_user_info(user_id)
        
        if user_info is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {
            "user_id": user_id,
            "metadata": user_info.get("metadata", {}),
            "embedding_shape": user_info.get("embedding", np.array([])).shape
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user(user_id: str) -> Dict:
    """
    删除用户
    
    Args:
        user_id: 用户ID
        
    Returns:
        删除结果字典，包含success、user_id、message等字段
        
    Raises:
        HTTPException: 当服务未就绪(503)、用户不存在(404)或服务器错误(500)时
    """
    try:
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        success = pipeline.delete_user(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "User deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
