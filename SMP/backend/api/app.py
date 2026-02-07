#!/usr/bin/env python3
"""
Flask应用主入口 - 网页版声纹识别系统
提供录音上传、声纹注册、声纹识别的 API 接口
"""
import sys
from pathlib import Path
from typing import Optional
import tempfile
import time
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
import soundfile as sf

# 导入项目模块（使用绝对路径）
from backend.training.add_speaker_template import add_speaker_template
from backend.modules.speaker_id.speaker_id_wrapper import SpeakerIDWrapper

# 导入web模块（使用绝对路径）
from backend.config.web_config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, CORS_ORIGINS,
    SAMPLE_RATE, UPLOAD_DIR, AUDIO_STORAGE_DIR,
    MAX_AUDIO_SIZE, ALLOWED_EXTENSIONS, Config
)
from backend.utils.audio_validator import AudioValidator
from backend.utils.audio_storage import AudioStorage

# 创建Flask应用
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_AUDIO_SIZE

# 启用CORS
CORS(app, origins=CORS_ORIGINS)

# 初始化音频存储管理器
storage = AudioStorage(
    storage_dir=AUDIO_STORAGE_DIR,
    metadata_file=AUDIO_STORAGE_DIR / "metadata.json"
)

# 全局变量：说话人识别器
_speaker_id_wrapper: Optional[SpeakerIDWrapper] = None


def get_speaker_id_wrapper() -> Optional[SpeakerIDWrapper]:
    """获取说话人识别器实例"""
    global _speaker_id_wrapper
    
    if _speaker_id_wrapper is None:
        # 查找最新的模型文件
        model_path = Config.get_latest_speaker_embeddings()
        if model_path is None:
            return None
        
        try:
            # 从模型配置读取参数
            from backend.config.model_config import ModelConfig
            _speaker_id_wrapper = SpeakerIDWrapper(
                model_path=str(model_path),
                sample_rate=ModelConfig.get_sample_rate(),
                multi_speaker=ModelConfig.get_multi_speaker(),
                encoder_root=str(ModelConfig.get_encoder_root())
            )
        except Exception as e:
            print(f"加载声纹识别模型失败: {e}")
            traceback.print_exc()
            return None
    
    return _speaker_id_wrapper


def reload_speaker_id_wrapper() -> None:
    """重新加载说话人识别器（注册新用户后需要调用）"""
    global _speaker_id_wrapper
    _speaker_id_wrapper = None


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ==================== 错误处理 ====================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """处理文件过大错误"""
    max_size_mb = MAX_AUDIO_SIZE / (1024 * 1024)
    return jsonify({
        "success": False,
        "error": "文件过大",
        "details": f"上传的音频文件超过了最大限制 ({max_size_mb:.1f}MB)。请使用较小的音频文件。"
    }), 413


# ==================== Web页面路由 ====================

@app.route('/')
def index():
    """返回系统主页面"""
    return render_template('index.html')


@app.route('/test-recorder')
def test_recorder():
    """返回录音诊断页面"""
    return render_template('test_recorder.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)


# ==================== API路由 ====================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """健康检查"""
    model_path = Config.get_latest_speaker_embeddings()
    return jsonify({
        "status": "ok",
        "model_loaded": model_path is not None,
        "model_path": str(model_path) if model_path else None
    })


@app.route('/api/v1/speaker/register', methods=['POST'])
def register_speaker():
    """
    注册新声纹
    
    请求：FormData(speaker_id, audio)
    响应：JSON结果
    """
    try:
        # 检查请求数据
        if 'speaker_id' not in request.form:
            return jsonify({"error": "缺少 speaker_id 参数"}), 400
        
        if 'audio' not in request.files:
            return jsonify({"error": "缺少音频文件"}), 400
        
        speaker_id = request.form['speaker_id'].strip()
        audio_file = request.files['audio']
        
        # 验证speaker_id
        if not speaker_id:
            return jsonify({"error": "speaker_id 不能为空"}), 400
        
        if not audio_file or audio_file.filename == '':
            return jsonify({"error": "未选择文件"}), 400
        
        # 检查文件扩展名（允许配置的格式）
        if not allowed_file(audio_file.filename):
            return jsonify({
                "error": f"不支持的文件格式: {Path(audio_file.filename).suffix}",
                "details": f"支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # 保存上传的文件到临时位置（保持原始扩展名）
        timestamp = int(time.time() * 1000)
        original_ext = Path(audio_file.filename).suffix or '.wav'
        temp_path = UPLOAD_DIR / f"{speaker_id}_temp_{timestamp}{original_ext}"
        audio_file.save(str(temp_path))
        
        try:
            # 先尝试验证音频格式（如果是WAV且格式正确）
            is_valid, message = AudioValidator.validate_audio_file(temp_path)
            
            if not is_valid:
                # 尝试转换格式（支持所有ffmpeg能处理的格式，包括MP4、MP3等）
                converted_path = UPLOAD_DIR / f"{speaker_id}_converted_{timestamp}.wav"
                success, conv_message = AudioValidator.convert_to_required_format(
                    temp_path, converted_path
                )
                
                if not success:
                    temp_path.unlink()
                    return jsonify({
                        "error": f"音频格式转换失败",
                        "details": {
                            "validation": message,
                            "conversion": conv_message
                        }
                    }), 400
                
                # 使用转换后的文件
                temp_path.unlink()
                temp_path = converted_path
            # 如果验证通过，直接使用原文件
            
            # 查找模型文件
            model_path = Config.get_latest_speaker_embeddings()
            if model_path is None:
                # 如果没有模型文件，创建一个空的
                model_path = AUDIO_STORAGE_DIR.parent / "speaker_embeddings.json"
                import json
                with open(model_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
            
            # 调用现有的 add_speaker_template 函数进行注册
            add_speaker_template(
                speaker_id=speaker_id,
                audio_files=[temp_path],
                model_path=model_path,
                sample_rate=SAMPLE_RATE,
                device="cpu"
            )
            
            # 保存音频文件到存储目录
            saved_path = storage.save_audio(speaker_id, temp_path, overwrite=True)
            
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            
            # 重新加载识别模型
            reload_speaker_id_wrapper()
            
            return jsonify({
                "success": True,
                "message": "声纹注册成功",
                "data": {
                    "speaker_id": speaker_id,
                    "audio_path": str(saved_path.relative_to(AUDIO_STORAGE_DIR.parent)) if saved_path else None,
                    "model_path": str(model_path)
                }
            })
            
        except Exception as e:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise e
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "注册失败",
            "details": str(e)
        }), 500


@app.route('/api/v1/speaker/recognize', methods=['POST'])
def recognize_speaker():
    """
    识别声纹
    
    请求：FormData(audio) 或二进制音频数据
    响应：JSON结果（匹配用户名和相似度分数）
    """
    try:
        # 获取识别器
        wrapper = get_speaker_id_wrapper()
        if wrapper is None:
            return jsonify({
                "error": "声纹识别模型未加载",
                "details": "请先注册至少一个用户"
            }), 400
        
        # 获取音频数据
        audio_data = None
        temp_path = None
        converted_path = None
        
        if 'audio' in request.files:
            # FormData方式上传
            audio_file = request.files['audio']
            
            if not audio_file or audio_file.filename == '':
                return jsonify({"error": "未选择文件"}), 400
            
            # 保存临时文件（保持原始格式）
            timestamp = int(time.time() * 1000)
            original_ext = Path(audio_file.filename).suffix or '.webm'
            temp_path = UPLOAD_DIR / f"recognize_temp_{timestamp}{original_ext}"
            audio_file.save(str(temp_path))
            
            # 尝试直接读取（如果是WAV格式）
            try:
                audio_data, sr = sf.read(str(temp_path), dtype='int16')
            except Exception as e:
                # 如果读取失败，尝试转换格式
                print(f"直接读取失败，尝试转换格式: {e}")
                converted_path = UPLOAD_DIR / f"recognize_converted_{timestamp}.wav"
                success, message = AudioValidator.convert_to_required_format(
                    temp_path, converted_path
                )
                
                if not success:
                    return jsonify({
                        "error": "音频格式转换失败",
                        "details": message
                    }), 400
                
                # 读取转换后的音频
                audio_data, sr = sf.read(str(converted_path), dtype='int16')
                
                # 清理原始临时文件
                if temp_path.exists():
                    temp_path.unlink()
                temp_path = converted_path
            
        else:
            # 二进制数据方式
            audio_bytes = request.get_data()
            if len(audio_bytes) == 0:
                return jsonify({"error": "未收到音频数据"}), 400
            
            # 将字节转换为numpy数组
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 计算音频时长（秒）
            audio_duration = len(audio_data) / sr if sr else 0
            
            # 使用简化版 diarization 进行分段识别
            from backend.modules.diarization_simple import SimpleDiarization
            
            # 初始化 diarization
            diarization = SimpleDiarization(
                window_size=2.0,      # 2秒窗口
                step_size=1.0,        # 1秒步长
                min_segment_duration=0.5,  # 最小0.5秒
                merge_threshold=0.3   # 0.3秒合并阈值
            )
            
            # 执行分段识别
            segments = diarization.process(audio_data, sr, wrapper)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 统计说话人
            detected_speakers = list(set([seg['user'] for seg in segments if seg['user'] != 'unknown']))
            total_speakers = len(detected_speakers)
            
            # 确定主要说话人（占用时间最长的）
            if segments:
                speaker_durations = {}
                for seg in segments:
                    user = seg['user']
                    duration = seg['end'] - seg['start']
                    speaker_durations[user] = speaker_durations.get(user, 0) + duration
                
                main_speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
                
                # 计算主要说话人的平均置信度
                main_speaker_confidences = [seg['confidence'] for seg in segments if seg['user'] == main_speaker]
                avg_confidence = sum(main_speaker_confidences) / len(main_speaker_confidences) if main_speaker_confidences else 0
            else:
                main_speaker = "unknown"
                avg_confidence = 0
            
            return jsonify({
                "success": True,
                "data": {
                    "speaker_id": main_speaker,
                    "similarity": float(avg_confidence),
                    "confidence": float(avg_confidence * 100)  # 转换为百分比
                },
                "details": {
                    "segments": segments,
                    "processing_time": processing_time,
                    "mode": "diarization",
                    "detected_speakers": detected_speakers,
                    "total_speakers": total_speakers
                }
            })
            
        finally:
            # 清理临时文件
            if temp_path and temp_path.exists():
                temp_path.unlink()
            if converted_path and converted_path.exists():
                converted_path.unlink()
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "识别失败",
            "details": str(e)
        }), 500


@app.route('/api/v1/speaker/list', methods=['GET'])
def list_speakers():
    """
    获取已注册员工列表（包含音频路径）
    
    响应：JSON结果
    """
    try:
        # 获取模型文件路径
        model_path = Config.get_latest_speaker_embeddings()
        
        # 从模型文件和metadata中获取用户列表
        speakers = storage.get_all_speakers(model_path)
        
        return jsonify({
            "success": True,
            "data": {
                "total": len(speakers),
                "speakers": speakers
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "获取列表失败",
            "details": str(e)
        }), 500


@app.route('/api/v1/speaker/delete/<speaker_id>', methods=['DELETE'])
def delete_speaker(speaker_id):
    """
    删除已注册的说话人
    
    Args:
        speaker_id: 说话人ID
    
    响应：JSON结果
    """
    try:
        # 获取模型文件路径
        model_path = Config.get_latest_speaker_embeddings()
        
        if not model_path or not model_path.exists():
            return jsonify({
                "error": "模型文件不存在"
            }), 404
        
        # 从模型文件中删除用户
        import json
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        if speaker_id not in model_data:
            return jsonify({
                "error": f"用户 {speaker_id} 不存在"
            }), 404
        
        # 删除用户
        del model_data[speaker_id]
        
        # 保存模型文件
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        # 删除音频文件（如果存在）
        storage.delete_speaker(speaker_id)
        
        # 重新加载识别模型
        reload_speaker_id_wrapper()
        
        return jsonify({
            "success": True,
            "message": f"用户 {speaker_id} 已删除"
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "删除失败",
            "details": str(e)
        }), 500


@app.route('/api/v1/speaker/verify', methods=['POST'])
def verify_speaker():
    """
    验证特定员工声纹
    
    请求：FormData(speaker_id, audio)
    响应：JSON结果
    """
    try:
        # 检查参数
        if 'speaker_id' not in request.form:
            return jsonify({"error": "缺少 speaker_id 参数"}), 400
        
        if 'audio' not in request.files:
            return jsonify({"error": "缺少音频文件"}), 400
        
        speaker_id = request.form['speaker_id'].strip()
        
        # 检查说话人是否存在
        if not storage.speaker_exists(speaker_id):
            return jsonify({"error": f"说话人 {speaker_id} 不存在"}), 404
        
        # 先进行识别
        audio_file = request.files['audio']
        
        # 保存临时文件
        filename = secure_filename(f"verify_temp_{audio_file.filename}")
        temp_path = UPLOAD_DIR / filename
        audio_file.save(str(temp_path))
        
        try:
            # 读取音频
            audio_data, sr = sf.read(str(temp_path), dtype='int16')
            
            # 获取识别器
            wrapper = get_speaker_id_wrapper()
            if wrapper is None:
                return jsonify({"error": "声纹识别模型未加载"}), 400
            
            # 识别
            recognized_id, similarity = wrapper.match_multi(audio_data)
            
            # 判断是否匹配
            is_match = recognized_id == speaker_id
            
            return jsonify({
                "success": True,
                "data": {
                    "is_match": is_match,
                    "expected_speaker": speaker_id,
                    "recognized_speaker": recognized_id,
                    "similarity": float(similarity),
                    "confidence": float(similarity * 100)
                }
            })
            
        finally:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "验证失败",
            "details": str(e)
        }), 500


@app.route('/api/v1/config', methods=['GET', 'PUT'])
def config():
    """
    获取/更新配置
    
    GET: 获取当前配置
    PUT: 更新配置（JSON配置）
    """
    if request.method == 'GET':
        return jsonify({
            "success": True,
            "data": {
                "sample_rate": SAMPLE_RATE,
                "max_audio_size": MAX_AUDIO_SIZE,
                "allowed_extensions": list(ALLOWED_EXTENSIONS),
                "model_path": str(Config.get_latest_speaker_embeddings()) if Config.get_latest_speaker_embeddings() else None
            }
        })
    
    elif request.method == 'PUT':
        # 暂不支持动态修改配置
        return jsonify({"error": "暂不支持动态修改配置"}), 501


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "文件太大，请上传小于10MB的文件"}), 413


# ==================== 主函数 ====================

def main():
    """启动Flask应用"""
    # 检查SSL证书
    cert_file = Path(__file__).parent / 'cert.pem'
    key_file = Path(__file__).parent / 'key.pem'
    
    use_ssl = cert_file.exists() and key_file.exists()
    protocol = "https" if use_ssl else "http"
    
    print("=" * 60)
    print("网页版声纹识别系统")
    print("=" * 60)
    print(f"服务地址: {protocol}://{FLASK_HOST}:{FLASK_PORT}")
    if use_ssl:
        print("✅ HTTPS已启用（自签名证书）")
        print("⚠️  浏览器会显示证书警告，点击\"继续访问\"即可")
    else:
        print("⚠️  HTTPS未启用，录音功能仅在localhost下可用")
        print("💡 运行以下命令生成证书启用HTTPS：")
        print("   openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
    print(f"音频存储目录: {AUDIO_STORAGE_DIR}")
    print(f"模型文件: {Config.get_latest_speaker_embeddings()}")
    
    # 预加载模型（避免首次请求时加载导致超时）
    print("-" * 60)
    print("正在预加载声纹识别模型...")
    wrapper = get_speaker_id_wrapper()
    if wrapper:
        print("✅ 声纹识别模型加载成功")
    else:
        print("⚠️  声纹识别模型未加载（尚无注册用户）")
    print("=" * 60)
    
    if use_ssl:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG,
            ssl_context=(str(cert_file), str(key_file))
        )
    else:
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=FLASK_DEBUG
        )


if __name__ == '__main__':
    main()

