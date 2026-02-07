"""Flask Web 测试界面"""
import sys
from pathlib import Path

from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 创建Flask应用
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """主页"""
    # 读取 API 端口配置并传递给前端
    config_file = Path(__file__).parent.parent / "config" / "server_config.yaml"
    api_port = 8000  # 默认端口
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            api_config = config.get('api', {})
            api_port = api_config.get('port', 8000)
    
    return render_template('index.html', api_port=api_port)


@app.route('/api/test-audio/list', methods=['GET'])
def list_test_audio():
    """获取测试音频列表"""
    try:
        test_audio_dir = Path(__file__).parent.parent / 'data' / 'test'
        if not test_audio_dir.exists():
            return jsonify({'success': False, 'error': '测试音频目录不存在'})
        
        audio_files = []
        for file in test_audio_dir.glob('*.wav'):
            file_size = file.stat().st_size
            audio_files.append({
                'name': file.name,
                'size': f'{file_size / 1024:.1f} KB',
                'path': str(file.relative_to(test_audio_dir.parent))
            })
        
        audio_files.sort(key=lambda x: x['name'])
        return jsonify({'success': True, 'files': audio_files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/test-audio/<filename>', methods=['GET'])
def get_test_audio(filename):
    """获取测试音频文件"""
    try:
        test_audio_dir = Path(__file__).parent.parent / 'data' / 'test'
        return send_from_directory(test_audio_dir, filename, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/test-audio/<filename>', methods=['DELETE'])
def delete_test_audio(filename):
    """删除测试音频文件"""
    try:
        test_audio_dir = Path(__file__).parent.parent / 'data' / 'test'
        file_path = test_audio_dir / filename
        
        if not file_path.exists():
            return jsonify({'success': False, 'error': '文件不存在'})
        
        file_path.unlink()
        return jsonify({'success': True, 'message': f'文件 {filename} 已删除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/test-audio/upload', methods=['POST'])
def upload_test_audio():
    """上传测试音频文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'})
        
        # 检查文件扩展名
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.wav'):
            return jsonify({'success': False, 'error': '只支持WAV格式音频'})
        
        # 保存文件
        test_audio_dir = Path(__file__).parent.parent / 'data' / 'test'
        test_audio_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = test_audio_dir / filename
        file.save(file_path)
        
        return jsonify({'success': True, 'message': f'文件 {filename} 已上传', 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # 读取配置
    config_file = Path(__file__).parent.parent / "config" / "server_config.yaml"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            web_config = config.get('web', {})
            host = web_config.get('host', '0.0.0.0')
            port = web_config.get('port', 5000)
            debug = web_config.get('debug', False)
            # 读取 API 端口配置
            api_config = config.get('api', {})
            api_port = api_config.get('port', 8000)
    else:
        host = '0.0.0.0'
        port = 5000
        debug = False
        api_port = 8000
    
    # SSL 证书路径
    cert_dir = Path(__file__).parent.parent / "config" / "certs"
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    # 检查证书是否存在
    if cert_file.exists() and key_file.exists():
        ssl_context = (str(cert_file), str(key_file))
        protocol = "https"
    else:
        ssl_context = None
        protocol = "http"
    
    print("=" * 60)
    print("Remote-V2 Web测试界面")
    print("=" * 60)
    print(f"本机访问: {protocol}://localhost:{port}")
    print(f"局域网访问: {protocol}://10.8.21.33:{port}")
    print(f"API地址: http://localhost:{api_port} (从 server_config.yaml 读取)")
    print("=" * 60)
    if ssl_context:
        print("🔒 HTTPS 已启用（自签名证书）")
        print("⚠️  重要提示：")
        print(f"   - 浏览器会显示安全警告，点击'高级'→'继续访问'即可")
        print(f"   - 或在浏览器中信任该证书")
    else:
        print("⚠️  重要提示：")
        print(f"   - 当前使用 HTTP 协议（未加密）")
        print(f"   - 如需 HTTPS，请生成 SSL 证书")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_context)

