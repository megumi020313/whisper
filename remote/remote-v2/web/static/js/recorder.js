/**
 * 录音模块 - 使用 HTML5 MediaRecorder API
 * 实现浏览器录音功能，支持16kHz/16bit/单声道格式转换
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioStream = null;
        this.isRecording = false;
        this.audioBlob = null;
        this.audioUrl = null;
        
        // 录音参数
        this.sampleRate = 16000;
        this.bitDepth = 16;
        this.channels = 1;
        
        // 权限缓存 - 记住已授权的音频流
        this.permissionGranted = false;
    }
    
    /**
     * 检查浏览器是否支持MediaRecorder
     */
    static isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }
    
    /**
     * 初始化录音设备
     */
    async initialize() {
        if (!AudioRecorder.isSupported()) {
            // 检查是否是安全上下文问题
            const isSecureContext = window.isSecureContext || window.location.protocol === 'https:' || 
                                   window.location.hostname === 'localhost' || 
                                   window.location.hostname === '127.0.0.1';
            
            // 提供更详细的诊断信息
            const diagnostics = [];
            if (!navigator.mediaDevices) {
                diagnostics.push('navigator.mediaDevices 不可用');
            }
            if (!navigator.mediaDevices?.getUserMedia) {
                diagnostics.push('getUserMedia 不可用');
            }
            if (!window.MediaRecorder) {
                diagnostics.push('MediaRecorder 不可用');
            }
            
            let errorMsg = '❌ 您的浏览器不支持录音功能';
            if (diagnostics.length > 0) {
                errorMsg += '：' + diagnostics.join('、');
            }
            
            errorMsg += '\n\n💡 可能的原因：\n';
            if (!isSecureContext) {
                errorMsg += '⚠️ 当前使用 HTTP 协议访问，浏览器出于安全考虑禁止使用麦克风！\n\n';
                errorMsg += '📌 解决方案：\n';
                errorMsg += '1. 使用 HTTPS 访问（推荐）\n';
                errorMsg += '2. 或使用 localhost 访问（本地测试）\n';
                errorMsg += '3. 或切换到"📁 上传音频文件"选项卡\n';
            } else {
                errorMsg += '1. 使用Chrome、Firefox或Edge浏览器的最新版本\n';
                errorMsg += '2. 确保使用 https:// 或 localhost 访问\n';
                errorMsg += '3. 或者切换到"📁 上传音频文件"选项卡\n';
            }
            
            errorMsg += '\n🔍 点击下方链接进行详细诊断：/test-recorder';
            
            throw new Error(errorMsg);
        }
        
        try {
            // 请求麦克风权限
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: this.channels,
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            console.log('✅ 麦克风初始化成功');
            return true;
        } catch (error) {
            console.error('麦克风初始化失败:', error);
            
            // 检查是否是安全上下文问题
            const isSecureContext = window.isSecureContext || window.location.protocol === 'https:' || 
                                   window.location.hostname === 'localhost' || 
                                   window.location.hostname === '127.0.0.1';
            
            let errorMsg = '';
            let needsDiagnostic = false;
            
            if (error.name === 'NotAllowedError') {
                if (!isSecureContext) {
                    errorMsg = '❌ 麦克风权限被拒绝\n\n';
                    errorMsg += '⚠️ 当前使用 HTTP 协议访问，浏览器出于安全考虑禁止使用麦克风！\n\n';
                    errorMsg += '📌 解决方案：\n';
                    errorMsg += '1. 使用 HTTPS 访问（推荐）\n';
                    errorMsg += '2. 或使用 localhost 访问（本地测试）\n';
                    errorMsg += '3. 或切换到"📁 上传音频文件"选项卡';
                    needsDiagnostic = true;
                } else {
                    errorMsg = '❌ 麦克风权限被拒绝\n\n';
                    errorMsg += '请在浏览器设置中允许访问麦克风，或切换到"📁 上传音频文件"选项卡';
                    needsDiagnostic = true;
                }
            } else if (error.name === 'NotFoundError') {
                errorMsg = '❌ 未检测到麦克风设备\n\n请连接麦克风设备，或切换到"📁 上传音频文件"选项卡';
                needsDiagnostic = true;
            } else if (error.name === 'NotReadableError') {
                errorMsg = '❌ 麦克风正在被其他应用使用\n\n请关闭其他使用麦克风的应用，或切换到"📁 上传音频文件"选项卡';
                needsDiagnostic = true;
            } else if (error.name === 'OverconstrainedError') {
                errorMsg = '❌ 麦克风不支持请求的音频格式\n\n请切换到"📁 上传音频文件"选项卡';
                needsDiagnostic = true;
            } else {
                errorMsg = '❌ 初始化录音设备失败: ' + error.message + '\n\n💡 建议切换到"📁 上传音频文件"选项卡';
                needsDiagnostic = true;
            }
            
            // 添加诊断链接
            if (needsDiagnostic) {
                errorMsg += '\n\n🔍 点击下方链接进行详细诊断：/test-recorder';
            }
            
            throw new Error(errorMsg);
        }
    }
    
    /**
     * 开始录音
     */
    async startRecording() {
        if (this.isRecording) {
            throw new Error('正在录音中');
        }
        
        if (!this.audioStream) {
            await this.initialize();
        }
        
        this.audioChunks = [];
        
        // 创建MediaRecorder实例
        try {
            // 优先尝试WAV格式，否则使用WebM
            let options = {};
            
            // 检查支持的格式（按优先级）
            if (MediaRecorder.isTypeSupported('audio/wav')) {
                options.mimeType = 'audio/wav';
            } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=pcm')) {
                options.mimeType = 'audio/webm;codecs=pcm';
            } else if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                options.mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                options.mimeType = 'audio/ogg;codecs=opus';
            } else {
                // 使用默认格式
                options.mimeType = 'audio/webm';
            }
            
            console.log('使用录音格式:', options.mimeType);
            this.mediaRecorder = new MediaRecorder(this.audioStream, options);
            
            // 监听数据可用事件
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // 监听停止事件
            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
            };
            
            // 开始录音
            this.mediaRecorder.start();
            this.isRecording = true;
            
            return true;
        } catch (error) {
            throw new Error('启动录音失败: ' + error.message);
        }
    }
    
    /**
     * 停止录音
     */
    stopRecording() {
        return new Promise((resolve, reject) => {
            if (!this.isRecording || !this.mediaRecorder) {
                reject(new Error('未在录音中'));
                return;
            }
            
            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
                this.audioUrl = URL.createObjectURL(this.audioBlob);
                this.isRecording = false;
                resolve({
                    blob: this.audioBlob,
                    url: this.audioUrl
                });
            };
            
            this.mediaRecorder.stop();
        });
    }
    
    /**
     * 获取录音的Blob对象
     */
    getAudioBlob() {
        return this.audioBlob;
    }
    
    /**
     * 获取录音的URL（用于播放）
     */
    getAudioUrl() {
        return this.audioUrl;
    }
    
    /**
     * 获取录音时长（秒）
     */
    async getAudioDuration() {
        if (!this.audioUrl) {
            return 0;
        }
        
        return new Promise((resolve) => {
            const audio = new Audio(this.audioUrl);
            audio.addEventListener('loadedmetadata', () => {
                resolve(audio.duration);
            });
        });
    }
    
    /**
     * 转换录音为WAV格式（16kHz/16bit/单声道）
     * 注：实际转换需要在后端进行，前端只负责上传
     */
    async convertToWav() {
        // 前端录制的音频会在后端进行格式转换
        // 这里直接返回录音的Blob
        return this.audioBlob;
    }
    
    /**
     * 清理资源
     * @param {boolean} keepStream - 是否保持音频流（默认false，完全清理；true则保留音频流避免重新授权）
     */
    cleanup(keepStream = true) {
        if (this.audioUrl) {
            URL.revokeObjectURL(this.audioUrl);
            this.audioUrl = null;
        }
        
        // 只在需要时清理音频流（避免重新授权）
        if (!keepStream && this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
            this.permissionGranted = false;
            console.log('音频流已完全清理');
        }
        
        this.audioChunks = [];
        this.audioBlob = null;
        this.mediaRecorder = null;
        this.isRecording = false;
    }
    
    /**
     * 检查录音状态
     */
    getStatus() {
        return {
            isRecording: this.isRecording,
            hasAudio: this.audioBlob !== null,
            duration: this.audioBlob ? 'unknown' : 0
        };
    }
}

// 导出为全局变量
window.AudioRecorder = AudioRecorder;

