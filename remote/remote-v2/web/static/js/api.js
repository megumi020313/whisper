/**
 * API调用模块 - 使用 Fetch API
 * 处理前后端数据交互
 */

class VoiceprintAPI {
    constructor(baseUrl = '') {
        // 自动检测主机名和协议，支持局域网访问和HTTPS
        if (!baseUrl) {
            const host = window.location.hostname;
            const protocol = window.location.protocol; // 'http:' 或 'https:'
            // 从后端配置读取 API 端口（优先），否则使用默认端口 8000
            const apiPort = window.API_PORT || 8000;
            // 使用与当前页面相同的协议
            this.baseUrl = `${protocol}//${host}:${apiPort}`;
            console.log(`🌐 API地址自动配置为: ${this.baseUrl} (端口: ${apiPort})`);
        } else {
            this.baseUrl = baseUrl;
        }
    }
    
    /**
     * 健康检查
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('健康检查失败:', error);
            throw new Error('无法连接到服务器: ' + error.message);
        }
    }
    
    /**
     * 注册声纹
     * @param {string} speakerId - 用户名（说话人ID）
     * @param {Blob} audioBlob - 录音数据
     */
    async registerSpeaker(speakerId, audioBlob) {
        try {
            // 创建FormData
            const formData = new FormData();
            formData.append('user_id', speakerId);  // 修复: 使用 user_id 匹配后端
            // 保持原始文件名和扩展名
            const fileName = audioBlob.name || `${speakerId}.wav`;
            formData.append('audio', audioBlob, fileName);
            
            // 发送请求（大文件上传需要更长的超时时间，600秒）
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 600000); // 10分钟超时
            
            try {
                const response = await fetch(`${this.baseUrl}/api/v1/speaker/register`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
            
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `HTTP ${response.status}`);
                }
                
                return result;
            } catch (error) {
                clearTimeout(timeoutId);
                if (error.name === 'AbortError') {
                    throw new Error('上传超时，请检查网络连接或使用较小的文件');
                }
                throw error;
            }
        } catch (error) {
            console.error('注册失败:', error);
            throw new Error('声纹注册失败: ' + error.message);
        }
    }
    
    /**
     * 识别声纹
     * @param {Blob} audioBlob - 录音数据
     */
    async recognizeSpeaker(audioBlob) {
        try {
            // 创建FormData
            const formData = new FormData();
            // 保持原始文件名和扩展名
            const fileName = audioBlob.name || 'recognize.wav';
            formData.append('audio', audioBlob, fileName);
            
            // 发送请求（大文件上传需要更长的超时时间，600秒）
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 600000); // 10分钟超时
            
            try {
                const response = await fetch(`${this.baseUrl}/api/v1/speaker/recognize`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
            
                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error || `HTTP ${response.status}`);
                }
                
                return result;
            } catch (error) {
                clearTimeout(timeoutId);
                if (error.name === 'AbortError') {
                    throw new Error('上传超时，请检查网络连接或使用较小的文件');
                }
                throw error;
            }
        } catch (error) {
            console.error('识别失败:', error);
            throw new Error('声纹识别失败: ' + error.message);
        }
    }

    /**
     * 识别声纹 (SMP模型 - 通过Proxy)
     * @param {Blob} audioBlob - 录音数据
     */
    async recognizeSpeakerSMP(audioBlob) {
        try {
            // 创建FormData
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recognize_smp.wav');
            
            // 发送请求到 Proxy 接口
            const response = await fetch(`${this.baseUrl}/api/v1/smp/recognize`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                // 处理 502 Bad Gateway (SMP未启动)
                   if (response.status === 502) {
                       throw new Error("SMP服务未启动 (Port 51003)");
                   }
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            console.error('SMP识别失败:', error);
            throw new Error(error.message);
        }
    }
    
    /**
     * 验证特定说话人
     * @param {string} speakerId - 说话人ID
     * @param {Blob} audioBlob - 录音数据
     */
    async verifySpeaker(speakerId, audioBlob) {
        try {
            // 创建FormData
            const formData = new FormData();
            formData.append('user_id', speakerId);  // 修复: 使用 user_id
            formData.append('audio', audioBlob, 'verify.wav');
            
            // 发送请求
            const response = await fetch(`${this.baseUrl}/api/v1/speaker/verify`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            console.error('验证失败:', error);
            throw new Error('声纹验证失败: ' + error.message);
        }
    }
    
    /**
     * 获取已注册说话人列表
     */
    async getSpeakerList() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/speaker/users`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            console.error('获取列表失败:', error);
            throw new Error('获取说话人列表失败: ' + error.message);
        }
    }
    
    /**
     * 删除已注册说话人
     */
    async deleteSpeaker(speakerId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/speaker/users/${speakerId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            console.error('删除用户失败:', error);
            throw new Error('删除用户失败: ' + error.message);
        }
    }
    
    /**
     * 获取系统配置
     */
    async getConfig() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/config`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            return result;
        } catch (error) {
            console.error('获取配置失败:', error);
            throw new Error('获取系统配置失败: ' + error.message);
        }
    }
    
    /**
     * 获取模型配置文件内容
     */
    async getModelConfig() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/config`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '获取配置失败');
            }
            
            return await response.json();
        } catch (error) {
            console.error('获取模型配置失败:', error);
            throw error;
        }
    }
    
    /**
     * 获取文件上传配置（文件大小限制和允许的格式）
     */
    async getFileSettings() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/config/file-settings`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (!response.ok || !result.success) {
                throw new Error(result.error || '获取文件配置失败');
            }
            
            return result.data;
        } catch (error) {
            console.error('获取文件配置失败:', error);
            // 返回默认值
            return {
                max_audio_size: 524288000,  // 500MB
                max_audio_size_mb: 500,
                allowed_extensions: ['.wav', '.mp3', '.mp4', '.m4a', '.webm', '.ogg', '.flac']
            };
        }
    }
    
    /**
     * 更新模型配置文件
     * @param {string} content - 配置文件内容
     * @param {boolean} restart - 是否重启服务
     */
    async updateModelConfig(content, restart = false) {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: content,
                    restart: restart
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || '更新配置失败');
            }
            
            return await response.json();
        } catch (error) {
            console.error('更新模型配置失败:', error);
            throw error;
        }
    }
    
}

// 导出为全局变量
window.VoiceprintAPI = VoiceprintAPI;

