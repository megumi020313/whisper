/**
 * 主逻辑模块
 * 协调各模块功能，处理用户交互
 */

class VoiceprintApp {
    constructor() {
        this.recorder = null;
        this.api = null;
        this.ui = null;
        
        this.recordingStartTime = 0;
        this.recordingTimer = null;
        
        // ✨ 推荐录音时长（提高识别准确率）
        this.recommendedDuration = 15; // 推荐至少15秒
        this.maxRecommendedDuration = 30; // 推荐最多30秒
        
        // 文件上传相关
        this.uploadedFile = null;
        this.uploadedAudioBlob = null;
        this.lastFileLoadTime = 0; // 上次文件加载时间
        
        // 文件配置（从后端加载）
        this.fileSettings = {
            max_audio_size: 524288000,  // 默认500MB
            max_audio_size_mb: 500,
            allowed_extensions: ['.wav', '.mp3', '.mp4', '.m4a', '.webm', '.ogg', '.flac']
        };
        
        // 当前激活的标签页
        this.activeTab = 'record'; // 'record' 或 'upload'
        
        // 每个标签页独立的识别结果
        this.recordTabResults = {
            v2: null,
            smp: null
        };
        this.uploadTabResults = {
            v2: null,
            smp: null
        };
    }
    
    /**
     * 初始化应用
     */
    async init() {
        console.log('初始化声纹识别系统...');
        
        // 初始化各模块
        this.recorder = new AudioRecorder();
        this.api = new VoiceprintAPI();
        this.ui = new UIController();
        
        // 加载文件配置
        try {
            this.fileSettings = await this.api.getFileSettings();
            console.log('✅ 文件配置已加载:', this.fileSettings);
        } catch (error) {
            console.warn('⚠️ 加载文件配置失败，使用默认值:', error);
        }
        
        // 初始化UI元素
        this.ui.initElements();
        
        // 绑定事件
        this.bindEvents();
        
        // 检查浏览器支持
        if (!AudioRecorder.isSupported()) {
            this.ui.showError('您的浏览器不支持录音功能，请使用Chrome、Firefox或Edge浏览器');
            return;
        }
        
        // 健康检查
        try {
            const health = await this.api.healthCheck();
            console.log('服务器健康检查:', health);
            
            // 更新版本号显示
            if (health.version) {
                const versionEl = document.getElementById('system-version');
                if (versionEl) {
                    versionEl.textContent = `v${health.version}`;
                }
            }
            
            if (!health.model_loaded) {
                this.ui.updateRecordStatus('warning', '提示：系统中暂无注册用户，请先注册');
            } else {
                this.ui.updateRecordStatus('ready', '就绪 - 点击"开始录音"');
            }
            
            // 初始化时预加载用户列表
            this.handleListSpeakers();
        } catch (error) {
            this.ui.showError('无法连接到服务器，请确认服务器已启动');
            console.error('健康检查失败:', error);
        }
    }
    
    /**
     * 绑定事件
     */
    bindEvents() {
        // 选项卡切换
        document.getElementById('tab-record')?.addEventListener('click', () => this.switchTab('record'));
        document.getElementById('tab-upload')?.addEventListener('click', () => this.switchTab('upload'));
        
        // 录音控制
        document.getElementById('start-record')?.addEventListener('click', () => this.handleStartRecord());
        document.getElementById('stop-record')?.addEventListener('click', () => this.handleStopRecord());
        
        // 文件上传
        const audioFileInput = document.getElementById('audio-file-input');
        if (audioFileInput) {
            audioFileInput.addEventListener('change', (e) => this.handleFileSelected(e));
        }
        document.getElementById('clear-upload-file-btn')?.addEventListener('click', () => this.handleClearFile());
        document.getElementById('reupload-file-btn')?.addEventListener('click', () => this.handleSelectFile());
        document.getElementById('select-upload-btn')?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.handleSelectFile();
        });
        
        // 拖拽上传
        const dropZone = document.getElementById('upload-drop-zone');
        if (dropZone) {
            dropZone.addEventListener('click', (e) => {
                // 只有点击区域本身时才触发，不包括按钮
                if (e.target === dropZone || e.target.closest('#upload-drop-zone') && !e.target.closest('#select-upload-btn')) {
                    this.handleSelectFile();
                }
            });
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('border-blue-500', 'bg-blue-50');
            });
            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
            });
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('border-blue-500', 'bg-blue-50');
                
                // 检查是否是测试音频拖拽
                const testAudioFilename = e.dataTransfer.getData('test-audio-filename');
                if (testAudioFilename) {
                    // 从测试音频加载
                    this.loadTestAudioFile(testAudioFilename);
                    return;
                }
                
                // 否则是普通文件拖拽
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = document.getElementById('audio-file-input');
                    if (fileInput) {
                        // 创建新的FileList并触发change事件
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(files[0]);
                        fileInput.files = dataTransfer.files;
                        fileInput.dispatchEvent(new Event('change'));
                    }
                }
            });
        }
        
        // 测试音频
        document.getElementById('toggle-test-audio-header')?.addEventListener('click', () => this.toggleTestAudio());
        document.getElementById('refresh-test-audio-btn')?.addEventListener('click', () => this.loadTestAudioList());
        document.getElementById('upload-test-audio-btn')?.addEventListener('click', () => {
            document.getElementById('test-audio-upload-input')?.click();
        });
        document.getElementById('test-audio-upload-input')?.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                this.uploadTestAudio(e.target.files[0]);
            }
        });
        document.getElementById('test-audio-close-btn')?.addEventListener('click', () => this.closeTestAudioPlayer());
        
        // 功能按钮（顶部统一识别按钮）
        document.getElementById('recognize-btn-top')?.addEventListener('click', () => this.handleRecognize());
        
        // 单独识别按钮
        document.getElementById('recognize-v2-btn')?.addEventListener('click', () => this.handleRecognizeV2());
        document.getElementById('recognize-smp-btn')?.addEventListener('click', () => this.handleRecognizeSMP());
        
        // 注册页面按钮
        document.getElementById('start-record-register-btn')?.addEventListener('click', () => this.startRecordRegister());
        document.getElementById('start-upload-register-btn')?.addEventListener('click', () => this.startUploadRegister());
        document.getElementById('register-record-btn')?.addEventListener('click', () => this.handleRegisterRecord());
        document.getElementById('register-stop-btn')?.addEventListener('click', () => this.handleRegisterStop());
        document.getElementById('cancel-register-btn')?.addEventListener('click', () => this.cancelRegister());
        document.getElementById('upload-save-btn')?.addEventListener('click', () => this.saveUploadRegister());
        document.getElementById('upload-reselect-btn')?.addEventListener('click', () => this.reselectUploadFile());
        document.getElementById('cancel-upload-btn')?.addEventListener('click', () => this.cancelRegister());
        
        // 时间轴播放控制
        document.getElementById('timeline-play-btn')?.addEventListener('click', () => this.playTimeline());
        document.getElementById('timeline-pause-btn')?.addEventListener('click', () => this.pauseTimeline());
        
        // 音频播放事件监听
        const timelineAudio = document.getElementById('timeline-audio');
        if (timelineAudio) {
            timelineAudio.addEventListener('timeupdate', () => this.updateTimelineProgress());
            timelineAudio.addEventListener('ended', () => this.onTimelineEnded());
        }
        
        // SMP时间轴播放控制
        const smpTimelineAudio = document.getElementById('smp-timeline-audio');
        if (smpTimelineAudio) {
            smpTimelineAudio.addEventListener('timeupdate', () => this.updateSMPTimelineProgress());
            smpTimelineAudio.addEventListener('ended', () => this.onSMPTimelineEnded());
        }
        
        // 管理页面功能
        document.getElementById('refresh-speakers-btn')?.addEventListener('click', () => this.handleListSpeakers());
        const searchInput = document.getElementById('search-speaker-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.handleSearchSpeaker(e.target.value));
        }
        
        // 配置管理
        document.getElementById('config-btn')?.addEventListener('click', () => this.openConfigModal());
        document.getElementById('config-close-btn')?.addEventListener('click', () => this.closeConfigModal());
        document.getElementById('config-restart-btn')?.addEventListener('click', () => this.saveAndRestartConfig());
        
        // 点击模态框外部关闭
        document.getElementById('config-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'config-modal') {
                this.closeConfigModal();
            }
        });
    }
    
    /**
     * 切换选项卡
     */
    switchTab(tab) {
        // 更新 activeTab 状态
        this.activeTab = tab;
        
        const recordTab = document.getElementById('tab-record');
        const uploadTab = document.getElementById('tab-upload');
        const recordContent = document.getElementById('record-tab-content');
        const uploadContent = document.getElementById('upload-tab-content');
        
        if (tab === 'record') {
            // 更新按钮样式
            recordTab?.classList.add('active', 'bg-blue-600', 'text-white', 'shadow-lg');
            recordTab?.classList.remove('bg-white', 'text-gray-700', 'border', 'border-gray-300');
            uploadTab?.classList.remove('active', 'bg-blue-600', 'text-white', 'shadow-lg');
            uploadTab?.classList.add('bg-white', 'text-gray-700', 'border', 'border-gray-300');
            
            // 切换内容
            recordContent?.classList.add('active');
            uploadContent?.classList.remove('active');
            uploadContent.style.display = 'none';
            recordContent.style.display = 'block';
            
            // 恢复录音标签页的识别结果
            this.restoreTabResults('record');
            
            // 检查录音状态，更新识别按钮
            this.updateRecognizeButtonState();
        } else {
            // 更新按钮样式
            uploadTab?.classList.add('active', 'bg-blue-600', 'text-white', 'shadow-lg');
            uploadTab?.classList.remove('bg-white', 'text-gray-700', 'border', 'border-gray-300');
            recordTab?.classList.remove('active', 'bg-blue-600', 'text-white', 'shadow-lg');
            recordTab?.classList.add('bg-white', 'text-gray-700', 'border', 'border-gray-300');
            
            // 切换内容
            recordContent?.classList.remove('active');
            uploadContent?.classList.add('active');
            recordContent.style.display = 'none';
            uploadContent.style.display = 'block';
            
            // 恢复上传标签页的识别结果
            this.restoreTabResults('upload');
            
            // 检查上传状态，更新识别按钮
            this.updateRecognizeButtonState();
        }
    }
    
    /**
     * 显示上传音频预览
     */
    showUploadAudioPreview(audioUrl, duration) {
        const audioPlayer = document.getElementById('upload-audio-player');
        const durationEl = document.getElementById('upload-audio-duration');
        
        if (audioPlayer && audioUrl) {
            audioPlayer.src = audioUrl;
        }
        
        if (durationEl && duration) {
            durationEl.textContent = `时长: ${duration.toFixed(1)}秒`;
        }
    }
    
    /**
     * 显示录音预览
     */
    showRecordAudioPreview(audioUrl, duration) {
        const previewDiv = document.getElementById('record-audio-preview');
        const audioPlayer = document.getElementById('record-audio-player');
        const durationEl = document.getElementById('record-audio-duration');
        
        if (previewDiv) {
            previewDiv.style.display = 'block';
        }
        
        if (audioPlayer && audioUrl) {
            audioPlayer.src = audioUrl;
        }
        
        if (durationEl && duration) {
            durationEl.textContent = `时长: ${duration.toFixed(1)}秒`;
        }
    }
    
    /**
     * 隐藏录音预览
     */
    hideRecordAudioPreview() {
        const previewDiv = document.getElementById('record-audio-preview');
        const audioPlayer = document.getElementById('record-audio-player');
        
        if (previewDiv) {
            previewDiv.style.display = 'none';
        }
        
        if (audioPlayer) {
            audioPlayer.src = '';
        }
    }
    
    /**
     * 清除上传的文件
     */
    clearUploadedFile() {
        // 清除上传文件相关状态
        this.uploadedFile = null;
        this.uploadedAudioBlob = null;
        
        // 重置上传区域显示
        const emptyState = document.getElementById('upload-empty-state');
        const fileState = document.getElementById('upload-file-state');
        const audioPlayer = document.getElementById('upload-audio-player');
        const fileInput = document.getElementById('audio-file-input');
        
        if (emptyState) emptyState.style.display = 'block';
        if (fileState) fileState.style.display = 'none';
        if (audioPlayer) audioPlayer.src = '';
        if (fileInput) fileInput.value = '';
    }
    
    /**
     * 清除当前标签页的识别结果
     */
    clearCurrentTabResults() {
        const results = this.activeTab === 'record' ? this.recordTabResults : this.uploadTabResults;
        
        // 清空保存的结果
        results.v2Container = '';
        results.v2Segments = '';
        results.v2Timeline = '';
        results.smpContainer = '';
        results.smpSegments = '';
        results.smpTimeline = '';
        results.v2CardDisplay = 'none';
        results.v2TimelineCardDisplay = 'none';
        results.smpCardDisplay = 'none';
        results.smpTimelineCardDisplay = 'none';
        
        // 清空DOM显示
        const v2Container = document.getElementById('speakers-list-container');
        const v2Segments = document.getElementById('segments-list');
        const v2Timeline = document.getElementById('timeline-content');
        const v2Card = document.getElementById('result-segments');
        const v2TimelineCard = document.getElementById('timeline-result');
        
        if (v2Container) v2Container.innerHTML = '';
        if (v2Segments) v2Segments.innerHTML = '';
        if (v2Timeline) v2Timeline.innerHTML = '';
        if (v2Card) v2Card.style.display = 'none';
        if (v2TimelineCard) v2TimelineCard.style.display = 'none';
        
        const smpContainer = document.getElementById('smp-speakers-list-container');
        const smpSegments = document.getElementById('smp-segments-list');
        const smpTimeline = document.getElementById('smp-timeline-content');
        const smpCard = document.getElementById('smp-result-segments');
        const smpTimelineCard = document.getElementById('smp-timeline-result');
        
        if (smpContainer) smpContainer.innerHTML = '';
        if (smpSegments) smpSegments.innerHTML = '';
        if (smpTimeline) smpTimeline.innerHTML = '';
        if (smpCard) smpCard.style.display = 'none';
        if (smpTimelineCard) smpTimelineCard.style.display = 'none';
    }
    
    /**
     * 保存当前标签页的识别结果
     */
    saveCurrentTabResults() {
        const results = this.activeTab === 'record' ? this.recordTabResults : this.uploadTabResults;
        
        // 保存V2结果
        const v2Container = document.getElementById('speakers-list-container');
        const v2Segments = document.getElementById('segments-list');
        const v2Timeline = document.getElementById('timeline-content');
        
        if (v2Container) results.v2Container = v2Container.innerHTML;
        if (v2Segments) results.v2Segments = v2Segments.innerHTML;
        if (v2Timeline) results.v2Timeline = v2Timeline.innerHTML;
        
        // 保存SMP结果
        const smpContainer = document.getElementById('smp-speakers-list-container');
        const smpSegments = document.getElementById('smp-segments-list');
        const smpTimeline = document.getElementById('smp-timeline-content');
        
        if (smpContainer) results.smpContainer = smpContainer.innerHTML;
        if (smpSegments) results.smpSegments = smpSegments.innerHTML;
        if (smpTimeline) results.smpTimeline = smpTimeline.innerHTML;
        
        // 保存显示状态
        const v2Card = document.getElementById('result-segments');
        const v2TimelineCard = document.getElementById('timeline-result');
        const smpCard = document.getElementById('smp-result-segments');
        const smpTimelineCard = document.getElementById('smp-timeline-result');
        
        results.v2CardDisplay = v2Card ? v2Card.style.display : 'none';
        results.v2TimelineCardDisplay = v2TimelineCard ? v2TimelineCard.style.display : 'none';
        results.smpCardDisplay = smpCard ? smpCard.style.display : 'none';
        results.smpTimelineCardDisplay = smpTimelineCard ? smpTimelineCard.style.display : 'none';
    }
    
    /**
     * 恢复指定标签页的识别结果
     */
    restoreTabResults(tab) {
        const results = tab === 'record' ? this.recordTabResults : this.uploadTabResults;
        
        // 恢复V2结果
        const v2Container = document.getElementById('speakers-list-container');
        const v2Segments = document.getElementById('segments-list');
        const v2Timeline = document.getElementById('timeline-content');
        const v2Card = document.getElementById('result-segments');
        const v2TimelineCard = document.getElementById('timeline-result');
        
        if (v2Container && results.v2Container !== undefined) {
            v2Container.innerHTML = results.v2Container || '';
        }
        if (v2Segments && results.v2Segments !== undefined) {
            v2Segments.innerHTML = results.v2Segments || '';
        }
        if (v2Timeline && results.v2Timeline !== undefined) {
            v2Timeline.innerHTML = results.v2Timeline || '';
        }
        if (v2Card && results.v2CardDisplay !== undefined) {
            v2Card.style.display = results.v2CardDisplay || 'none';
        }
        if (v2TimelineCard && results.v2TimelineCardDisplay !== undefined) {
            v2TimelineCard.style.display = results.v2TimelineCardDisplay || 'none';
        }
        
        // 恢复SMP结果
        const smpContainer = document.getElementById('smp-speakers-list-container');
        const smpSegments = document.getElementById('smp-segments-list');
        const smpTimeline = document.getElementById('smp-timeline-content');
        const smpCard = document.getElementById('smp-result-segments');
        const smpTimelineCard = document.getElementById('smp-timeline-result');
        
        if (smpContainer && results.smpContainer !== undefined) {
            smpContainer.innerHTML = results.smpContainer || '';
        }
        if (smpSegments && results.smpSegments !== undefined) {
            smpSegments.innerHTML = results.smpSegments || '';
        }
        if (smpTimeline && results.smpTimeline !== undefined) {
            smpTimeline.innerHTML = results.smpTimeline || '';
        }
        if (smpCard && results.smpCardDisplay !== undefined) {
            smpCard.style.display = results.smpCardDisplay || 'none';
        }
        if (smpTimelineCard && results.smpTimelineCardDisplay !== undefined) {
            smpTimelineCard.style.display = results.smpTimelineCardDisplay || 'none';
        }
        
        // 如果没有保存的结果，隐藏所有结果区域
        if (results.v2Container === undefined) {
            if (v2Container) v2Container.innerHTML = '';
            if (v2Segments) v2Segments.innerHTML = '';
            if (v2Timeline) v2Timeline.innerHTML = '';
            if (v2Card) v2Card.style.display = 'none';
            if (v2TimelineCard) v2TimelineCard.style.display = 'none';
        }
        
        if (results.smpContainer === undefined) {
            if (smpContainer) smpContainer.innerHTML = '';
            if (smpSegments) smpSegments.innerHTML = '';
            if (smpTimeline) smpTimeline.innerHTML = '';
            if (smpCard) smpCard.style.display = 'none';
            if (smpTimelineCard) smpTimelineCard.style.display = 'none';
        }
    }
    
    /**
     * 更新识别按钮状态
     */
    updateRecognizeButtonState() {
        const hasAudio = this.hasAvailableAudio();
        
        // 更新所有识别按钮
        const buttons = [
            'recognize-btn-top',
            'recognize-v2-btn',
            'recognize-smp-btn'
        ];
        
        buttons.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) {
                if (hasAudio) {
                    btn.disabled = false;
                    btn.classList.remove('opacity-50', 'cursor-not-allowed');
                } else {
                    btn.disabled = true;
                    btn.classList.add('opacity-50', 'cursor-not-allowed');
                }
            }
        });
    }
    
    /**
     * 检查是否有可用的音频
     */
    hasAvailableAudio() {
        if (this.activeTab === 'upload') {
            return !!this.uploadedAudioBlob;
        } else {
            return !!this.recorder.getAudioBlob();
        }
    }
    
    /**
     * 处理选择文件
     */
    handleSelectFile() {
        const fileInput = document.getElementById('audio-file-input');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    /**
     * 处理文件选择
     */
    async handleFileSelected(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const now = Date.now();
        if (now - this.lastFileLoadTime < 100) return;
        this.lastFileLoadTime = now;
        
        try {
            const fileName = file.name.toLowerCase();
            // 从配置获取允许的扩展名
            const audioExtensions = this.fileSettings.allowed_extensions.map(ext => ext.toLowerCase());
            const isAudioFile = audioExtensions.some(ext => fileName.endsWith(ext)) || file.type.startsWith('audio/') || file.type.startsWith('video/');
            
            if (!isAudioFile) {
                const allowedStr = this.fileSettings.allowed_extensions.join(', ');
                this.ui.showError(`请选择音频文件，支持的格式: ${allowedStr}`);
                event.target.value = '';
                return;
            }
            
            // 从配置获取文件大小限制
            if (file.size > this.fileSettings.max_audio_size) {
                this.ui.showError(`文件大小超过${this.fileSettings.max_audio_size_mb}MB限制`);
                event.target.value = '';
                return;
            }
            
            this.uploadedFile = file;
            this.uploadedAudioBlob = file;
            this.showFileInfo(file);
            this.ui.setFunctionButtonsEnabled(true);
            
            // 更新识别按钮状态
            this.updateRecognizeButtonState();
            
            const audioUrl = URL.createObjectURL(file);
            const audio = new Audio(audioUrl);
            
            audio.addEventListener('loadedmetadata', () => {
                this.showUploadAudioPreview(audioUrl, audio.duration);
            });
            
            audio.addEventListener('error', () => {
                this.showUploadAudioPreview(null, 0);
            });
            
            // 延迟清空input.value，确保下次选择同一文件也能触发change事件
            setTimeout(() => {
                event.target.value = '';
            }, 100);
            
        } catch (error) {
            this.ui.showError('文件加载失败: ' + error.message);
            event.target.value = '';
        }
    }
    
    showFileInfo(file) {
        const emptyState = document.getElementById('upload-empty-state');
        const fileState = document.getElementById('upload-file-state');
        const fileNameSpan = document.getElementById('upload-file-name');
        const fileSizeSpan = document.getElementById('upload-file-size');
        
        if (emptyState && fileState && fileNameSpan && fileSizeSpan) {
            fileNameSpan.textContent = file.name;
            fileSizeSpan.textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
            emptyState.style.display = 'none';
            fileState.style.display = 'block';
        }
    }
    
    handleClearFile() {
        this.uploadedFile = null;
        this.uploadedAudioBlob = null;
        
        const fileInput = document.getElementById('audio-file-input');
        const emptyState = document.getElementById('upload-empty-state');
        const fileState = document.getElementById('upload-file-state');
        const audioPlayer = document.getElementById('upload-audio-player');
        
        if (fileInput) fileInput.value = '';
        if (emptyState) emptyState.style.display = 'block';
        if (fileState) fileState.style.display = 'none';
        if (audioPlayer) audioPlayer.src = '';
        
        this.ui.setFunctionButtonsEnabled(false);
        this.updateRecognizeButtonState();
    }
    
    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.classList.add('drag-over');
    }
    
    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.classList.remove('drag-over');
    }
    
    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.classList.remove('drag-over');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelected({ target: { files: [files[0]] } });
        }
    }
    
    async handleStartRecord() {
        try {
            this.ui.hideResult();
            this.hideRecordAudioPreview();
            
            await this.recorder.startRecording();
            
            this.ui.updateRecordStatus('recording', '正在录音...');
            this.ui.toggleButtons(true);
            
            this.recordingStartTime = Date.now();
            this.recordingTimer = setInterval(() => {
                const elapsed = (Date.now() - this.recordingStartTime) / 1000;
                this.ui.updateRecordTime(elapsed);
            }, 100);
            
        } catch (error) {
            this.ui.showError(error.message);
            this.ui.updateRecordStatus('error', '录音失败');
        }
    }
    
    async handleStopRecord() {
        try {
            if (this.recordingTimer) {
                clearInterval(this.recordingTimer);
                this.recordingTimer = null;
            }
            
            const result = await this.recorder.stopRecording();
            
            this.ui.toggleButtons(false);
            this.ui.updateRecordStatus('ready', '录音完成');
            
            const duration = await this.recorder.getAudioDuration();
            this.showRecordAudioPreview(result.url, duration);
            this.ui.setFunctionButtonsEnabled(true);
            
            // 更新识别按钮状态
            this.updateRecognizeButtonState();
            
            if (duration < this.recommendedDuration) {
                this.ui.showResult('warning', '提示', 
                    `录音时长较短(${duration.toFixed(1)}秒)，建议录制${this.recommendedDuration}-${this.maxRecommendedDuration}秒以提高识别准确率`);
            }
            
        } catch (error) {
            this.ui.showError(error.message);
            this.ui.updateRecordStatus('error', '停止录音失败');
        }
    }
    
    async handleRegister() {
        try {
            if (!this.ui.validateSpeakerId()) return;
            
            const speakerId = this.ui.getSpeakerId();
            let audioBlob = null;
            
            if (this.activeTab === 'upload') {
                audioBlob = this.uploadedAudioBlob || this.recorder.getAudioBlob();
            } else {
                audioBlob = this.recorder.getAudioBlob() || this.uploadedAudioBlob;
            }
            
            if (!audioBlob) {
                this.ui.showError('请先录音或上传音频文件');
                return;
            }
            
            this.ui.showLoading(`正在注册用户 "${speakerId}"...`);
            this.ui.setFunctionButtonsEnabled(false);
            
            const result = await this.api.registerSpeaker(speakerId, audioBlob);
            this.ui.showSuccess(`用户 "${speakerId}" 注册成功！`);
            
            this.recorder.cleanup(true);
            this.handleClearFile();
            this.hideRecordAudioPreview();
            this.ui.updateRecordTime(0);
            
        } catch (error) {
            this.ui.showError(error.message);
        } finally {
            this.ui.setFunctionButtonsEnabled(true);
        }
    }
    
    async handleRecognize() {
        try {
            let audioBlob = null;
            
            if (this.activeTab === 'upload') {
                audioBlob = this.uploadedAudioBlob || this.recorder.getAudioBlob();
            } else {
                audioBlob = this.recorder.getAudioBlob() || this.uploadedAudioBlob;
            }
            
            if (!audioBlob) {
                this.ui.showError('请先录音或上传音频文件');
                return;
            }
            
            // 清除当前标签页的旧识别结果
            this.clearCurrentTabResults();
            
            this.ui.showLoading('正在识别...');
            this.ui.setFunctionButtonsEnabled(false);
            
            // 调整布局为双模型
            this.adjustTimelineLayout('dual');
            
            // 启动进度条动画（针对V2）
            this.startRecognizeV2Progress();
            
            // 重置 SMP 显示为识别中
            this.ui.updateSMPSpeaker('identifying', 0);
            this.startRecognizeSMPProgress();
            
            // 并行调用两个模型
            const v2Promise = this.api.recognizeSpeaker(audioBlob)
                .then(result => {
                    // V2 结果处理
                    if (result.success) {
                        const data = result.data;
                        const details = result.details || {};
                        const speakerId = data.speaker_id;
                        const confidence = data.confidence.toFixed(1);
                        
                        // 更新右侧当前说话人显示（传入所有片段）
                        this.updateCurrentSpeaker(speakerId, confidence, details.segments);
                        
                        const basicInfo = {
                            '主要识别用户': speakerId === 'unknown' ? '未识别到注册用户' : speakerId,
                            '整体相似度': `${confidence}%`,
                            '置信度': confidence >= 70 ? '高' : (confidence >= 50 ? '中' : '低'),
                            '识别模式': details.mode || 'N/A',
                            '检测到的说话人': details.detected_speakers ? details.detected_speakers.join(', ') : 'N/A',
                            '说话人数量': details.total_speakers || 0
                        };
                        
                        // 加载音频到时间轴播放器
                        this.loadTimelineAudio(audioBlob);
                        
                        // 获取处理时间
                        const processingTime = details.processing_time || 0;
                        
                        if (speakerId === 'unknown') {
                            this.ui.showRecognitionResult('warning', '识别结果', basicInfo, details.segments, processingTime);
                        } else {
                            this.ui.showRecognitionResult('success', '识别结果', basicInfo, details.segments, processingTime);
                        }
                    } else {
                        this.ui.showError('V2识别失败: ' + (result.error || '未知错误'));
                    }
                    this.completeRecognizeV2Progress();
                })
                .catch(error => {
                    console.error("V2 识别错误:", error);
                    this.ui.showError('V2识别失败: ' + error.message);
                    this.completeRecognizeV2Progress();
                });
                
            const smpPromise = this.api.recognizeSpeakerSMP(audioBlob)
                .then(result => {
                    console.log('SMP 识别结果:', result); // 调试日志
                    
                    // SMP 结果处理
                    if (result.success) {
                        const data = result.data;
                        const details = result.details || {};
                        const speakerId = data.speaker_id;
                        const confidence = data.confidence.toFixed(1);
                        
                        // 更新 SMP 说话人卡片
                        this.ui.updateSMPSpeaker(speakerId, confidence);
                        
                        // 显示 SMP 时间轴和分段结果（如果有数据）
                        if (details.segments && details.segments.length > 0) {
                            console.log('SMP segments 数据:', details.segments);
                            const processingTime = details.processing_time || 0;
                            this.ui.showSMPTimelineAndSegments(details.segments, processingTime);
                        } else {
                            console.log('SMP 没有返回 segments 数据');
                        }
                    } else {
                        this.ui.updateSMPSpeaker('error', 0, result.details || result.error || '未知错误');
                    }
                    this.completeRecognizeSMPProgress();
                })
                .catch(error => {
                    console.error("SMP 识别错误:", error);
                    this.ui.updateSMPSpeaker('error', 0, error.message);
                    this.completeRecognizeSMPProgress();
                });
                
            // 等待两个 Promise 都完成（但各自独立更新UI）
            await Promise.allSettled([v2Promise, smpPromise]);
            
            // 保存当前标签页的识别结果
            this.saveCurrentTabResults();
            
        } catch (error) {
            this.ui.showError(error.message);
        } finally {
            this.ui.setFunctionButtonsEnabled(true);
            // 隐藏进度条
            this.hideRecognizeV2Progress();
            this.hideRecognizeSMPProgress();
        }
    }
    
    /**
     * 单独识别 Remote-V2
     */
    async handleRecognizeV2() {
        try {
            let audioBlob = null;
            
            if (this.activeTab === 'upload') {
                audioBlob = this.uploadedAudioBlob || this.recorder.getAudioBlob();
            } else {
                audioBlob = this.recorder.getAudioBlob() || this.uploadedAudioBlob;
            }
            
            if (!audioBlob) {
                this.ui.showError('请先录音或上传音频文件');
                return;
            }
            
            // 清除当前标签页的旧识别结果
            this.clearCurrentTabResults();
            
            this.ui.setFunctionButtonsEnabled(false);
            
            // 调整布局为单模型 - V2占满
            this.adjustTimelineLayout('v2-only');
            
            // 启动V2进度条
            this.startRecognizeV2Progress();
            
            const result = await this.api.recognizeSpeaker(audioBlob);
            
            if (result.success) {
                const data = result.data;
                const details = result.details || {};
                const speakerId = data.speaker_id;
                const confidence = data.confidence.toFixed(1);
                
                // 更新说话人显示
                this.updateCurrentSpeaker(speakerId, confidence, details.segments);
                
                // 加载音频到时间轴播放器
                this.loadTimelineAudio(audioBlob);
                
                // 显示时间轴和分段结果
                const processingTime = details.processing_time || 0;
                const basicInfo = {};
                this.ui.showRecognitionResult('success', '识别结果', basicInfo, details.segments, processingTime);
            } else {
                this.ui.showError('Remote-V2识别失败: ' + (result.error || '未知错误'));
            }
            
            this.completeRecognizeV2Progress();
            
            // 保存当前标签页的识别结果
            this.saveCurrentTabResults();
        } catch (error) {
            console.error("Remote-V2 识别错误:", error);
            this.ui.showError('Remote-V2识别失败: ' + error.message);
            this.completeRecognizeV2Progress();
        } finally {
            this.ui.setFunctionButtonsEnabled(true);
            this.hideRecognizeV2Progress();
        }
    }
    
    /**
     * 单独识别 SMP
     */
    async handleRecognizeSMP() {
        try {
            let audioBlob = null;
            
            if (this.activeTab === 'upload') {
                audioBlob = this.uploadedAudioBlob || this.recorder.getAudioBlob();
            } else {
                audioBlob = this.recorder.getAudioBlob() || this.uploadedAudioBlob;
            }
            
            if (!audioBlob) {
                this.ui.showError('请先录音或上传音频文件');
                return;
            }
            
            // 清除当前标签页的旧识别结果
            this.clearCurrentTabResults();
            
            this.ui.setFunctionButtonsEnabled(false);
            
            // 调整布局为单模型 - SMP占满
            this.adjustTimelineLayout('smp-only');
            
            // 启动SMP进度条
            this.ui.updateSMPSpeaker('identifying', 0);
            this.startRecognizeSMPProgress();
            
            const result = await this.api.recognizeSpeakerSMP(audioBlob);
            
            if (result.success) {
                const data = result.data;
                const details = result.details || {};
                const speakerId = data.speaker_id;
                const confidence = data.confidence.toFixed(1);
                
                // 更新 SMP 说话人卡片
                this.ui.updateSMPSpeaker(speakerId, confidence);
                
                // 显示 SMP 时间轴和分段结果
                if (details.segments && details.segments.length > 0) {
                    const processingTime = details.processing_time || 0;
                    this.ui.showSMPTimelineAndSegments(details.segments, processingTime);
                }
            } else {
                this.ui.updateSMPSpeaker('error', 0, result.details || result.error || '未知错误');
            }
            
            this.completeRecognizeSMPProgress();
            
            // 保存当前标签页的识别结果
            this.saveCurrentTabResults();
        } catch (error) {
            console.error("SMP 识别错误:", error);
            this.ui.updateSMPSpeaker('error', 0, error.message);
            this.completeRecognizeSMPProgress();
        } finally {
            this.ui.setFunctionButtonsEnabled(true);
            this.hideRecognizeSMPProgress();
        }
    }
    
    /**
     * 启动Remote-V2识别进度（按时间分配百分比）
     */
    startRecognizeV2Progress() {
        const progressBar = document.getElementById('recognize-v2-progress-bar');
        const progressFill = document.getElementById('recognize-v2-progress-fill');
        const progressText = document.getElementById('recognize-v2-progress-text');
        const progressStage = document.getElementById('recognize-v2-progress-stage');
        
        if (!progressBar || !progressFill || !progressText || !progressStage) return;
        
        progressBar.style.display = 'block';
        
        // 根据实际处理时间分配百分比（基于技术白皮书架构）
        const stages = [
            { start: 0, end: 5, text: '音频预处理...', duration: 200 },
            { start: 5, end: 15, text: 'VAD切分...', duration: 400 },
            { start: 15, end: 35, text: '特征提取...', duration: 800 },
            { start: 35, end: 50, text: '向量比对...', duration: 600 },
            { start: 50, end: 70, text: 'ASR识别...', duration: 800 },
            { start: 70, end: 80, text: '对齐分析...', duration: 400 },
            { start: 80, end: 90, text: '边界校正...', duration: 400 },
            { start: 90, end: 95, text: '生成结果...', duration: 200 }
        ];
        
        this.recognizeV2ProgressInterval = null;
        
        const animateStage = (stageIndex) => {
            if (stageIndex >= stages.length) return;
            
            const stage = stages[stageIndex];
            const startPercent = stage.start;
            const endPercent = stage.end;
            const duration = stage.duration;
            const steps = 30;
            const stepDuration = duration / steps;
            const stepIncrement = (endPercent - startPercent) / steps;
            
            progressStage.textContent = stage.text;
            
            let currentStep = 0;
            let currentPercent = startPercent;
            
            this.recognizeV2ProgressInterval = setInterval(() => {
                currentStep++;
                currentPercent += stepIncrement;
                
                if (currentStep >= steps) {
                    clearInterval(this.recognizeV2ProgressInterval);
                    progressFill.style.width = endPercent + '%';
                    progressText.textContent = Math.round(endPercent) + '%';
                    setTimeout(() => animateStage(stageIndex + 1), 50);
                } else {
                    progressFill.style.width = currentPercent + '%';
                    progressText.textContent = Math.round(currentPercent) + '%';
                }
            }, stepDuration);
        };
        
        animateStage(0);
    }
    
    /**
     * 完成Remote-V2识别进度到100%
     */
    completeRecognizeV2Progress() {
        const progressFill = document.getElementById('recognize-v2-progress-fill');
        const progressText = document.getElementById('recognize-v2-progress-text');
        const progressStage = document.getElementById('recognize-v2-progress-stage');
        
        if (!progressFill || !progressText || !progressStage) return;
        
        if (this.recognizeV2ProgressInterval) {
            clearInterval(this.recognizeV2ProgressInterval);
        }
        
        const currentPercent = parseFloat(progressFill.style.width) || 90;
        const targetPercent = 100;
        const duration = 400;
        const steps = 20;
        const stepDuration = duration / steps;
        const stepIncrement = (targetPercent - currentPercent) / steps;
        
        progressStage.textContent = '识别完成！';
        
        let step = 0;
        let percent = currentPercent;
        
        const completeInterval = setInterval(() => {
            step++;
            percent += stepIncrement;
            
            if (step >= steps) {
                clearInterval(completeInterval);
                progressFill.style.width = '100%';
                progressText.textContent = '100%';
            } else {
                progressFill.style.width = percent + '%';
                progressText.textContent = Math.round(percent) + '%';
            }
        }, stepDuration);
    }
    
    /**
     * 隐藏Remote-V2识别进度
     */
    hideRecognizeV2Progress() {
        const progressBar = document.getElementById('recognize-v2-progress-bar');
        if (progressBar) {
            setTimeout(() => {
                progressBar.style.display = 'none';
                const progressFill = document.getElementById('recognize-v2-progress-fill');
                const progressText = document.getElementById('recognize-v2-progress-text');
                if (progressFill) progressFill.style.width = '0%';
                if (progressText) progressText.textContent = '0%';
            }, 800);
        }
    }
    
    /**
     * 启动SMP识别进度
     */
    startRecognizeSMPProgress() {
        const progressBar = document.getElementById('recognize-smp-progress-bar');
        const progressFill = document.getElementById('recognize-smp-progress-fill');
        const progressText = document.getElementById('recognize-smp-progress-text');
        const progressStage = document.getElementById('recognize-smp-progress-stage');
        
        if (!progressBar || !progressFill || !progressText || !progressStage) return;
        
        progressBar.style.display = 'block';
        
        const stages = [
            { start: 0, end: 10, text: '音频预处理...', duration: 300 },
            { start: 10, end: 30, text: 'VAD切分...', duration: 500 },
            { start: 30, end: 60, text: '特征提取...', duration: 1000 },
            { start: 60, end: 85, text: '向量比对...', duration: 800 },
            { start: 85, end: 95, text: '生成结果...', duration: 300 }
        ];
        
        this.recognizeSMPProgressInterval = null;
        
        const animateStage = (stageIndex) => {
            if (stageIndex >= stages.length) return;
            
            const stage = stages[stageIndex];
            const startPercent = stage.start;
            const endPercent = stage.end;
            const duration = stage.duration;
            const steps = 30;
            const stepDuration = duration / steps;
            const stepIncrement = (endPercent - startPercent) / steps;
            
            progressStage.textContent = stage.text;
            
            let currentStep = 0;
            let currentPercent = startPercent;
            
            this.recognizeSMPProgressInterval = setInterval(() => {
                currentStep++;
                currentPercent += stepIncrement;
                
                if (currentStep >= steps) {
                    clearInterval(this.recognizeSMPProgressInterval);
                    progressFill.style.width = endPercent + '%';
                    progressText.textContent = Math.round(endPercent) + '%';
                    setTimeout(() => animateStage(stageIndex + 1), 50);
                } else {
                    progressFill.style.width = currentPercent + '%';
                    progressText.textContent = Math.round(currentPercent) + '%';
                }
            }, stepDuration);
        };
        
        animateStage(0);
    }
    
    /**
     * 完成SMP识别进度到100%
     */
    completeRecognizeSMPProgress() {
        const progressFill = document.getElementById('recognize-smp-progress-fill');
        const progressText = document.getElementById('recognize-smp-progress-text');
        const progressStage = document.getElementById('recognize-smp-progress-stage');
        
        if (!progressFill || !progressText || !progressStage) return;
        
        if (this.recognizeSMPProgressInterval) {
            clearInterval(this.recognizeSMPProgressInterval);
        }
        
        const currentPercent = parseFloat(progressFill.style.width) || 90;
        const targetPercent = 100;
        const duration = 400;
        const steps = 20;
        const stepDuration = duration / steps;
        const stepIncrement = (targetPercent - currentPercent) / steps;
        
        progressStage.textContent = '识别完成！';
        
        let step = 0;
        let percent = currentPercent;
        
        const completeInterval = setInterval(() => {
            step++;
            percent += stepIncrement;
            
            if (step >= steps) {
                clearInterval(completeInterval);
                progressFill.style.width = '100%';
                progressText.textContent = '100%';
            } else {
                progressFill.style.width = percent + '%';
                progressText.textContent = Math.round(percent) + '%';
            }
        }, stepDuration);
    }
    
    /**
     * 隐藏SMP识别进度
     */
    hideRecognizeSMPProgress() {
        const progressBar = document.getElementById('recognize-smp-progress-bar');
        if (progressBar) {
            setTimeout(() => {
                progressBar.style.display = 'none';
                const progressFill = document.getElementById('recognize-smp-progress-fill');
                const progressText = document.getElementById('recognize-smp-progress-text');
                if (progressFill) progressFill.style.width = '0%';
                if (progressText) progressText.textContent = '0%';
            }, 800);
        }
    }
    
    /**
     * 更新右侧当前说话人显示 - 显示所有识别到的人
     */
    updateCurrentSpeaker(speakerId, confidence, allSegments) {
        const container = document.getElementById('speakers-list-container');
        if (!container) return;
        
        // 统计所有说话人
        const speakers = {};
        if (allSegments && allSegments.length > 0) {
            allSegments.forEach(segment => {
                const user = segment.user || segment.user_id || 'unknown';
                if (user !== 'silence') {
                    if (!speakers[user]) {
                        speakers[user] = {
                            count: 0,
                            totalScore: 0,
                            totalDuration: 0
                        };
                    }
                    speakers[user].count++;
                    speakers[user].totalScore += (segment.score || 0);
                    speakers[user].totalDuration += ((segment.end || 0) - (segment.start || 0));
                }
            });
        }
        
        // 生成显示
        let html = '';
        const speakerList = Object.keys(speakers);
        
        if (speakerList.length === 0) {
            html = `
                <div class="text-center py-12 text-gray-400">
                    <i class="fa-solid fa-user-slash text-5xl mb-4"></i>
                    <p class="text-lg">未识别</p>
                </div>
            `;
        } else {
            speakerList.forEach((user, index) => {
                const avgScore = Math.round(speakers[user].totalScore / speakers[user].count);
                const duration = speakers[user].totalDuration.toFixed(1);
                const isUnknown = user === 'unknown';
                
                const colorClass = isUnknown ? 'bg-gray-50 border-gray-300' : 
                    (index === 0 ? 'bg-blue-50 border-blue-500' : 'bg-green-50 border-green-500');
                const textClass = isUnknown ? 'text-gray-600' : 
                    (index === 0 ? 'text-blue-900' : 'text-green-900');
                const iconClass = isUnknown ? 'fa-user-slash' : 'fa-user-check';
                
                html += `
                    <div class="border-l-4 ${colorClass} p-4 rounded-r shadow-sm">
                        <div class="flex items-center mb-2">
                            <i class="fa-solid ${iconClass} text-2xl ${textClass} mr-3"></i>
                            <div class="flex-1">
                                <h3 class="text-xl font-black ${textClass}">${user}</h3>
                                <p class="text-sm text-gray-500">${speakers[user].count} 个片段 · ${duration}秒</p>
                            </div>
                        </div>
                        <div class="flex items-center">
                            <div class="flex-grow bg-gray-200 h-2 rounded-full mr-3 overflow-hidden">
                                <div class="bg-blue-600 h-full" style="width: ${avgScore}%"></div>
                            </div>
                            <span class="${textClass} font-mono font-bold text-sm">${avgScore}%</span>
                        </div>
                    </div>
                `;
            });
        }
        
        container.innerHTML = html;
    }
    
    async handleListSpeakers() {
        try {
            console.log('开始获取用户列表...');
            const result = await this.api.getSpeakerList();
            console.log('API返回结果:', result);
            
            if (result.success) {
                const speakers = result.data.speakers;
                console.log('用户列表:', speakers);
                
                // 保存完整列表用于搜索
                this.allSpeakers = speakers;
                
                this.ui.onDeleteSpeaker = (speakerId) => this.handleDeleteSpeaker(speakerId);
                this.ui.showSpeakerList(speakers);
            } else {
                console.error('获取列表失败:', result.error);
                this.allSpeakers = [];
                this.ui.showError('获取列表失败: ' + (result.error || '未知错误'));
            }
            
        } catch (error) {
            console.error('获取用户列表异常:', error);
            this.allSpeakers = [];
            this.ui.showError('获取用户列表失败: ' + error.message);
        }
    }
    
    /**
     * 搜索用户
     */
    handleSearchSpeaker(keyword) {
        if (!this.allSpeakers) {
            return;
        }
        
        const searchTerm = keyword.trim().toLowerCase();
        
        if (!searchTerm) {
            // 如果搜索框为空，显示所有用户
            this.ui.showSpeakerList(this.allSpeakers);
            return;
        }
        
        // 过滤用户列表
        const filteredSpeakers = this.allSpeakers.filter(speaker => {
            const speakerName = (speaker.speaker_id || speaker.name || '').toLowerCase();
            return speakerName.includes(searchTerm);
        });
        
        this.ui.showSpeakerList(filteredSpeakers);
    }
    
    async handleDeleteSpeaker(speakerId) {
        try {
            if (!confirm(`确定要删除用户 "${speakerId}" 吗？\n删除后无法恢复！`)) {
                return;
            }
            
            this.ui.showLoading(`正在删除用户 "${speakerId}"...`);
            
            const result = await this.api.deleteSpeaker(speakerId);
            
            if (result.success) {
                this.ui.showSuccess(`用户 "${speakerId}" 已删除`);
                setTimeout(() => {
                    this.handleListSpeakers();
                }, 1000);
            } else {
                this.ui.showError('删除失败: ' + (result.error || '未知错误'));
            }
            
        } catch (error) {
            this.ui.showError(error.message);
        }
    }
    
    // ==================== 注册流程方法 ====================
    
    /**
     * 开始录音注册流程
     */
    startRecordRegister() {
        const speakerId = document.getElementById('register-speaker-id')?.value.trim();
        const errorEl = document.getElementById('register-name-error');
        
        if (!speakerId) {
            if (errorEl) {
                errorEl.textContent = '请先输入人员姓名！';
                errorEl.style.display = 'block';
            }
            return;
        }
        
        if (!/^[a-zA-Z0-9_]+$/.test(speakerId)) {
            if (errorEl) {
                errorEl.textContent = '用户名只能包含字母、数字和下划线';
                errorEl.style.display = 'block';
            }
            return;
        }
        
        if (errorEl) errorEl.style.display = 'none';
        
        // 初始化注册状态
        this.registerState = {
            speakerId: speakerId,
            currentStep: 1,
            recordings: [],
            sentences: [
                '我爱学习人工智能和深度学习技术',
                '今天天气真好，适合出去走走',
                '声纹识别系统可以准确识别说话人'
            ]
        };
        
        // 显示录音界面
        document.getElementById('register-init').style.display = 'none';
        document.getElementById('register-recording').style.display = 'block';
        const nameEdit = document.getElementById('register-user-name-edit');
        if (nameEdit) nameEdit.value = speakerId;
        document.getElementById('register-sentence').textContent = this.registerState.sentences[0];
        this.updateStepIndicator(1);
    }
    
    /**
     * 开始上传注册流程
     */
    startUploadRegister() {
        const speakerId = document.getElementById('register-speaker-id')?.value.trim();
        const errorEl = document.getElementById('register-name-error');
        
        if (!speakerId) {
            if (errorEl) {
                errorEl.textContent = '请先输入人员姓名！';
                errorEl.style.display = 'block';
            }
            return;
        }
        
        if (!/^[a-zA-Z0-9_]+$/.test(speakerId)) {
            if (errorEl) {
                errorEl.textContent = '用户名只能包含字母、数字和下划线';
                errorEl.style.display = 'block';
            }
            return;
        }
        
        if (errorEl) errorEl.style.display = 'none';
        
        this.registerState = {
            speakerId: speakerId,
            uploadedFile: null
        };
        
        // 显示上传界面
        document.getElementById('register-init').style.display = 'none';
        document.getElementById('register-upload').style.display = 'block';
        const nameEdit = document.getElementById('upload-user-name-edit');
        if (nameEdit) nameEdit.value = speakerId;
        
        // 触发文件选择
        setTimeout(() => {
            const fileInput = document.getElementById('register-audio-input');
            if (fileInput) {
                fileInput.click();
                fileInput.onchange = (e) => this.handleUploadFileSelected(e);
            }
        }, 100);
    }
    
    /**
     * 处理注册录音
     */
    async handleRegisterRecord() {
        try {
            await this.recorder.startRecording();
            document.getElementById('register-record-btn').disabled = true;
            document.getElementById('register-stop-btn').disabled = false;
        } catch (error) {
            alert('录音失败: ' + error.message);
        }
    }
    
    /**
     * 停止注册录音
     */
    async handleRegisterStop() {
        try {
            const result = await this.recorder.stopRecording();
            document.getElementById('register-record-btn').disabled = false;
            document.getElementById('register-stop-btn').disabled = true;
            
            const duration = await this.recorder.getAudioDuration();
            
            // 检查录音质量
            if (duration < 3) {
                this.showRegisterFeedback('error', '录音时长太短（少于3秒），请重新录制');
                return;
            }
            
            if (duration > 10) {
                this.showRegisterFeedback('warning', '录音时长较长（超过10秒），建议控制在5-8秒');
            }
            
            // 保存当前录音
            this.registerState.recordings.push(result.blob);
            this.showRegisterFeedback('success', `第${this.registerState.currentStep}段录制成功！`);
            
            // 进入下一步或完成注册
            setTimeout(() => {
                if (this.registerState.currentStep < 3) {
                    this.registerState.currentStep++;
                    this.updateStepIndicator(this.registerState.currentStep);
                    document.getElementById('current-step').textContent = this.registerState.currentStep;
                    document.getElementById('register-sentence').textContent = 
                        this.registerState.sentences[this.registerState.currentStep - 1];
                    document.getElementById('register-feedback').style.display = 'none';
                } else {
                    this.completeRecordRegister();
                }
            }, 1500);
            
        } catch (error) {
            alert('停止录音失败: ' + error.message);
        }
    }
    
    /**
     * 完成录音注册
     */
    async completeRecordRegister() {
        try {
            this.showRegisterFeedback('info', '正在处理注册...');
            
            // 获取最终的用户名（可能已修改）
            const finalName = document.getElementById('register-user-name-edit')?.value.trim() || this.registerState.speakerId;
            
            // 合并3段录音（这里简化处理，实际应该发送所有录音到后端）
            // 暂时使用第一段录音作为代表
            const audioBlob = this.registerState.recordings[0];
            
            const result = await this.api.registerSpeaker(finalName, audioBlob);
            
            if (result.success) {
                this.showRegisterFeedback('success', '声纹注册成功！');
                // 刷新注册库列表
                this.handleListSpeakers();
            } else {
                this.showRegisterFeedback('error', '注册失败: ' + (result.error || '未知错误'));
            }
        } catch (error) {
            this.showRegisterFeedback('error', '注册失败: ' + error.message);
        }
    }
    
    /**
     * 处理上传文件选择
     */
    async handleUploadFileSelected(event) {
        const file = event.target.files[0];
        if (!file) {
            this.cancelRegister();
            return;
        }
        
        // 验证文件格式
        const fileName = file.name.toLowerCase();
        const audioExtensions = this.fileSettings.allowed_extensions.map(ext => ext.toLowerCase());
        const isAudioFile = audioExtensions.some(ext => fileName.endsWith(ext)) || file.type.startsWith('audio/') || file.type.startsWith('video/');
        
        if (!isAudioFile) {
            const allowedStr = this.fileSettings.allowed_extensions.join(', ');
            this.showUploadFeedback('error', `不支持的文件格式，支持的格式: ${allowedStr}`);
            event.target.value = '';
            this.cancelRegister();
            return;
        }
        
        // 验证文件大小
        if (file.size > this.fileSettings.max_audio_size) {
            this.showUploadFeedback('error', `文件大小超过${this.fileSettings.max_audio_size_mb}MB限制`);
            event.target.value = '';
            this.cancelRegister();
            return;
        }
        
        // 显示文件信息
        const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
        document.getElementById('upload-file-name').textContent = file.name;
        document.getElementById('upload-file-size').textContent = fileSizeMB + ' MB';
        document.getElementById('upload-file-info').style.display = 'block';
        
        // 预览音频
        const audioUrl = URL.createObjectURL(file);
        const audioPlayer = document.getElementById('upload-audio-player');
        audioPlayer.src = audioUrl;
        
        this.registerState.uploadedFile = file;
        
        // 检查音频质量
        const audio = new Audio(audioUrl);
        audio.addEventListener('loadedmetadata', () => {
            if (audio.duration < 10) {
                this.showUploadFeedback('warning', '音频时长较短（少于10秒），建议使用15-30秒的音频以获得更好效果');
                document.getElementById('upload-save-btn').disabled = false;
            } else {
                this.showUploadFeedback('success', `音频质量良好（时长: ${audio.duration.toFixed(1)}秒）`);
                document.getElementById('upload-save-btn').disabled = false;
            }
        });
    }
    
    /**
     * 保存上传注册
     */
    async saveUploadRegister() {
        try {
            this.showUploadFeedback('info', '正在注册...');
            document.getElementById('upload-save-btn').disabled = true;
            
            // 启动注册进度（快速到85%）
            this.startRegisterProgress();
            
            // 获取最终的用户名（可能已修改）
            const finalName = document.getElementById('upload-user-name-edit')?.value.trim() || this.registerState.speakerId;
            
            const result = await this.api.registerSpeaker(
                finalName, 
                this.registerState.uploadedFile
            );
            
            // 完成进度到100%
            this.completeRegisterProgress();
            
            if (result.success) {
                this.showUploadFeedback('success', '声纹注册成功！');
                // 刷新注册库列表
                this.handleListSpeakers();
                // 重新启用保存按钮，允许用户继续操作
                document.getElementById('upload-save-btn').disabled = false;
            } else {
                this.showUploadFeedback('error', '注册失败: ' + (result.error || '未知错误'));
                document.getElementById('upload-save-btn').disabled = false;
                document.getElementById('register-progress').style.display = 'none';
                document.getElementById('register-progress-bar').style.display = 'none';
            }
        } catch (error) {
            this.showUploadFeedback('error', '注册失败: ' + error.message);
            document.getElementById('upload-save-btn').disabled = false;
            document.getElementById('register-progress').style.display = 'none';
            document.getElementById('register-progress-bar').style.display = 'none';
        }
    }
    
    /**
     * 启动注册进度（按时间分配百分比）
     */
    startRegisterProgress() {
        const progressDiv = document.getElementById('register-progress');
        const progressBar = document.getElementById('register-progress-bar');
        const progressFill = document.getElementById('register-progress-fill');
        const progressText = document.getElementById('register-progress-text');
        const progressStage = document.getElementById('register-progress-stage');
        
        // 显示详细进度面板
        if (progressDiv) {
            progressDiv.style.display = 'block';
        }
        
        // 显示进度条
        if (progressBar) {
            progressBar.style.display = 'block';
        }
        
        // 根据实际处理时间分配百分比（基于技术白皮书注册架构）
        // 3.1 音频预处理与验证 - 5%
        // 3.2 VAD切分与质量检测 - 20%
        // 3.3 声纹特征提取 - 40%
        // 3.4 阶梯式质量控制 - 10%
        // 3.5 加权平均计算 - 15%
        // 3.6 向量存储 - 10%
        const stages = [
            { id: 'preprocess', name: '音频预处理与验证', start: 0, end: 5, duration: 200 },
            { id: 'vad', name: 'VAD切分音频片段', start: 5, end: 15, duration: 400 },
            { id: 'quality', name: '质量检测（SNR过滤）', start: 15, end: 25, duration: 400 },
            { id: 'extract', name: '声纹特征提取（远端大模型）', start: 25, end: 65, duration: 1600 },
            { id: 'decision', name: '阶梯式质量控制', start: 65, end: 75, duration: 400 },
            { id: 'aggregate', name: '加权平均计算（质心）', start: 75, end: 90, duration: 600 },
            { id: 'save', name: '保存声纹模板', start: 90, end: 95, duration: 200 }
        ];
        
        this.registerProgressInterval = null;
        
        const animateStage = (stageIndex) => {
            if (stageIndex >= stages.length) return;
            
            const stage = stages[stageIndex];
            const stageEl = document.getElementById(`stage-${stage.id}`);
            const startPercent = stage.start;
            const endPercent = stage.end;
            const duration = stage.duration;
            const steps = 30; // 每个阶段30步
            const stepDuration = duration / steps;
            const stepIncrement = (endPercent - startPercent) / steps;
            
            if (progressStage) progressStage.textContent = stage.name;
            
            // 显示加载图标
            if (stageEl) {
                const spinner = stageEl.querySelector('.fa-spin');
                const circle = stageEl.querySelector('.fa-circle');
                if (spinner && circle) {
                    spinner.style.display = 'inline-block';
                    circle.style.display = 'none';
                }
            }
            
            let currentStep = 0;
            let currentPercent = startPercent;
            
            this.registerProgressInterval = setInterval(() => {
                currentStep++;
                currentPercent += stepIncrement;
                
                if (currentStep >= steps) {
                    clearInterval(this.registerProgressInterval);
                    if (progressFill) progressFill.style.width = endPercent + '%';
                    if (progressText) progressText.textContent = Math.round(endPercent) + '%';
                    
                    // 完成当前阶段图标
                    if (stageEl) {
                        const spinner = stageEl.querySelector('.fa-spin');
                        const circle = stageEl.querySelector('.fa-circle');
                        if (spinner && circle) {
                            spinner.style.display = 'none';
                            circle.style.display = 'inline-block';
                            circle.classList.remove('text-gray-300');
                            circle.classList.add('text-green-600');
                        }
                    }
                    
                    // 进入下一个阶段
                    setTimeout(() => animateStage(stageIndex + 1), 50);
                } else {
                    if (progressFill) progressFill.style.width = currentPercent + '%';
                    if (progressText) progressText.textContent = Math.round(currentPercent) + '%';
                }
            }, stepDuration);
        };
        
        animateStage(0);
    }
    
    /**
     * 完成注册进度到100%（平滑渐进）
     */
    completeRegisterProgress() {
        const progressFill = document.getElementById('register-progress-fill');
        const progressText = document.getElementById('register-progress-text');
        const progressStage = document.getElementById('register-progress-stage');
        
        if (!progressFill || !progressText || !progressStage) return;
        
        // 清除之前的定时器
        if (this.registerProgressInterval) {
            clearInterval(this.registerProgressInterval);
        }
        
        // 获取当前进度
        const currentPercent = parseFloat(progressFill.style.width) || 85;
        const targetPercent = 100;
        const duration = 500; // 完成阶段持续500ms
        const steps = 25;
        const stepDuration = duration / steps;
        const stepIncrement = (targetPercent - currentPercent) / steps;
        
        progressStage.textContent = '注册完成！';
        
        let step = 0;
        let percent = currentPercent;
        
        const completeInterval = setInterval(() => {
            step++;
            percent += stepIncrement;
            
            if (step >= steps) {
                clearInterval(completeInterval);
                progressFill.style.width = '100%';
                progressText.textContent = '100%';
            } else {
                progressFill.style.width = percent + '%';
                progressText.textContent = Math.round(percent) + '%';
            }
        }, stepDuration);
    }
    
    /**
     * 重新选择上传文件
     */
    reselectUploadFile() {
        const fileInput = document.getElementById('register-audio-input');
        if (fileInput) {
            fileInput.value = '';
            fileInput.click();
        }
    }
    
    /**
     * 取消注册/继续注册（重置表单）
     */
    cancelRegister() {
        // 重置所有注册相关的显示
        document.getElementById('register-init').style.display = 'block';
        document.getElementById('register-recording').style.display = 'none';
        document.getElementById('register-upload').style.display = 'none';
        
        // 清空输入
        const speakerIdInput = document.getElementById('register-speaker-id');
        if (speakerIdInput) speakerIdInput.value = '';
        
        // 隐藏文件信息
        const fileInfo = document.getElementById('upload-file-info');
        if (fileInfo) fileInfo.style.display = 'none';
        
        // 隐藏进度
        const registerProgress = document.getElementById('register-progress');
        if (registerProgress) registerProgress.style.display = 'none';
        
        const progressBar = document.getElementById('register-progress-bar');
        if (progressBar) progressBar.style.display = 'none';
        
        // 重置进度条
        const progressFill = document.getElementById('register-progress-fill');
        const progressText = document.getElementById('register-progress-text');
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = '0%';
        
        // 重置所有阶段图标
        const stages = ['preprocess', 'vad', 'quality', 'extract', 'decision', 'aggregate', 'save'];
        stages.forEach(stage => {
            const stageEl = document.getElementById(`stage-${stage}`);
            if (stageEl) {
                const spinner = stageEl.querySelector('.fa-spin');
                const circle = stageEl.querySelector('.fa-circle');
                if (spinner) spinner.style.display = 'none';
                if (circle) {
                    circle.style.display = 'inline-block';
                    circle.classList.remove('text-green-600');
                    circle.classList.add('text-gray-300');
                }
            }
        });
        
        // 清空反馈信息
        const uploadFeedback = document.getElementById('upload-feedback');
        if (uploadFeedback) uploadFeedback.style.display = 'none';
        
        const registerFeedback = document.getElementById('register-feedback');
        if (registerFeedback) registerFeedback.style.display = 'none';
        
        // 清空错误提示
        const nameError = document.getElementById('register-name-error');
        if (nameError) nameError.style.display = 'none';
        
        // 重置状态
        this.registerState = null;
    }
    
    /**
     * 更新步骤指示器
     */
    updateStepIndicator(step) {
        for (let i = 1; i <= 3; i++) {
            const indicator = document.getElementById(`step-${i}`);
            if (indicator) {
                if (i <= step) {
                    indicator.className = 'w-3 h-3 rounded-full bg-blue-600';
                } else {
                    indicator.className = 'w-3 h-3 rounded-full bg-gray-300';
                }
            }
        }
    }
    
    /**
     * 显示注册反馈
     */
    showRegisterFeedback(type, message) {
        const feedback = document.getElementById('register-feedback');
        if (!feedback) return;
        
        const colors = {
            success: 'bg-green-50 border-green-500 text-green-800',
            error: 'bg-red-50 border-red-500 text-red-800',
            warning: 'bg-yellow-50 border-yellow-500 text-yellow-800',
            info: 'bg-blue-50 border-blue-500 text-blue-800'
        };
        
        feedback.className = `p-4 rounded-lg border-l-4 ${colors[type]}`;
        feedback.textContent = message;
        feedback.style.display = 'block';
    }
    
    /**
     * 显示上传反馈
     */
    showUploadFeedback(type, message) {
        const feedback = document.getElementById('upload-feedback');
        if (!feedback) return;
        
        const colors = {
            success: 'bg-green-50 border-green-500 text-green-800',
            error: 'bg-red-50 border-red-500 text-red-800',
            warning: 'bg-yellow-50 border-yellow-500 text-yellow-800',
            info: 'bg-blue-50 border-blue-500 text-blue-800'
        };
        
        feedback.className = `p-4 rounded-lg border-l-4 ${colors[type]}`;
        feedback.textContent = message;
        feedback.style.display = 'block';
    }
    
    // ==================== 时间轴播放方法 ====================
    
    /**
     * 加载音频到时间轴播放器
     */
    loadTimelineAudio(audioBlob) {
        const audio = document.getElementById('timeline-audio');
        if (!audio) return;
        
        // 创建音频URL
        const audioUrl = URL.createObjectURL(audioBlob);
        audio.src = audioUrl;
        
        // 重置播放按钮状态
        const playBtn = document.getElementById('timeline-play-btn');
        const pauseBtn = document.getElementById('timeline-pause-btn');
        if (playBtn) playBtn.style.display = 'flex';
        if (pauseBtn) pauseBtn.style.display = 'none';
    }
    
    /**
     * 播放时间轴音频
     */
    playTimeline() {
        const audio = document.getElementById('timeline-audio');
        const playBtn = document.getElementById('timeline-play-btn');
        const pauseBtn = document.getElementById('timeline-pause-btn');
        
        if (!audio || !audio.src) {
            alert('请先导入音频文件并完成识别');
            return;
        }
        
        audio.play();
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'flex';
    }
    
    /**
     * 暂停时间轴音频
     */
    pauseTimeline() {
        const audio = document.getElementById('timeline-audio');
        const playBtn = document.getElementById('timeline-play-btn');
        const pauseBtn = document.getElementById('timeline-pause-btn');
        
        if (audio) {
            audio.pause();
            playBtn.style.display = 'flex';
            pauseBtn.style.display = 'none';
        }
    }
    
    /**
     * 更新时间轴播放进度
     */
    updateTimelineProgress() {
        const audio = document.getElementById('timeline-audio');
        const timeDisplay = document.getElementById('timeline-current-time');
        const progressIndicator = document.getElementById('timeline-progress-indicator');
        
        if (!audio || !timeDisplay) return;
        
        const currentTime = this.formatTime(audio.currentTime);
        const duration = this.formatTime(audio.duration || 0);
        timeDisplay.textContent = `${currentTime} / ${duration}`;
        
        // 更新进度指示器位置
        if (progressIndicator && audio.duration > 0) {
            const percent = (audio.currentTime / audio.duration) * 100;
            progressIndicator.style.left = `${percent}%`;
            progressIndicator.style.display = 'block';
        }
        
        // 高亮当前播放的片段
        this.highlightCurrentSegment(audio.currentTime);
    }
    
    /**
     * 高亮当前播放的片段
     */
    highlightCurrentSegment(currentTime) {
        const timelineContainer = document.querySelector('.timeline-container');
        if (!timelineContainer) return;
        
        // 移除所有高亮
        const allSegments = timelineContainer.querySelectorAll('div[title]');
        allSegments.forEach(seg => {
            seg.classList.remove('ring-4', 'ring-yellow-400', 'z-10');
        });
        
        // 找到当前时间对应的片段并高亮
        allSegments.forEach(seg => {
            const title = seg.getAttribute('title');
            if (title) {
                const match = title.match(/\((\d+\.?\d*)-(\d+\.?\d*)s\)/);
                if (match) {
                    const start = parseFloat(match[1]);
                    const end = parseFloat(match[2]);
                    if (currentTime >= start && currentTime <= end) {
                        seg.classList.add('ring-4', 'ring-yellow-400', 'z-10');
                    }
                }
            }
        });
    }
    
    /**
     * 时间轴播放结束
     */
    onTimelineEnded() {
        const playBtn = document.getElementById('timeline-play-btn');
        const pauseBtn = document.getElementById('timeline-pause-btn');
        const progressIndicator = document.getElementById('timeline-progress-indicator');
        
        if (playBtn && pauseBtn) {
            playBtn.style.display = 'flex';
            pauseBtn.style.display = 'none';
        }
        
        // 隐藏进度指示器
        if (progressIndicator) {
            progressIndicator.style.display = 'none';
        }
        
        // 移除所有高亮
        const timelineContainer = document.querySelector('.timeline-container');
        if (timelineContainer) {
            const allSegments = timelineContainer.querySelectorAll('div[title]');
            allSegments.forEach(seg => {
                seg.classList.remove('ring-4', 'ring-yellow-400', 'z-10');
            });
        }
    }
    
    /**
     * 更新SMP时间轴播放进度
     */
    updateSMPTimelineProgress() {
        const audio = document.getElementById('smp-timeline-audio');
        const timeDisplay = document.getElementById('smp-timeline-current-time');
        const progressIndicator = document.getElementById('smp-timeline-progress-indicator');
        
        if (!audio || !timeDisplay) return;
        
        const currentTime = this.formatTime(audio.currentTime);
        const duration = this.formatTime(audio.duration || 0);
        timeDisplay.textContent = `${currentTime} / ${duration}`;
        
        // 更新进度指示器位置
        if (progressIndicator && audio.duration > 0) {
            const percent = (audio.currentTime / audio.duration) * 100;
            progressIndicator.style.left = `${percent}%`;
            progressIndicator.style.display = 'block';
        }
    }
    
    /**
     * SMP时间轴播放结束
     */
    onSMPTimelineEnded() {
        const playBtn = document.getElementById('smp-timeline-play-btn');
        const pauseBtn = document.getElementById('smp-timeline-pause-btn');
        const progressIndicator = document.getElementById('smp-timeline-progress-indicator');
        
        if (playBtn && pauseBtn) {
            playBtn.style.display = 'flex';
            pauseBtn.style.display = 'none';
        }
        
        // 隐藏进度指示器
        if (progressIndicator) {
            progressIndicator.style.display = 'none';
        }
    }
    
    /**
     * 格式化时间（秒转为 mm:ss）
     */
    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * 跳转到指定时间并播放
     */
    seekToTime(time) {
        const audio = document.getElementById('timeline-audio');
        const playBtn = document.getElementById('timeline-play-btn');
        const pauseBtn = document.getElementById('timeline-pause-btn');
        
        if (!audio || !audio.src) {
            alert('请先完成音频识别');
            return;
        }
        
        audio.currentTime = time;
        audio.play();
        
        if (playBtn && pauseBtn) {
            playBtn.style.display = 'none';
            pauseBtn.style.display = 'flex';
        }
    }
    
    /**
     * 切换测试音频显示/隐藏
     */
    toggleTestAudio() {
        const container = document.getElementById('test-audio-container');
        const icon = document.getElementById('test-audio-toggle-icon');
        
        if (!container || !icon) return;
        
        if (container.style.display === 'none') {
            // 显示并加载
            container.style.display = 'block';
            icon.className = 'fa-solid fa-chevron-up ml-auto text-gray-400';
            this.loadTestAudioList();
        } else {
            // 隐藏
            container.style.display = 'none';
            icon.className = 'fa-solid fa-chevron-down ml-auto text-gray-400';
        }
    }
    
    /**
     * 加载测试音频列表
     */
    async loadTestAudioList() {
        try {
            const response = await fetch('/api/test-audio/list');
            const result = await response.json();
            
            const listContainer = document.getElementById('test-audio-list');
            if (!listContainer) return;
            
            if (!result.success || result.files.length === 0) {
                listContainer.innerHTML = '<p class="text-xs text-gray-400 text-center py-4">暂无测试音频</p>';
                return;
            }
            
            let html = '';
            result.files.forEach(file => {
                html += `
                    <div class="test-audio-item flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-blue-50 transition cursor-pointer group" 
                         data-filename="${file.name}" draggable="true">
                        <div class="flex items-center space-x-2 flex-1 pointer-events-none">
                            <i class="fa-solid fa-file-audio text-blue-500 text-xs"></i>
                            <span class="text-xs font-medium text-gray-700 group-hover:text-blue-600">${file.name}</span>
                            <span class="text-xs text-gray-400">(${file.size_mb} MB)</span>
                        </div>
                        <div class="flex items-center space-x-1">
                            <button class="test-audio-delete-btn bg-red-400 hover:bg-red-500 text-white px-2 py-1 rounded text-xs transition" data-filename="${file.name}">
                                <i class="fa-solid fa-trash text-[10px]"></i>
                            </button>
                        </div>
                    </div>
                `;
            });
            
            listContainer.innerHTML = html;
            
            // 绑定点击事件
            listContainer.querySelectorAll('.test-audio-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (!e.target.closest('.test-audio-delete-btn')) {
                        const filename = item.getAttribute('data-filename');
                        this.showTestAudioPlayer(filename);
                    }
                });
                
                // 绑定拖拽事件
                item.addEventListener('dragstart', (e) => {
                    const filename = item.getAttribute('data-filename');
                    e.dataTransfer.setData('test-audio-filename', filename);
                    e.dataTransfer.effectAllowed = 'copy';
                });
            });
            
            // 绑定删除按钮事件
            listContainer.querySelectorAll('.test-audio-delete-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation(); // 阻止触发播放事件
                    const filename = btn.getAttribute('data-filename');
                    this.deleteTestAudio(filename);
                });
            });
            
        } catch (error) {
            console.error('加载测试音频列表失败:', error);
        }
    }
    
    /**
     * 显示测试音频播放器
     */
    async showTestAudioPlayer(filename) {
        try {
            const player = document.getElementById('test-audio-player');
            const container = document.getElementById('test-audio-player-container');
            const nameSpan = document.getElementById('test-audio-player-name');
            
            if (!player || !container || !nameSpan) return;
            
            // 设置音频源
            const audioUrl = `/api/test-audio/${filename}`;
            player.src = audioUrl;
            
            // 显示播放器
            nameSpan.textContent = filename;
            container.style.display = 'block';
            
            console.log('显示测试音频播放器:', filename);
        } catch (error) {
            console.error('显示测试音频播放器失败:', error);
        }
    }
    
    /**
     * 关闭测试音频播放器
     */
    closeTestAudioPlayer() {
        const player = document.getElementById('test-audio-player');
        const container = document.getElementById('test-audio-player-container');
        
        if (!player || !container) return;
        
        // 停止播放
        player.pause();
        player.src = '';
        
        // 隐藏播放器
        container.style.display = 'none';
    }
    
    /**
     * 从测试音频加载文件（拖拽使用）
     */
    async loadTestAudioFile(filename) {
        try {
            const response = await fetch(`/api/test-audio/${filename}`);
            const blob = await response.blob();
            
            // 创建File对象
            const file = new File([blob], filename, { type: 'audio/wav' });
            
            // 设置为上传的文件（视为导入音频文件）
            this.uploadedFile = file;
            this.uploadedAudioBlob = file;
            this.showFileInfo(file);
            this.ui.setFunctionButtonsEnabled(true);
            
            // 更新识别按钮状态
            this.updateRecognizeButtonState();
            
            // 显示音频预览
            const audioUrl = URL.createObjectURL(file);
            const audio = new Audio(audioUrl);
            
            audio.addEventListener('loadedmetadata', () => {
                this.showUploadAudioPreview(audioUrl, audio.duration);
            });
            
            audio.addEventListener('error', () => {
                this.showUploadAudioPreview(null, 0);
            });
            
        } catch (error) {
            this.ui.showError('加载测试音频失败: ' + error.message);
        }
    }
    
    /**
     * 删除测试音频
     */
    async deleteTestAudio(filename) {
        if (!confirm(`确定要删除测试音频 "${filename}" 吗？`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/test-audio/${filename}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            
            if (result.success) {
                // 重新加载列表
                this.loadTestAudioList();
            } else {
                this.ui.showError('删除失败: ' + result.error);
            }
        } catch (error) {
            this.ui.showError('删除测试音频失败: ' + error.message);
        }
    }
    
    /**
     * 上传测试音频
     */
    async uploadTestAudio(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/test-audio/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                // 重新加载列表
                this.loadTestAudioList();
                // 清空输入
                const input = document.getElementById('test-audio-upload-input');
                if (input) input.value = '';
            } else {
                this.ui.showError('上传失败: ' + result.error);
            }
        } catch (error) {
            this.ui.showError('上传测试音频失败: ' + error.message);
        }
    }
    
    /**
     * 调整时间轴布局
     * @param {string} mode - 'dual' | 'v2-only' | 'smp-only'
     */
    adjustTimelineLayout(mode) {
        const v2Wrapper = document.getElementById('v2-timeline-wrapper');
        const smpWrapper = document.getElementById('smp-timeline-wrapper');
        
        if (!v2Wrapper || !smpWrapper) return;
        
        if (mode === 'dual') {
            // 双模型：各占一半
            v2Wrapper.className = 'col-span-6 space-y-6';
            smpWrapper.className = 'col-span-6 space-y-6';
            v2Wrapper.style.display = 'block';
            smpWrapper.style.display = 'block';
        } else if (mode === 'v2-only') {
            // 只显示V2：占满
            v2Wrapper.className = 'col-span-12 space-y-6';
            smpWrapper.style.display = 'none';
        } else if (mode === 'smp-only') {
            // 只显示SMP：占满
            v2Wrapper.style.display = 'none';
            smpWrapper.className = 'col-span-12 space-y-6';
        }
    }
    
    /**
     * 点击时间轴跳转播放（远端大模型）
     */
    seekTimelineByClick(event, element, maxTime) {
        const rect = element.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const percent = clickX / rect.width;
        const seekTime = percent * maxTime;
        
        const audio = document.getElementById('timeline-audio');
        if (audio && audio.src) {
            audio.currentTime = seekTime;
            audio.play();
            
            // 更新播放按钮状态
            const playBtn = document.getElementById('timeline-play-btn');
            const pauseBtn = document.getElementById('timeline-pause-btn');
            if (playBtn && pauseBtn) {
                playBtn.style.display = 'none';
                pauseBtn.style.display = 'flex';
            }
        }
    }
    
    /**
     * 点击时间轴跳转播放（离线小模型）
     */
    seekSMPTimelineByClick(event, element, maxTime) {
        const rect = element.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const percent = clickX / rect.width;
        const seekTime = percent * maxTime;
        
        const audio = document.getElementById('smp-timeline-audio');
        if (audio && audio.src) {
            audio.currentTime = seekTime;
            audio.play();
            
            // 更新播放按钮状态
            const playBtn = document.getElementById('smp-timeline-play-btn');
            const pauseBtn = document.getElementById('smp-timeline-pause-btn');
            if (playBtn && pauseBtn) {
                playBtn.style.display = 'none';
                pauseBtn.style.display = 'flex';
            }
        }
    }
    
    cleanup() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
        }
        
        if (this.recorder) {
            this.recorder.cleanup(false);
        }
        
        if (this.ui) {
            this.ui.cleanup();
        }
    }
    
    /**
     * 打开配置弹窗
     */
    async openConfigModal() {
        const modal = document.getElementById('config-modal');
        const content = document.getElementById('config-content');
        const errorDiv = document.getElementById('config-error');
        const successDiv = document.getElementById('config-success');
        const progressDiv = document.getElementById('config-restart-progress');
        
        if (!modal || !content) return;
        
        // 隐藏所有消息
        if (errorDiv) errorDiv.style.display = 'none';
        if (successDiv) successDiv.style.display = 'none';
        if (progressDiv) progressDiv.style.display = 'none';
        
        try {
            // 获取配置文件内容
            const result = await this.api.getModelConfig();
            content.value = result.content;
            modal.style.display = 'flex';
        } catch (error) {
            alert('获取配置文件失败: ' + error.message);
        }
    }
    
    /**
     * 关闭配置弹窗
     */
    closeConfigModal() {
        const modal = document.getElementById('config-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    /**
     * 保存配置并重启服务
     */
    async saveAndRestartConfig() {
        const content = document.getElementById('config-content');
        const errorDiv = document.getElementById('config-error');
        const successDiv = document.getElementById('config-success');
        const progressDiv = document.getElementById('config-restart-progress');
        const statusText = document.getElementById('config-restart-status');
        const progressBar = document.getElementById('config-restart-bar');
        const restartBtn = document.getElementById('config-restart-btn');
        const closeBtn = document.getElementById('config-close-btn');
        const textArea = document.getElementById('config-content');
        
        if (!content) return;
        
        // 隐藏之前的消息
        if (errorDiv) errorDiv.style.display = 'none';
        if (successDiv) successDiv.style.display = 'none';
        if (progressDiv) progressDiv.style.display = 'none';
        
        // 确认重启
        const confirmed = confirm('确定要保存配置并重启服务吗？\n\n重启期间服务将暂时不可用（约10-15秒）。');
        if (!confirmed) return;
        
        // 禁用所有交互
        if (restartBtn) restartBtn.disabled = true;
        if (closeBtn) closeBtn.disabled = true;
        if (textArea) textArea.disabled = true;
        
        try {
            // 显示进度
            if (progressDiv) progressDiv.style.display = 'block';
            if (statusText) statusText.textContent = '正在保存配置文件...';
            if (progressBar) progressBar.style.width = '10%';
            
            // 保存配置并重启
            const result = await this.api.updateModelConfig(content.value, true);
            
            if (statusText) statusText.textContent = '配置已保存，服务正在重启...';
            if (progressBar) progressBar.style.width = '30%';
            
            // 等待2秒让服务开始重启
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            if (statusText) statusText.textContent = '等待服务重启完成...';
            if (progressBar) progressBar.style.width = '50%';
            
            // 轮询检查服务是否恢复
            const maxAttempts = 30; // 最多尝试30次（30秒）
            let attempts = 0;
            let serviceReady = false;
            
            while (attempts < maxAttempts && !serviceReady) {
                attempts++;
                const progress = 50 + (attempts / maxAttempts) * 40; // 50% -> 90%
                if (progressBar) progressBar.style.width = `${progress}%`;
                if (statusText) statusText.textContent = `检查服务状态... (${attempts}/${maxAttempts})`;
                
                try {
                    // 尝试健康检查
                    await this.api.healthCheck();
                    serviceReady = true;
                    
                    if (statusText) statusText.textContent = '服务已恢复，即将刷新页面...';
                    if (progressBar) progressBar.style.width = '100%';
                    
                    // 等待1秒后刷新页面
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    window.location.reload();
                } catch (error) {
                    // 服务还未恢复，继续等待
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            // 如果超时仍未恢复
            if (!serviceReady) {
                if (errorDiv) {
                    errorDiv.querySelector('p').textContent = '服务重启超时，请手动刷新页面或检查服务状态';
                    errorDiv.style.display = 'block';
                }
                if (progressDiv) progressDiv.style.display = 'none';
                
                // 恢复交互
                if (restartBtn) restartBtn.disabled = false;
                if (closeBtn) closeBtn.disabled = false;
                if (textArea) textArea.disabled = false;
            }
        } catch (error) {
            // 显示错误消息
            if (errorDiv) {
                errorDiv.querySelector('p').textContent = '保存失败: ' + error.message;
                errorDiv.style.display = 'block';
            }
            if (progressDiv) progressDiv.style.display = 'none';
            
            // 恢复交互
            if (restartBtn) restartBtn.disabled = false;
            if (closeBtn) closeBtn.disabled = false;
            if (textArea) textArea.disabled = false;
        }
    }
}

// 全局应用实例
let app = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', async () => {
    app = new VoiceprintApp();
    await app.init();
    
    // 暴露为全局变量，供HTML中的onclick调用
    window.appController = app;
});

// 页面卸载前清理
window.addEventListener('beforeunload', () => {
    if (app) {
        app.cleanup();
    }
});
