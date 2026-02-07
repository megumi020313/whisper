/**
 * UI控制模块
 * 处理页面动态效果和录音文件预览
 */

class UIController {
    constructor() {
        this.elements = {};
        this.currentAudioUrl = null;
    }
    
    /**
     * 初始化UI元素引用
     */
    initElements() {
        // 录音控制
        this.elements.startRecordBtn = document.getElementById('start-record');
        this.elements.stopRecordBtn = document.getElementById('stop-record');
        this.elements.recordStatus = document.getElementById('record-status');
        this.elements.recordTime = document.getElementById('record-time');
        
        // 音频预览
        this.elements.audioPreview = document.getElementById('audio-preview');
        this.elements.audioPlayer = document.getElementById('audio-player');
        this.elements.audioDuration = document.getElementById('audio-duration');
        
        // 用户名输入
        this.elements.speakerIdInput = document.getElementById('speaker-id');
        
        // 功能按钮
        this.elements.registerBtn = document.getElementById('register-btn');
        this.elements.recognizeBtn = document.getElementById('recognize-btn');
        this.elements.listBtn = document.getElementById('list-btn');
        
        // 结果显示
        this.elements.resultDiv = document.getElementById('result');
        this.elements.resultContent = document.getElementById('result-content');
        
        // 说话人列表
        this.elements.speakerList = document.getElementById('speaker-list');
        this.elements.speakerListContent = document.getElementById('speaker-list-content');
    }
    
    /**
     * 更新录音状态显示
     */
    updateRecordStatus(status, message) {
        if (!this.elements.recordStatus) return;
        
        this.elements.recordStatus.className = `status ${status}`;
        this.elements.recordStatus.textContent = message;
    }
    
    /**
     * 更新录音时间显示
     */
    updateRecordTime(seconds) {
        if (!this.elements.recordTime) return;
        
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        this.elements.recordTime.textContent = 
            `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * 显示录音文件预览
     */
    showAudioPreview(audioUrl, duration = null) {
        if (!this.elements.audioPreview || !this.elements.audioPlayer) return;
        
        // 清理旧的URL
        if (this.currentAudioUrl) {
            URL.revokeObjectURL(this.currentAudioUrl);
        }
        
        this.currentAudioUrl = audioUrl;
        
        // 更新播放器
        this.elements.audioPlayer.src = audioUrl;
        this.elements.audioPreview.style.display = 'block';
        
        // 更新时长
        if (duration && this.elements.audioDuration) {
            this.elements.audioDuration.textContent = `时长: ${duration.toFixed(1)}秒`;
        }
    }
    
    /**
     * 隐藏录音文件预览
     */
    hideAudioPreview() {
        if (!this.elements.audioPreview) return;
        
        this.elements.audioPreview.style.display = 'none';
        
        if (this.currentAudioUrl) {
            URL.revokeObjectURL(this.currentAudioUrl);
            this.currentAudioUrl = null;
        }
    }
    
    /**
     * 切换按钮状态
     */
    toggleButtons(recording) {
        if (this.elements.startRecordBtn) {
            this.elements.startRecordBtn.disabled = recording;
        }
        if (this.elements.stopRecordBtn) {
            this.elements.stopRecordBtn.disabled = !recording;
        }
    }
    
    /**
     * 设置功能按钮状态
     */
    setFunctionButtonsEnabled(enabled) {
        const registerBtn = document.getElementById('register-btn');
        const registerBtnPage = document.getElementById('register-btn-page');
        const recognizeBtnTop = document.getElementById('recognize-btn-top');
        const recognizeV2Btn = document.getElementById('recognize-v2-btn');
        const recognizeSMPBtn = document.getElementById('recognize-smp-btn');
        
        if (registerBtn) {
            registerBtn.disabled = !enabled;
        }
        if (registerBtnPage) {
            registerBtnPage.disabled = !enabled;
        }
        if (recognizeBtnTop) {
            recognizeBtnTop.disabled = !enabled;
        }
        if (recognizeV2Btn) {
            recognizeV2Btn.disabled = !enabled;
        }
        if (recognizeSMPBtn) {
            recognizeSMPBtn.disabled = !enabled;
        }
    }
    
    /**
     * 显示结果
     */
    showResult(type, title, content) {
        const resultDiv = document.getElementById('result');
        const resultContent = document.getElementById('result-content');
        
        if (!resultDiv || !resultContent) return;
        
        resultDiv.className = `result-box ${type}`;
        
        let html = `<h3>${title}</h3>`;
        
        if (typeof content === 'string') {
            html += `<p>${content}</p>`;
        } else if (typeof content === 'object') {
            html += '<div class="result-details">';
            for (const [key, value] of Object.entries(content)) {
                html += `<p><strong>${key}:</strong> ${value}</p>`;
            }
            html += '</div>';
        }
        
        resultContent.innerHTML = html;
        resultDiv.style.display = 'block';
        
        // 自动滚动到结果区域
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    /**
     * 隐藏结果
     */
    hideResult() {
        const resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
    }
    
    /**
     * 显示加载中
     */
    showLoading(message = '处理中...') {
        this.showResult('info', '提示', message);
    }
    
    /**
     * 显示错误
     */
    showError(message) {
        // 检查是否包含诊断链接
        if (message.includes('/test-recorder')) {
            // 分离消息和链接
            const parts = message.split('🔍');
            const mainMessage = parts[0].trim();
            const linkPart = parts[1] ? parts[1].trim() : '';
            
            // 提取URL
            const urlMatch = linkPart.match(/(https?:\/\/[^\s]+)/);
            const diagnosticUrl = urlMatch ? urlMatch[1] : '/test-recorder';
            
            // 创建带链接的内容
            const content = `
                <p>${mainMessage.replace(/\n/g, '<br>')}</p>
                <div style="margin-top: 15px; padding: 15px; background: #e3f2fd; border-radius: 5px;">
                    <p style="margin: 0 0 10px 0;">
                        <strong>🔍 需要帮助？</strong>
                    </p>
                    <a href="${diagnosticUrl}" 
                       target="_blank" 
                       style="display: inline-block; padding: 10px 20px; background: #2196f3; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
                        打开录音诊断工具
                    </a>
                    <p style="margin: 10px 0 0 0; font-size: 0.9em; color: #666;">
                        诊断工具会帮您检测浏览器、权限和API支持情况
                    </p>
                </div>
            `;
            
            this.showResult('error', '错误', content);
        } else {
            this.showResult('error', '错误', message);
        }
    }
    
    /**
     * 显示成功
     */
    showSuccess(message) {
        this.showResult('success', '成功', message);
    }
    
    /**
     * 显示识别结果（包含分段详情）
     */
    showRecognitionResult(type, title, basicInfo, segments, processingTime = 0) {
        // 只显示分段结果和时间轴，不显示详细表格
        this.showSegmentsList(segments);
        this.showTimeline(segments);
        
        // 显示处理时间
        if (processingTime > 0) {
            const processingTimeEl = document.getElementById('processing-time');
            const processingTimeValue = document.getElementById('processing-time-value');
            if (processingTimeEl && processingTimeValue) {
                processingTimeValue.textContent = processingTime.toFixed(2);
                processingTimeEl.style.display = 'inline-flex';
            }
        }
    }
    
    /**
     * 显示分段列表（模板样式）- 不同人不同颜色
     */
    showSegmentsList(segments) {
        const segmentsCard = document.getElementById('result-segments');
        const segmentsList = document.getElementById('segments-list');
        
        if (!segmentsCard || !segmentsList || !segments || segments.length === 0) {
            if (segmentsCard) segmentsCard.style.display = 'none';
            return;
        }
        
        // 收集所有用户并分配颜色
        const userColors = {};
        const colors = [
            { bg: 'bg-blue-50', border: 'border-blue-500', text: 'text-blue-900', bar: 'bg-blue-600' },
            { bg: 'bg-green-50', border: 'border-green-500', text: 'text-green-900', bar: 'bg-green-600' },
            { bg: 'bg-purple-50', border: 'border-purple-500', text: 'text-purple-900', bar: 'bg-purple-600' },
            { bg: 'bg-orange-50', border: 'border-orange-500', text: 'text-orange-900', bar: 'bg-orange-600' },
            { bg: 'bg-pink-50', border: 'border-pink-500', text: 'text-pink-900', bar: 'bg-pink-600' }
        ];
        
        let colorIndex = 0;
        segments.forEach(segment => {
            const user = segment.user || segment.user_id || 'unknown';
            if (!userColors[user]) {
                if (user === 'unknown' || user === 'silence') {
                    userColors[user] = { bg: 'bg-gray-50', border: 'border-gray-300', text: 'text-gray-400', bar: 'bg-gray-400' };
                } else {
                    userColors[user] = colors[colorIndex % colors.length];
                    colorIndex++;
                }
            }
        });
        
        // 添加表头
        let html = `
            <div class="grid grid-cols-12 gap-2 mb-3 px-3 text-xs font-bold text-gray-500 uppercase tracking-wider">
                <div class="col-span-2">时间范围</div>
                <div class="col-span-1">时长</div>
                <div class="col-span-2">说话人</div>
                <div class="col-span-3">识别文本</div>
                <div class="col-span-3">置信度</div>
                <div class="col-span-1 text-right">Z-score</div>
            </div>
        `;
        
        segments.forEach((segment, index) => {
            const startTime = segment.start ? segment.start.toFixed(2) : '0.00';
            const endTime = segment.end ? segment.end.toFixed(2) : '0.00';
            const duration = ((segment.end || 0) - (segment.start || 0)).toFixed(2);
            const user = segment.user || segment.user_id || 'unknown';
            
            // 后端返回的字段：
            // segment.raw_z_score: 原始 Z-score 值 (0-8 范围)
            // segment.score: 已转换的百分比 (0-100)
            const rawScore = segment.raw_z_score !== undefined ? segment.raw_z_score : 
                           (segment.avg_similarity !== undefined ? segment.avg_similarity : 0);
            const scorePercent = segment.score !== undefined ? Math.round(segment.score) : 0;
            
            // 获取真实时间
            const absoluteStart = segment.absolute_start_time || '';
            const absoluteEnd = segment.absolute_end_time || '';
            
            const color = userColors[user];
            
            // 格式化真实时间
            let timeDisplay = `${startTime}s - ${endTime}s`;
            let realTimeDisplay = '';
            if (absoluteStart && absoluteEnd) {
                const startDate = new Date(absoluteStart);
                const endDate = new Date(absoluteEnd);
                realTimeDisplay = `${startDate.toLocaleTimeString('zh-CN')} - ${endDate.toLocaleTimeString('zh-CN')}`;
            }
            
            // 获取ASR文本
            const text = segment.text || '';
            const textDisplay = text ? text : '<span class="text-gray-400 italic text-xs">未启用ASR</span>';
            const titleText = text ? `\n文本: ${text}` : '';
            
            html += `
                <div class="grid grid-cols-12 gap-2 items-center p-3 ${color.bg} border-l-4 ${color.border} rounded-r shadow-sm mb-2" 
                     title="片段${index + 1}: ${user} (Z-score: ${rawScore.toFixed(2)}, 置信度: ${scorePercent}%)${titleText}">
                    <div class="col-span-2">
                        <div class="text-xs font-mono text-gray-600">${timeDisplay}</div>
                        ${realTimeDisplay ? `<div class="text-[10px] text-gray-400 mt-1">${realTimeDisplay}</div>` : ''}
                    </div>
                    <div class="col-span-1">
                        <span class="text-xs font-bold ${color.text}">${duration}s</span>
                    </div>
                    <div class="col-span-2">
                        <span class="font-bold ${color.text}">${user}</span>
                    </div>
                    <div class="col-span-3">
                        <div class="text-xs ${color.text} line-clamp-2 leading-tight" title="${text}">${textDisplay}</div>
                    </div>
                    <div class="col-span-3 flex items-center gap-2">
                        <div class="flex-grow bg-gray-200 h-2 rounded-full overflow-hidden">
                            <div class="${color.bar} h-full transition-all" style="width: ${scorePercent}%"></div>
                        </div>
                        <span class="${color.text} font-mono text-xs whitespace-nowrap">${scorePercent}%</span>
                    </div>
                    <div class="col-span-1 text-right">
                        <span class="${color.text} font-mono font-bold text-sm">${rawScore.toFixed(2)}</span>
                    </div>
                </div>
            `;
        });
        
        segmentsList.innerHTML = html;
        segmentsCard.style.display = 'block';
    }
    
    /**
     * 显示时间轴（Diarization）- 只显示人的片段
     */
    showTimeline(segments) {
        const timelineCard = document.getElementById('timeline-result');
        const timelineContent = document.getElementById('timeline-content');
        
        if (!timelineCard || !timelineContent || !segments || segments.length === 0) {
            if (timelineCard) timelineCard.style.display = 'none';
            return;
        }
        
        // 过滤掉silence片段
        const humanSegments = segments.filter(seg => {
            const user = seg.user || seg.user_id || '';
            return user !== 'silence';
        });
        
        if (humanSegments.length === 0) {
            timelineCard.style.display = 'none';
            return;
        }
        
        // 计算总时长
        let maxTime = 0;
        segments.forEach(segment => {
            if (segment.end > maxTime) maxTime = segment.end;
        });
        
        // 生成时间轴HTML
        let html = '<div class="relative pt-8">';
        
        // 时间刻度（清晰显示，避免乱码）
        html += '<div class="absolute top-0 left-0 right-0 flex justify-between text-[10px] text-gray-500 font-mono px-2 mb-2">';
        const timeMarks = Math.ceil(maxTime / 5);
        for (let i = 0; i <= timeMarks; i++) {
            const timeLabel = `${i * 5}s`;
            html += `<span class="inline-block">${timeLabel}</span>`;
        }
        html += '</div>';
        
        // 时间轴容器（可点击任意位置播放）
        html += `<div class="h-12 w-full bg-gray-100 rounded-lg relative timeline-container flex items-center px-0 shadow-inner overflow-hidden border mt-1 cursor-pointer" 
                      onclick="window.appController.seekTimelineByClick(event, this, ${maxTime})" 
                      title="点击任意位置跳转播放">`;
        
        // 播放进度指示器
        html += `<div id="timeline-progress-indicator" class="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none" style="left: 0%; display: none;">
                    <div class="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full">
                        <div class="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-red-500"></div>
                    </div>
                 </div>`;
        
        // 绘制分段（只显示人的片段）- 与分段结果颜色对应
        const userColors = {};
        const colorClasses = ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-pink-500'];
        let colorIndex = 0;
        
        // 先收集所有用户并分配颜色
        const uniqueUsers = [...new Set(humanSegments.map(seg => seg.user || seg.user_id || 'unknown'))];
        uniqueUsers.forEach(user => {
            if (user === 'unknown') {
                userColors[user] = 'bg-gray-400';
            } else {
                userColors[user] = colorClasses[colorIndex % colorClasses.length];
                colorIndex++;
            }
        });
        
        humanSegments.forEach(segment => {
            const user = segment.user || segment.user_id || 'unknown';
            const start = segment.start || 0;
            const end = segment.end || 0;
            const duration = end - start;
            
            const leftPercent = (start / maxTime) * 100;
            const widthPercent = (duration / maxTime) * 100;
            const opacity = (user === 'unknown') ? 'opacity-50' : '';
            
            html += `<div class="${userColors[user]} h-8 rounded-sm hover:brightness-110 transition-all ${opacity} pointer-events-none" 
                          style="position: absolute; left: ${leftPercent}%; width: ${widthPercent}%;" 
                          title="${user} (${start.toFixed(1)}-${end.toFixed(1)}s)"
                          data-start="${start}" data-end="${end}"></div>`;
        });
        
        html += '</div>';
        
        // 图例（只显示人的）- 使用相同的颜色
        html += '<div class="mt-4 flex flex-wrap gap-4 text-[11px] font-medium">';
        uniqueUsers.forEach(user => {
            html += `<div class="flex items-center"><span class="w-3 h-3 ${userColors[user]} rounded-sm mr-2"></span> ${user}</div>`;
        });
        html += '</div>';
        
        html += '</div>';
        
        timelineContent.innerHTML = html;
        timelineCard.style.display = 'block';
    }
    
    /**
     * 显示说话人列表
     */
    showSpeakerList(speakers) {
        const speakerListContent = document.getElementById('speaker-list-content');
        if (!speakerListContent) return;
        
        // 更新计数
        const countElement = document.getElementById('speaker-count');
        if (countElement) {
            countElement.textContent = `${speakers.length} 条目`;
        }
        
        if (speakers.length === 0) {
            speakerListContent.innerHTML = '<p class="empty-message">暂无注册用户</p>';
        } else {
            let html = '<table class="speaker-table">';
            html += '<thead><tr><th class="px-6 py-4">姓名</th><th class="px-6 py-4 text-center">特征状态</th><th class="px-6 py-4">注册日期</th><th class="px-6 py-4 text-right">操作</th></tr></thead>';
            html += '<tbody class="text-sm divide-y divide-gray-100">';
            
            speakers.forEach(speaker => {
                const registeredAt = speaker.registered_at ? 
                    new Date(speaker.registered_at).toLocaleDateString('zh-CN') : '未知';
                
                html += `
                    <tr class="hover:bg-gray-50 transition">
                        <td class="px-6 py-4 font-bold text-gray-700">${speaker.speaker_id}</td>
                        <td class="px-6 py-4 text-center"><span class="px-2 py-1 bg-green-100 text-green-700 text-[10px] rounded-full font-bold">NORMAL</span></td>
                        <td class="px-6 py-4 text-gray-400 font-mono">${registeredAt}</td>
                        <td class="px-6 py-4 text-right space-x-3">
                            <button class="text-red-400 hover:text-red-600 delete-speaker-btn" data-speaker-id="${speaker.speaker_id}">
                                <i class="fa-solid fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            speakerListContent.innerHTML = html;
            
            // 绑定删除按钮事件
            const deleteButtons = speakerListContent.querySelectorAll('.delete-speaker-btn');
            deleteButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    const speakerId = btn.getAttribute('data-speaker-id');
                    if (this.onDeleteSpeaker) {
                        this.onDeleteSpeaker(speakerId);
                    }
                });
            });
        }
    }
    
    /**
     * 隐藏说话人列表
     */
    hideSpeakerList() {
        if (!this.elements.speakerList) return;
        this.elements.speakerList.style.display = 'none';
    }
    
    /**
     * 获取用户名输入
     */
    getSpeakerId() {
        if (!this.elements.speakerIdInput) return '';
        return this.elements.speakerIdInput.value.trim();
    }
    
    /**
     * 验证用户名输入
     */
    validateSpeakerId() {
        const speakerId = this.getSpeakerId();
        
        if (!speakerId) {
            this.showError('请输入用户名');
            return false;
        }
        
        if (!/^[a-zA-Z0-9_]+$/.test(speakerId)) {
            this.showError('用户名只能包含字母、数字和下划线');
            return false;
        }
        
        return true;
    }
    
    /**
     * 清理资源
     */
    cleanup() {
        if (this.currentAudioUrl) {
            URL.revokeObjectURL(this.currentAudioUrl);
            this.currentAudioUrl = null;
        }
    }

    /**
     * 更新 SMP 说话人显示
     */
    updateSMPSpeaker(speakerId, confidence, errorMsg = null) {
        const container = document.getElementById('smp-speakers-list-container');
        if (!container) return;
        
        if (speakerId === 'identifying') {
             container.innerHTML = `
                <div class="text-center py-12 text-blue-500">
                    <i class="fa-solid fa-circle-notch fa-spin text-5xl mb-4"></i>
                    <p class="text-lg">正在识别...</p>
                </div>
            `;
            return;
        }
        
        if (speakerId === 'error') {
             container.innerHTML = `
                <div class="text-center py-12 text-red-400">
                    <i class="fa-solid fa-triangle-exclamation text-5xl mb-4"></i>
                    <p class="text-lg">识别出错</p>
                    <p class="text-xs mt-2">${errorMsg || ''}</p>
                </div>
            `;
            return;
        }
        
        // 显示结果
        const isUnknown = speakerId === 'unknown';
        const colorClass = isUnknown ? 'bg-gray-50 border-gray-300' : 'bg-orange-50 border-orange-500';
        const textClass = isUnknown ? 'text-gray-600' : 'text-orange-900';
        const iconClass = isUnknown ? 'fa-user-slash' : 'fa-user-check';
        const barClass = isUnknown ? 'bg-gray-400' : 'bg-orange-500';
        
        const html = `
            <div class="border-l-4 ${colorClass} p-4 rounded-r shadow-sm">
                <div class="flex items-center mb-2">
                    <i class="fa-solid ${iconClass} text-2xl ${textClass} mr-3"></i>
                    <div class="flex-1">
                        <h3 class="text-xl font-black ${textClass}">${speakerId}</h3>
                        <p class="text-sm text-gray-500">Confidence</p>
                    </div>
                </div>
                <div class="flex items-center">
                    <div class="flex-grow bg-gray-200 h-2 rounded-full mr-3 overflow-hidden">
                        <div class="${barClass} h-full" style="width: ${confidence}%"></div>
                    </div>
                    <span class="${textClass} font-mono font-bold text-sm">${confidence}%</span>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }

    /**
     * 显示 SMP 时间轴和分段结果
     */
    showSMPTimelineAndSegments(segments, processingTime) {
        if (!segments || segments.length === 0) {
            return;
        }

        // 显示 SMP 时间轴
        this.showSMPTimeline(segments);
        
        // 显示 SMP 分段结果
        this.showSMPSegments(segments);
        
        // 显示处理时间
        if (processingTime) {
            const timeSpan = document.getElementById('smp-processing-time');
            const timeValue = document.getElementById('smp-processing-time-value');
            if (timeSpan && timeValue) {
                timeValue.textContent = processingTime.toFixed(2);
                timeSpan.style.display = 'inline-flex';
            }
        }
    }

    /**
     * 显示 SMP 时间轴
     */
    showSMPTimeline(segments) {
        const timelineCard = document.getElementById('smp-timeline-result');
        const timelineContent = document.getElementById('smp-timeline-content');
        
        if (!timelineCard || !timelineContent || !segments || segments.length === 0) {
            return;
        }

        // 过滤出人的片段
        const humanSegments = segments.filter(seg => seg.type === 'human');
        if (humanSegments.length === 0) {
            return;
        }

        const maxTime = Math.max(...humanSegments.map(seg => seg.end || 0));
        
        let html = '<div class="space-y-2">';
        
        // 时间轴容器（可点击任意位置播放）
        html += `<div class="h-12 w-full bg-gray-100 rounded-lg relative timeline-container flex items-center px-0 shadow-inner overflow-hidden border mt-1 cursor-pointer" 
                      onclick="window.appController.seekSMPTimelineByClick(event, this, ${maxTime})" 
                      title="点击任意位置跳转播放">`;
        
        // 播放进度指示器
        html += `<div id="smp-timeline-progress-indicator" class="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none" style="left: 0%; display: none;">
                    <div class="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full">
                        <div class="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-red-500"></div>
                    </div>
                 </div>`;
        
        // 绘制分段 - 使用橙色系
        const userColors = {};
        const colorClasses = ['bg-orange-500', 'bg-amber-500', 'bg-yellow-500', 'bg-red-500', 'bg-rose-500'];
        let colorIndex = 0;
        
        // 收集所有用户并分配颜色
        const uniqueUsers = [...new Set(humanSegments.map(seg => seg.user || seg.user_id || 'unknown'))];
        uniqueUsers.forEach(user => {
            if (user === 'unknown') {
                userColors[user] = 'bg-gray-400';
            } else {
                userColors[user] = colorClasses[colorIndex % colorClasses.length];
                colorIndex++;
            }
        });
        
        humanSegments.forEach(segment => {
            const user = segment.user || segment.user_id || 'unknown';
            const start = segment.start || 0;
            const end = segment.end || 0;
            const duration = end - start;
            
            const leftPercent = (start / maxTime) * 100;
            const widthPercent = (duration / maxTime) * 100;
            const opacity = (user === 'unknown') ? 'opacity-50' : '';
            
            html += `<div class="${userColors[user]} h-8 rounded-sm hover:brightness-110 transition-all ${opacity} pointer-events-none" 
                          style="position: absolute; left: ${leftPercent}%; width: ${widthPercent}%;" 
                          title="${user} (${start.toFixed(1)}-${end.toFixed(1)}s)"
                          data-start="${start}" data-end="${end}"></div>`;
        });
        
        html += '</div>';
        
        // 图例
        html += '<div class="mt-4 flex flex-wrap gap-4 text-[11px] font-medium">';
        uniqueUsers.forEach(user => {
            html += `<div class="flex items-center"><span class="w-3 h-3 ${userColors[user]} rounded-sm mr-2"></span> ${user}</div>`;
        });
        html += '</div>';
        
        html += '</div>';
        
        timelineContent.innerHTML = html;
        timelineCard.style.display = 'block';
    }

    /**
     * 显示 SMP 分段结果
     */
    showSMPSegments(segments) {
        const segmentsCard = document.getElementById('smp-result-segments');
        const segmentsList = document.getElementById('smp-segments-list');
        
        if (!segmentsCard || !segmentsList || !segments || segments.length === 0) {
            return;
        }

        // 过滤出人的片段
        const humanSegments = segments.filter(seg => seg.type === 'human');
        if (humanSegments.length === 0) {
            return;
        }

        let html = '';
        humanSegments.forEach((segment, index) => {
            const user = segment.user || segment.user_id || 'unknown';
            const start = segment.start || 0;
            const end = segment.end || 0;
            const duration = (end - start).toFixed(1);
            const confidence = segment.confidence ? (segment.confidence * 100).toFixed(1) : 'N/A';
            
            const isUnknown = user === 'unknown';
            const bgClass = isUnknown ? 'bg-gray-50' : 'bg-orange-50';
            const borderClass = isUnknown ? 'border-gray-300' : 'border-orange-300';
            const textClass = isUnknown ? 'text-gray-600' : 'text-orange-900';
            
            html += `
                <div class="${bgClass} border ${borderClass} p-4 rounded-lg">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <span class="text-xs font-mono ${textClass} bg-white px-2 py-1 rounded">#${index + 1}</span>
                            <span class="font-bold ${textClass}">${user}</span>
                            <span class="text-xs text-gray-500">${start.toFixed(1)}s - ${end.toFixed(1)}s (${duration}s)</span>
                        </div>
                        <span class="text-xs ${textClass} font-mono">置信度: ${confidence}%</span>
                    </div>
                </div>
            `;
        });
        
        segmentsList.innerHTML = html;
        segmentsCard.style.display = 'block';
    }
}

// 导出为全局变量
window.UIController = UIController;

