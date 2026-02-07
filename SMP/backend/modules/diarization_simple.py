"""
简化版说话人分离模块
使用滑动窗口 + 声纹识别实现基础的说话人分段功能
"""
import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple, Any


class SimpleDiarization:
    """简化版说话人分离
    
    使用滑动窗口对音频进行分段识别，实现基础的说话人分离功能
    """
    
    def __init__(
        self,
        window_size: float = 2.0,      # 窗口大小（秒）
        step_size: float = 1.0,         # 步长（秒）
        min_segment_duration: float = 0.5,  # 最小分段时长（秒）
        merge_threshold: float = 0.3    # 合并阈值（秒）
    ):
        """初始化
        
        Args:
            window_size: 滑动窗口大小（秒）
            step_size: 滑动窗口步长（秒）
            min_segment_duration: 最小分段时长（秒）
            merge_threshold: 相邻同说话人分段的合并阈值（秒）
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_segment_duration = min_segment_duration
        self.merge_threshold = merge_threshold
    
    def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id_wrapper
    ) -> List[Dict[str, Any]]:
        """处理音频，生成说话人分段
        
        Args:
            audio_data: 音频数据（numpy数组）
            sample_rate: 采样率
            speaker_id_wrapper: 声纹识别器实例
            
        Returns:
            分段列表，每个分段包含：
            - start: 开始时间（秒）
            - end: 结束时间（秒）
            - user: 说话人ID
            - user_id: 说话人ID（同user）
            - confidence: 置信度（0-1）
            - type: 类型（'human'）
        """
        # 计算窗口和步长的样本数
        window_samples = int(self.window_size * sample_rate)
        step_samples = int(self.step_size * sample_rate)
        
        # 音频总时长
        total_duration = len(audio_data) / sample_rate
        
        # 如果音频太短，直接识别整个音频
        if total_duration < self.window_size:
            speaker_id, similarity = speaker_id_wrapper.match_multi(audio_data)
            return [{
                'type': 'human',
                'user': speaker_id,
                'user_id': speaker_id,
                'start': 0.0,
                'end': total_duration,
                'confidence': float(similarity)
            }]
        
        # 滑动窗口识别
        raw_segments = []
        current_pos = 0
        
        while current_pos + window_samples <= len(audio_data):
            # 提取窗口音频
            window_audio = audio_data[current_pos:current_pos + window_samples]
            
            # 识别说话人
            speaker_id, similarity = speaker_id_wrapper.match_multi(window_audio)
            
            # 计算时间
            start_time = current_pos / sample_rate
            end_time = (current_pos + window_samples) / sample_rate
            
            raw_segments.append({
                'start': start_time,
                'end': end_time,
                'user': speaker_id,
                'confidence': float(similarity)
            })
            
            # 移动窗口
            current_pos += step_samples
        
        # 处理最后一个窗口（如果有剩余）
        if current_pos < len(audio_data):
            window_audio = audio_data[current_pos:]
            if len(window_audio) / sample_rate >= self.min_segment_duration:
                speaker_id, similarity = speaker_id_wrapper.match_multi(window_audio)
                start_time = current_pos / sample_rate
                end_time = len(audio_data) / sample_rate
                
                raw_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'user': speaker_id,
                    'confidence': float(similarity)
                })
        
        # 合并相邻的同说话人分段
        merged_segments = self._merge_segments(raw_segments)
        
        # 转换为最终格式
        final_segments = []
        for seg in merged_segments:
            final_segments.append({
                'type': 'human',
                'user': seg['user'],
                'user_id': seg['user'],
                'start': seg['start'],
                'end': seg['end'],
                'confidence': seg['confidence']
            })
        
        return final_segments
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并相邻的同说话人分段
        
        Args:
            segments: 原始分段列表
            
        Returns:
            合并后的分段列表
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for i in range(1, len(segments)):
            seg = segments[i]
            
            # 如果是同一个说话人，且时间间隔小于阈值，则合并
            if (seg['user'] == current['user'] and 
                seg['start'] - current['end'] <= self.merge_threshold):
                # 合并：扩展结束时间，更新置信度（取平均）
                current['end'] = seg['end']
                current['confidence'] = (current['confidence'] + seg['confidence']) / 2
            else:
                # 不同说话人或间隔太大，保存当前分段，开始新分段
                if current['end'] - current['start'] >= self.min_segment_duration:
                    merged.append(current)
                current = seg.copy()
        
        # 添加最后一个分段
        if current['end'] - current['start'] >= self.min_segment_duration:
            merged.append(current)
        
        return merged


