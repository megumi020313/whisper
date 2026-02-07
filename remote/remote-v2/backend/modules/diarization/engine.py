"""对话智能引擎 (Diarization Engine)

V3.0 核心算法：以ASR词语为骨架，以声纹特征为血肉，构建高置信度对话流。
V3.1 优化：基于多模态证据链的"邻近优先"安全吸附模型
V3.3 两阶段校正：边界校正（动态梯度决策）
V3.4 平滑修正：三明治法则（修复A->B->A短时跳变）

核心思想：
1. 以"语言事件"（词语）为最小原子单位
2. 通过时间对齐，将声纹识别结果"认领"到每个词
3. 使用锚点思想，进行置信度感知的智能聚合
4. V3.1新增：三把安全锁的间隙词仲裁机制
   - 第一把锁：Z-score静态防御（声学拒识）
   - 第二把锁：ASR语义与置信度保护（语义不可侵犯）
   - 第三把锁：双向相似度权重（DNA鉴定）
5. V3.3新增：边界校正（基于SV片段嫌疑区的动态梯度决策）
6. V3.4新增：平滑修正（基于时长的A-B-A夹心结构修正）
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np

from backend.utils.logger import get_logger
from backend.core.config import get_config
from backend.core.vector_storage import VectorStorage

logger = get_logger()


class DiarizationEngine:
    """对话智能引擎
    
    负责将碎片化的SV和ASR结果融合成高质量的对话流。
    """
    
    def __init__(
        self,
        time_threshold: float = 1.5,
        similarity_threshold: float = 0.80,
        low_confidence_threshold: float = 2.5,
        max_merge_duration: float = 10.0
    ):
        """初始化对话智能引擎
        
        Args:
            time_threshold: 时间间隔阈值（秒），超过此值认为是不同说话人
            similarity_threshold: 声纹相似度阈值，低于此值认为是不同说话人
            low_confidence_threshold: 低置信度阈值（Z-score），低于此值需要锚点修正
            max_merge_duration: 最大合并时长（秒），超过此值需要安全校验
        """
        self.logger = get_logger()
        self.config = get_config()
        
        self.time_threshold = time_threshold
        self.similarity_threshold = similarity_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.max_merge_duration = max_merge_duration
        
        # V3.1.1 重叠质量检查配置
        diarization_config = self.config.__dict__
        self.min_overlap_ratio = diarization_config.get('diarization_min_overlap_ratio', 0.5)
        
        # V3.2 词-说话人对齐的声纹验证配置
        self.min_similarity_for_alignment = diarization_config.get('diarization_min_similarity_for_alignment', 0.65)
        
        # V3.3 两阶段校正配置（核心机制）
        self.boundary_correction_enabled = diarization_config.get('boundary_correction_enabled', True)
        self.boundary_correction_window_s = diarization_config.get('boundary_correction_window_s', 1.5)
        self.boundary_correction_min_confidence_diff = diarization_config.get('boundary_correction_min_confidence_diff', 0.1)
        
        # 初始化VectorStorage用于边界校正（获取注册模板）
        try:
            self.vector_storage = VectorStorage()
        except Exception as e:
            self.logger.warning(f"无法初始化VectorStorage: {e}, 边界校正将被禁用")
            self.vector_storage = None
            self.boundary_correction_enabled = False
        
        self.logger.info(
            f"对话智能引擎初始化 (V3.3两阶段): time_threshold={time_threshold}s, "
            f"similarity_threshold={similarity_threshold}, "
            f"low_confidence_threshold={low_confidence_threshold}, "
            f"min_overlap_ratio={self.min_overlap_ratio}, "
            f"min_similarity_for_alignment={self.min_similarity_for_alignment}, "
            f"boundary_correction={'enabled' if self.boundary_correction_enabled else 'disabled'}"
        )
    
    def process(
        self,
        sv_results: List[Dict[str, Any]],
        asr_results: List[Dict[str, Any]],
        log_path: str = None,
        audio_tensor: Optional[np.ndarray] = None,
        speaker_model: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """处理SV和ASR结果，生成高质量对话流
        
        Args:
            sv_results: 声纹识别结果列表
                格式: [{start, end, speaker, z_score, embedding}, ...]
            asr_results: ASR词级别结果列表
                格式: [{word, start, end, confidence}, ...]
            log_path: 日志文件路径（用于追加 Diarization 日志）
            audio_tensor: 原始音频数据（用于边界校正时提取词级别特征）
            speaker_model: 声纹识别模型实例（用于边界校正时提取词级别特征）
        
        Returns:
            高质量对话流列表
                格式: [{start, end, speaker, transcript, avg_z_score, word_count}, ...]
        """
        if not asr_results:
            self.logger.warning("ASR结果为空，无法生成对话流")
            return []
        
        if not sv_results:
            self.logger.warning("SV结果为空，无法确定说话人身份")
            return []
        
        # 保存动态资源引用（仅在process执行期间有效）
        self.audio_tensor = audio_tensor
        self.speaker_model = speaker_model
        
        self.logger.info(
            f"开始对话智能处理 (V3.4三阶段): {len(sv_results)} 个SV片段, {len(asr_results)} 个ASR词"
        )
        
        # 阶段一：初步对齐（可能会有边界错误）
        initial_aligned_words = self._align_words_to_speakers(asr_results, sv_results)
        
        # 阶段二：边界校正（修正边界附近的归属错误）
        corrected_aligned_words = self._boundary_correction_pass(initial_aligned_words, sv_results, log_path=log_path)

        # 阶段二补充：低重叠词声纹复核（结构性修正）
        rechecked_aligned_words = self._recheck_low_overlap_words(corrected_aligned_words, sv_results, log_path=log_path)
        
        # 阶段三：词级声纹覆盖（短词强证据修正）
        voiceprint_aligned_words = self._override_by_voiceprint(rechecked_aligned_words, sv_results, log_path=log_path)

        # 阶段四：平滑修正（修复A->B->A的短时跳变错误）
        smoothed_aligned_words = self._apply_smoothing(voiceprint_aligned_words, log_path=log_path)
        
        # 步骤四：置信度感知的聚合
        final_transcript = self._aggregate_with_confidence(smoothed_aligned_words)
        
        self.logger.info(f"对话智能处理完成: 生成 {len(final_transcript)} 个对话片段")
        
        # 追加详细日志到现有日志文件
        if log_path:
            self._append_diarization_log(voiceprint_aligned_words, final_transcript, log_path)
        
        # 清理动态资源引用
        self.audio_tensor = None
        self.speaker_model = None
        
        return final_transcript
    
    def _align_words_to_speakers(
        self,
        asr_results: List[Dict[str, Any]],
        sv_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """步骤1: 词-说话人对齐 (V3.1优化版)
        
        对于每个ASR词，找到时间上重叠最多的SV片段，将说话人身份赋予该词。
        V3.1新增：对于落在间隙中的词（overlap=0），启动三把安全锁的仲裁流程。
        
        Args:
            asr_results: ASR词列表
            sv_results: SV片段列表
        
        Returns:
            带说话人标签的词列表
                格式: [{word, start, end, speaker, z_score, confidence}, ...]
        """
        aligned_words = []
        
        for idx, word_info in enumerate(asr_results):
            word = word_info['word']
            word_start = word_info['start']
            word_end = word_info['end']
            asr_confidence = word_info.get('confidence', 1.0)
            
            # 跳过空词
            if not word or word.strip() == '':
                continue
            
            # 查找重叠最多的SV片段
            result = self._find_speaker_for_timestamp(
                word_start, word_end, sv_results
            )
            
            best_speaker = result['speaker']
            best_z_score = result['z_score']
            max_overlap = result['overlap']
            best_embedding = result.get('embedding', None)
            status = result.get('status', 'found')
            decision_level = result.get('decision_level', 'unknown')
            iou = result.get('iou', 0.0)
            
            # V3.4优化：使用IoU/Center/Score三级决策逻辑
            # 这里只做初步对齐，边界校正阶段会进一步优化
            
            aligned_words.append({
                'word': word.strip(),
                'start': word_start,
                'end': word_end,
                'speaker': best_speaker,
                'z_score': best_z_score,
                'asr_confidence': asr_confidence,
                'overlap_duration': max_overlap,
                'embedding': best_embedding,
                'decision_level': decision_level,
                'iou': iou
            })
            
            self.logger.debug(
                f"词对齐: '{word}' [{word_start:.2f}-{word_end:.2f}s] -> "
                f"{best_speaker} (z={best_z_score:.2f}, iou={iou:.3f}, level={decision_level})"
            )
        
        self.logger.info(f"词-说话人对齐完成: {len(aligned_words)} 个词已标注")
        
        return aligned_words
    
    def _boundary_correction_pass(
        self,
        aligned_words: List[Dict[str, Any]],
        sv_results: List[Dict[str, Any]],
        log_path: str = None
    ) -> List[Dict[str, Any]]:
        """V3.3 阶段二：边界校正（基于SV片段的嫌疑区定义）
        
        专门审查和修正说话人切换边界附近的词的归属。
        
        核心思想：
        1. 识别所有说话人切换边界
        2. 基于SV片段结束时间定义嫌疑区（VAD滞后污染区）
        3. 对嫌疑区内的词进行重新评估，基于时间位置决策
        
        Args:
            aligned_words: 初步对齐后的词列表
            sv_results: SV片段列表（用于确定嫌疑区范围）
            log_path: 日志文件路径（用于追加详细诊断日志）
        
        Returns:
            经过边界校正的词列表
        """
        if not self.boundary_correction_enabled or not self.vector_storage:
            return aligned_words
        
        if not aligned_words:
            return aligned_words
        
        # 准备详细诊断日志
        diagnostic_log = []
        diagnostic_log.append("\n" + "="*80)
        diagnostic_log.append("边界校正诊断日志 (Boundary Correction Diagnostic)")
        diagnostic_log.append("="*80)
        
        self.logger.info("开始边界校正...")
        diagnostic_log.append("\n[阶段1] 识别说话人切换边界")
        
        # 步骤1：识别说话人切换边界
        boundaries = []
        for i in range(len(aligned_words) - 1):
            current_speaker = aligned_words[i]['speaker']
            next_speaker = aligned_words[i + 1]['speaker']
            
            # 发现说话人切换
            if current_speaker != next_speaker and current_speaker != 'unknown' and next_speaker != 'unknown':
                boundary_time = aligned_words[i + 1]['start']
                boundaries.append({
                    'index': i + 1,
                    'time': boundary_time,
                    'prev_speaker': current_speaker,
                    'next_speaker': next_speaker
                })
        
        self.logger.info(f"发现 {len(boundaries)} 个说话人切换边界")
        diagnostic_log.append(f"✓ 发现 {len(boundaries)} 个说话人切换边界")
        
        if not boundaries:
            diagnostic_log.append("✗ 无边界，跳过校正")
            if log_path:
                self._append_diagnostic_log(log_path, diagnostic_log)
            return aligned_words
        
        # 列出所有边界
        for idx, boundary in enumerate(boundaries):
            diagnostic_log.append(
                f"  边界{idx+1}: {boundary['prev_speaker']} -> {boundary['next_speaker']} "
                f"@ {boundary['time']:.2f}s (index={boundary['index']})"
            )
        
        # 步骤2：对每个边界进行校正
        corrected_words = aligned_words.copy()
        total_corrections = 0
        
        diagnostic_log.append(f"\n[阶段2] 边界校正处理")
        
        for boundary_idx, boundary in enumerate(boundaries):
            boundary_time = boundary['time']
            prev_speaker = boundary['prev_speaker']
            next_speaker = boundary['next_speaker']
            
            # 查找prev_speaker的SV片段结束时间和next_speaker的SV片段开始时间
            prev_sv_end = None
            next_sv_start = None
            
            for sv_seg in sv_results:
                sv_speaker = sv_seg.get('speaker', sv_seg.get('user_id', 'unknown'))
                sv_start = sv_seg['start']
                sv_end = sv_seg['end']
                
                # 找到在边界前结束的、属于prev_speaker的最后一个SV片段
                if sv_speaker == prev_speaker and sv_end <= boundary_time:
                    if prev_sv_end is None or sv_end > prev_sv_end:
                        prev_sv_end = sv_end
                
                # 找到在边界后开始的、属于next_speaker的第一个SV片段
                if sv_speaker == next_speaker and sv_start >= boundary_time:
                    if next_sv_start is None or sv_start < next_sv_start:
                        next_sv_start = sv_start
            
            # 定义嫌疑区：VAD滞后导致prev_speaker的SV片段延伸到next_speaker的区域
            # 嫌疑区 = [max(SV片段结束 - window, 边界 - window), SV片段结束]
            # 检查SV片段结束前window_s时间内的词
            if prev_sv_end is not None:
                # 嫌疑区从SV片段结束前window_s开始，到SV片段结束
                # 使用配置的窗口大小，不限制最小值
                suspicion_zone_start = prev_sv_end - self.boundary_correction_window_s
                suspicion_zone_end = prev_sv_end
            else:
                # 如果找不到SV片段，使用原来的逻辑
                suspicion_zone_end = boundary_time
                suspicion_zone_start = boundary_time - self.boundary_correction_window_s
            
            diagnostic_log.append(f"\n--- 边界 {boundary_idx+1}/{len(boundaries)} ---")
            diagnostic_log.append(f"边界位置: {prev_speaker} -> {next_speaker} @ {boundary_time:.2f}s")
            if prev_sv_end is not None:
                diagnostic_log.append(f"SV片段结束: {prev_sv_end:.2f}s (prev_speaker={prev_speaker})")
                if next_sv_start is not None:
                    diagnostic_log.append(f"下一SV开始: {next_sv_start:.2f}s (next_speaker={next_speaker})")
                diagnostic_log.append(f"嫌疑区范围: [{suspicion_zone_start:.2f}s - {suspicion_zone_end:.2f}s] (SV片段结束前{self.boundary_correction_window_s}s)")
            else:
                diagnostic_log.append(f"嫌疑区范围: [{suspicion_zone_start:.2f}s - {suspicion_zone_end:.2f}s] (边界前{self.boundary_correction_window_s}s)")
            
            self.logger.debug(
                f"处理边界 {prev_speaker}->{next_speaker} @ {boundary_time:.2f}s, "
                f"嫌疑区: [{suspicion_zone_start:.2f}s - {boundary_time:.2f}s]"
            )
            
            # 获取两个候选说话人的注册模板
            prev_template = self._get_speaker_template(prev_speaker)
            next_template = self._get_speaker_template(next_speaker)
            
            if prev_template is None or next_template is None:
                diagnostic_log.append(f"✗ 无法获取模板 (prev_template={'存在' if prev_template is not None else '缺失'}, next_template={'存在' if next_template is not None else '缺失'})")
                self.logger.debug(f"无法获取模板，跳过此边界")
                continue
            
            diagnostic_log.append(f"✓ 成功加载候选人模板:")
            prev_shape = getattr(prev_template, 'shape', f'list len={len(prev_template)}' if isinstance(prev_template, list) else 'unknown')
            next_shape = getattr(next_template, 'shape', f'list len={len(next_template)}' if isinstance(next_template, list) else 'unknown')
            diagnostic_log.append(f"  - {prev_speaker}: embedding shape = {prev_shape}")
            diagnostic_log.append(f"  - {next_speaker}: embedding shape = {next_shape}")
            
            # 收集嫌疑区内的词（使用时间重叠判断）
            suspicious_words_info = []
            for i, word_info in enumerate(corrected_words):
                word_start = word_info['start']
                word_end = word_info['end']
                word_speaker = word_info['speaker']
                
                # 检查词与嫌疑区是否有时间重叠
                # 重叠条件：词的开始时间 < 嫌疑区结束 AND 词的结束时间 > 嫌疑区开始
                has_overlap = (word_start < suspicion_zone_end and word_end > suspicion_zone_start)
                
                if has_overlap and word_speaker == prev_speaker:
                    suspicious_words_info.append((i, word_info))
            
            diagnostic_log.append(f"\n嫌疑区内的词: {len(suspicious_words_info)} 个")
            
            if not suspicious_words_info:
                diagnostic_log.append("  (无词需要重新评估)")
                continue
            
            # ========== 批量推理优化：收集所有需要提取特征的词 ==========
            word_audios = []
            word_indices = []
            word_metadata = []
            
            for word_idx, (i, word_info) in enumerate(suspicious_words_info):
                word_duration = word_info['end'] - word_info['start']
                
                # 时长过滤（<0.2s的词跳过）
                if word_duration >= 0.2 and self.audio_tensor is not None:
                    try:
                        sr = 16000
                        start_idx = int(word_info['start'] * sr)
                        end_idx = int(word_info['end'] * sr)
                        
                        # 边界保护
                        start_idx = max(0, start_idx)
                        end_idx = min(len(self.audio_tensor), end_idx)
                        
                        if start_idx < end_idx:
                            word_audio = self.audio_tensor[start_idx:end_idx]
                            word_audios.append(word_audio)
                            word_indices.append(word_idx)
                            word_metadata.append((i, word_info))
                    except Exception as e:
                        self.logger.warning(f"词 '{word_info.get('word')}' 音频提取失败: {e}")
            
            # ========== 批量提取词级别特征 ==========
            embedding_map = {}
            if word_audios and self.speaker_model is not None:
                self.logger.info(f"🚀 批量提取词级别特征: {len(word_audios)} 个词")
                try:
                    word_embeddings = self.speaker_model.extract_batch_embeddings(
                        word_audios,
                        batch_size=self.config.speaker_batch_size
                    )
                    
                    # 将embedding映射回对应的词
                    for idx, emb in zip(word_indices, word_embeddings):
                        embedding_map[idx] = emb
                        
                except Exception as e:
                    self.logger.error(f"批量特征提取失败: {e}")
            
            # 步骤3：重新评估嫌疑区内的词
            # V3.3混合决策策略：声纹相似度优先 + 时间位置兜底
            for word_idx, (i, word_info) in enumerate(suspicious_words_info):
                word = word_info['word']
                word_start = word_info['start']
                word_end = word_info['end']
                word_speaker = word_info['speaker']
                word_duration = word_end - word_start
                
                diagnostic_log.append(f"\n  [{word_idx+1}/{len(suspicious_words_info)}] 重新评估词: '{word}'")
                diagnostic_log.append(f"      时间: [{word_start:.2f}s - {word_end:.2f}s] (时长={word_duration:.2f}s)")
                diagnostic_log.append(f"      原归属: {word_speaker}")
                
                # 计算词在嫌疑区中的相对位置
                distance_from_zone_start = word_start - suspicion_zone_start
                zone_width = suspicion_zone_end - suspicion_zone_start
                position_in_zone = distance_from_zone_start / zone_width if zone_width > 0 else 0
                distance_to_zone_end = suspicion_zone_end - word_start
                
                diagnostic_log.append(f"      嫌疑区位置: {position_in_zone:.2%} (0%=起点, 100%=SV片段结束)")
                diagnostic_log.append(f"      到SV结束距离: {distance_to_zone_end:.2f}s")
                
                # 策略：分段动态梯度阈值（位置感知的置信度要求）
                # 核心思想：
                # 1. 区外（position < 0）：使用固定高阈值，严格保护
                # 2. 区内（position >= 0）：使用动态梯度，离边界越远要求越高
                # 
                # 公式：
                # - position < 0: required_diff = out_of_zone_threshold (固定高阈值)
                # - position >= 0: required_diff = base_diff + penalty_factor * (1 - position)
                # 
                # 参数说明（从配置文件读取）：
                # - base_diff: 基础阈值（边界处，position=1.0时的最低要求）
                # - penalty_factor: 惩罚因子（控制梯度陡峭程度）
                # - out_of_zone_threshold: 区外固定阈值
                
                # 从配置读取参数
                base_diff = self.config.boundary_correction_base_diff
                penalty_factor = self.config.boundary_correction_penalty_factor
                out_of_zone_threshold = self.config.boundary_correction_out_of_zone_threshold
                
                # 计算动态阈值（分段）
                if position_in_zone < 0:
                    # 区外：使用固定高阈值，严格保护正常词
                    required_diff = out_of_zone_threshold
                else:
                    # 区内：使用动态梯度
                    required_diff = base_diff + penalty_factor * (1.0 - position_in_zone)
                
                # 获取词级别特征（从批量提取的结果中获取）
                word_embedding = embedding_map.get(word_idx)
                
                should_correct = False
                correction_reason = ""
                
                if word_embedding is not None and prev_template is not None and next_template is not None:
                    # 情况A：有声纹（长词）- 动态梯度决策
                    score_prev = self._calculate_cosine_similarity(word_embedding, prev_template)
                    score_next = self._calculate_cosine_similarity(word_embedding, next_template)
                    actual_diff = score_next - score_prev
                    
                    diagnostic_log.append(f"      声纹相似度: prev={score_prev:.3f}, next={score_next:.3f}")
                    diagnostic_log.append(f"      动态阈值: required_diff={required_diff:.3f} (base={base_diff:.2f} + penalty={penalty_factor:.2f}*(1-{position_in_zone:.2f}))")
                    
                    # 判断：实际差异是否超过动态阈值
                    if actual_diff > required_diff:
                        should_correct = True
                        correction_reason = f"声纹梯度 (diff={actual_diff:.3f} > required={required_diff:.3f}, pos={position_in_zone:.2%})"
                    else:
                        should_correct = False
                        correction_reason = f"声纹不足 (diff={actual_diff:.3f} <= required={required_diff:.3f}, pos={position_in_zone:.2%})"
                else:
                    # 情况B：无声纹（短词）- 保守决策（只切最后20%区域）
                    fallback_threshold = 0.8
                    if position_in_zone > fallback_threshold:
                        should_correct = True
                        correction_reason = f"短词兜底 (位置={position_in_zone:.2%} > {fallback_threshold:.2%}，极度靠近边界)"
                    else:
                        should_correct = False
                        correction_reason = f"短词保持 (位置={position_in_zone:.2%} <= {fallback_threshold:.2%}，未达兜底阈值)"
                
                # 执行决策
                if should_correct:
                    diagnostic_log.append(f"      ✓ 决策: 修正归属 {prev_speaker} -> {next_speaker}")
                    diagnostic_log.append(f"         理由: {correction_reason}")
                    
                    self.logger.debug(
                        f"修正词 '{word}' @ {word_start:.2f}s: "
                        f"{prev_speaker} -> {next_speaker} ({correction_reason})"
                    )
                    
                    corrected_words[i]['speaker'] = next_speaker
                    corrected_words[i]['correction'] = 'boundary_corrected_voiceprint' if word_embedding is not None else 'boundary_corrected_time'
                    corrected_words[i]['original_speaker'] = prev_speaker
                    corrected_words[i]['position_in_zone'] = position_in_zone
                    corrected_words[i]['distance_to_zone_end'] = distance_to_zone_end
                    total_corrections += 1
                else:
                    diagnostic_log.append(f"      ✗ 决策: 保持原归属 {prev_speaker}")
                    diagnostic_log.append(f"         理由: {correction_reason}")
        
        diagnostic_log.append(f"\n[阶段3] 校正总结")
        diagnostic_log.append(f"✓ 边界校正完成: 修正了 {total_corrections} 个词的归属")
        diagnostic_log.append("="*80 + "\n")
        
        self.logger.info(f"边界校正完成: 修正了 {total_corrections} 个词的归属")
        
        # 追加诊断日志到文件
        if log_path:
            self._append_diagnostic_log(log_path, diagnostic_log)
        
        return corrected_words
    
    def _apply_smoothing(
        self,
        words_with_speaker: List[Dict[str, Any]],
        log_path: str = None
    ) -> List[Dict[str, Any]]:
        """阶段三：平滑修正（三明治法则）
        
        修复 A -> B -> A 的短时跳变错误。
        
        核心原则：
        1. 只看时长，不看Z-Score（避免"虚高"误导）
        2. 使用保守的时长阈值（从配置读取，默认0.8秒）
        3. 🛡️ 保护边界校正的决策（已被V3.3修正的词不再修改）
        4. 记录详细日志，便于调优
        
        Args:
            words_with_speaker: 带说话人标签的词列表
            log_path: 日志文件路径（用于追加诊断日志）
        
        Returns:
            平滑修正后的词列表
        """
        if len(words_with_speaker) < 3:
            return words_with_speaker
        
        # 从配置读取时长阈值
        duration_threshold = self.config.__dict__.get('smoothing_duration_threshold', 0.8)
        
        # 准备诊断日志
        diagnostic_log = []
        diagnostic_log.append("\n" + "="*80)
        diagnostic_log.append("平滑修正诊断日志 (Smoothing Correction Diagnostic)")
        diagnostic_log.append("="*80)
        diagnostic_log.append(f"时长阈值: {duration_threshold}s")
        diagnostic_log.append("")
        
        smoothed_count = 0
        smoothed_words = words_with_speaker.copy()
        
        for i in range(1, len(smoothed_words) - 1):
            prev_word = smoothed_words[i-1]
            curr_word = smoothed_words[i]
            next_word = smoothed_words[i+1]
            
            # 安全检查
            if not all('speaker' in w for w in [prev_word, curr_word, next_word]):
                continue
            if not all(k in curr_word for k in ['start', 'end', 'text']):
                # 兼容'word'字段
                if 'word' not in curr_word:
                    continue
            
            speaker_prev = prev_word['speaker']
            speaker_curr = curr_word['speaker']
            speaker_next = next_word['speaker']
            
            # 🛡️【关键保护逻辑】🛡️
            # 如果当前词已经被前面的 V3.3 边界校正逻辑修正过（correction='boundary_corrected_voiceprint'或'boundary_corrected_time'），
            # 说明它是基于声纹证据的"高智商决策"，平滑逻辑不要覆盖它。
            # 这确保了决策优先级：高智商决策（基于声纹）> 兜底逻辑（基于时长）
            if curr_word.get('correction') in ['boundary_corrected_voiceprint', 'boundary_corrected_time', 'low_overlap_voiceprint', 'voiceprint_override']:
                word_text = curr_word.get('text', curr_word.get('word', ''))
                diagnostic_log.append(f"🛡️ 保护已修正词: '{word_text}' (原因={curr_word.get('correction')})")
                continue
            
            # 检查夹心结构: A -> B -> A
            if speaker_prev == speaker_next and speaker_curr != speaker_prev:
                # 跳过unknown说话人
                if speaker_prev == 'unknown' or speaker_curr == 'unknown':
                    continue
                
                duration = curr_word['end'] - curr_word['start']
                word_text = curr_word.get('text', curr_word.get('word', ''))
                
                # 核心判断：只看时长
                if duration < duration_threshold:
                    old_speaker = speaker_curr
                    old_z_score = curr_word.get('z_score', 'N/A')
                    
                    # 记录诊断信息
                    diagnostic_log.append(f"🔧 平滑修正 [{smoothed_count + 1}]:")
                    diagnostic_log.append(f"   词: '{word_text}' [{curr_word['start']:.2f}s - {curr_word['end']:.2f}s]")
                    diagnostic_log.append(f"   时长: {duration:.2f}s < {duration_threshold}s")
                    diagnostic_log.append(f"   模式: {speaker_prev} -> {old_speaker} -> {speaker_next} (A-B-A夹心)")
                    diagnostic_log.append(f"   原Z-Score: {old_z_score}")
                    diagnostic_log.append(f"   决策: 修正为 {speaker_prev}")
                    diagnostic_log.append("")
                    
                    # 执行修正
                    smoothed_words[i]['speaker'] = speaker_prev
                    smoothed_words[i]['corrected'] = True
                    smoothed_words[i]['correction_reason'] = 'smoothing_sandwich'
                    smoothed_words[i]['original_speaker'] = old_speaker
                    smoothed_words[i]['smoothing_duration'] = duration
                    
                    self.logger.info(
                        f"🔧 平滑修正: '{word_text}' "
                        f"[{old_speaker} -> {speaker_prev}] "
                        f"(时长={duration:.2f}s, z_score={old_z_score}, "
                        f"原因=A-B-A夹心结构)"
                    )
                    smoothed_count += 1
        
        diagnostic_log.append("="*80)
        diagnostic_log.append(f"✅ 平滑修正完成: 修正了 {smoothed_count} 个词")
        diagnostic_log.append("="*80 + "\n")
        
        if smoothed_count > 0:
            self.logger.info(f"✅ 平滑修正完成: 修正了 {smoothed_count} 个词")
        
        # 追加诊断日志到文件
        if log_path:
            self._append_diagnostic_log(log_path, diagnostic_log)
        
        return smoothed_words

    def _override_by_voiceprint(
        self,
        aligned_words: List[Dict[str, Any]],
        sv_results: List[Dict[str, Any]],
        log_path: str = None
    ) -> List[Dict[str, Any]]:
        """对短词进行声纹强证据覆盖

        当词级声纹与当前说话人模板差异显著时，允许覆盖当前归属。
        该过程通过配置阈值控制，避免硬编码。
        """
        if not aligned_words or not self.vector_storage:
            return aligned_words

        if self.audio_tensor is None or self.speaker_model is None:
            return aligned_words

        # 配置参数
        cfg = self.config.__dict__
        min_diff = cfg.get('voiceprint_override_min_diff', 0.25)
        min_duration = cfg.get('voiceprint_override_min_duration', 0.2)
        max_duration = cfg.get('voiceprint_override_max_duration', 0.8)

        # 预加载候选说话人模板
        speaker_ids = {
            seg.get('speaker', seg.get('user_id', 'unknown')) for seg in sv_results
        }
        speaker_ids.discard('unknown')

        templates = {}
        for speaker_id in speaker_ids:
            tmpl = self._get_speaker_template(speaker_id)
            if tmpl is not None:
                templates[speaker_id] = tmpl

        if not templates:
            return aligned_words

        # 选取候选短词（且未被修正）
        candidates = []
        for i, word_info in enumerate(aligned_words):
            if word_info.get('correction') in ['boundary_corrected_voiceprint', 'boundary_corrected_time', 'low_overlap_voiceprint']:
                continue

            duration = word_info['end'] - word_info['start']
            if duration < min_duration or duration > max_duration:
                continue

            if word_info.get('speaker') == 'unknown':
                continue

            candidates.append((i, word_info))

        if not candidates:
            return aligned_words

        # 批量提取词级声纹特征
        word_audios = []
        word_indices = []
        for idx, word_info in candidates:
            try:
                sr = 16000
                start_idx = int(word_info['start'] * sr)
                end_idx = int(word_info['end'] * sr)
                start_idx = max(0, start_idx)
                end_idx = min(len(self.audio_tensor), end_idx)
                if start_idx < end_idx:
                    word_audios.append(self.audio_tensor[start_idx:end_idx])
                    word_indices.append(idx)
            except Exception as e:
                self.logger.debug(f"词 '{word_info.get('word')}' 覆盖音频提取失败: {e}")

        if not word_audios:
            return aligned_words

        embedding_map = {}
        try:
            embeddings = self.speaker_model.extract_batch_embeddings(
                word_audios,
                batch_size=self.config.speaker_batch_size
            )
            for idx, emb in zip(word_indices, embeddings):
                embedding_map[idx] = emb
        except Exception as e:
            self.logger.warning(f"短词声纹覆盖特征提取失败: {e}")
            return aligned_words

        corrected = aligned_words.copy()
        diagnostic_log = []
        diagnostic_log.append("\n" + "="*80)
        diagnostic_log.append("短词声纹覆盖诊断日志 (Voiceprint Override Diagnostic)")
        diagnostic_log.append("="*80)

        correction_count = 0
        for idx, word_info in candidates:
            embedding = embedding_map.get(idx)
            if embedding is None:
                continue

            current_speaker = word_info.get('speaker')
            current_template = templates.get(current_speaker)
            if current_template is None:
                continue

            scores = {}
            for speaker_id, tmpl in templates.items():
                scores[speaker_id] = self._calculate_cosine_similarity(embedding, tmpl)

            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]
            current_score = scores.get(current_speaker, -1.0)
            diff = best_score - current_score

            if best_speaker != current_speaker and diff >= min_diff:
                corrected[idx]['speaker'] = best_speaker
                corrected[idx]['correction'] = 'voiceprint_override'
                corrected[idx]['original_speaker'] = current_speaker
                corrected[idx]['similarity_diff'] = diff
                correction_count += 1

                diagnostic_log.append(
                    f"✓ 覆盖 '{word_info.get('word')}' [{word_info['start']:.2f}-{word_info['end']:.2f}s] "
                    f"{current_speaker} -> {best_speaker} (diff={diff:.3f})"
                )
            else:
                diagnostic_log.append(
                    f"✗ 保持 '{word_info.get('word')}' [{word_info['start']:.2f}-{word_info['end']:.2f}s] "
                    f"{current_speaker} (best={best_speaker}, diff={diff:.3f})"
                )

        diagnostic_log.append(f"覆盖完成：修正 {correction_count} 个词")
        diagnostic_log.append("="*80 + "\n")

        if log_path:
            self._append_diagnostic_log(log_path, diagnostic_log)

        return corrected

    def _recheck_low_overlap_words(
        self,
        aligned_words: List[Dict[str, Any]],
        sv_results: List[Dict[str, Any]],
        log_path: str = None
    ) -> List[Dict[str, Any]]:
        """低重叠词声纹复核

        对与SV片段重叠比例过低的词，使用词级声纹与注册模板复核归属，
        仅在相似度差值显著时才调整归属。
        """
        if not aligned_words or not self.vector_storage:
            return aligned_words

        if self.audio_tensor is None or self.speaker_model is None:
            return aligned_words

        min_overlap_ratio = self.min_overlap_ratio
        min_diff = self.min_similarity_for_alignment

        # 预加载候选说话人模板
        speaker_ids = {
            seg.get('speaker', seg.get('user_id', 'unknown')) for seg in sv_results
        }
        speaker_ids.discard('unknown')

        templates = {}
        for speaker_id in speaker_ids:
            tmpl = self._get_speaker_template(speaker_id)
            if tmpl is not None:
                templates[speaker_id] = tmpl

        if not templates:
            return aligned_words

        # 选出需要复核的词
        candidates = []
        for i, word_info in enumerate(aligned_words):
            duration = word_info['end'] - word_info['start']
            if duration <= 0:
                continue

            overlap_duration = word_info.get('overlap_duration') or 0.0
            overlap_ratio = overlap_duration / duration if duration > 0 else 0.0
            if overlap_ratio >= min_overlap_ratio:
                continue

            # 时长过短的词跳过（与特征提取一致）
            if duration < 0.2:
                continue

            candidates.append((i, word_info))

        if not candidates:
            return aligned_words

        # 批量提取词级声纹特征
        word_audios = []
        word_indices = []
        for idx, word_info in candidates:
            try:
                sr = 16000
                start_idx = int(word_info['start'] * sr)
                end_idx = int(word_info['end'] * sr)
                start_idx = max(0, start_idx)
                end_idx = min(len(self.audio_tensor), end_idx)
                if start_idx < end_idx:
                    word_audios.append(self.audio_tensor[start_idx:end_idx])
                    word_indices.append(idx)
            except Exception as e:
                self.logger.debug(f"词 '{word_info.get('word')}' 复核音频提取失败: {e}")

        if not word_audios:
            return aligned_words

        embedding_map = {}
        try:
            embeddings = self.speaker_model.extract_batch_embeddings(
                word_audios,
                batch_size=self.config.speaker_batch_size
            )
            for idx, emb in zip(word_indices, embeddings):
                embedding_map[idx] = emb
        except Exception as e:
            self.logger.warning(f"低重叠词特征提取失败: {e}")
            return aligned_words

        # 复核并修正
        corrected = aligned_words.copy()
        diagnostic_log = []
        diagnostic_log.append("\n" + "="*80)
        diagnostic_log.append("低重叠词复核诊断日志 (Low Overlap Recheck Diagnostic)")
        diagnostic_log.append("="*80)

        correction_count = 0
        for idx, word_info in candidates:
            embedding = embedding_map.get(idx)
            if embedding is None:
                continue

            current_speaker = word_info.get('speaker')
            if current_speaker == 'unknown':
                continue

            current_template = templates.get(current_speaker)
            if current_template is None:
                continue

            # 计算与所有模板的相似度
            scores = {}
            for speaker_id, tmpl in templates.items():
                scores[speaker_id] = self._calculate_cosine_similarity(embedding, tmpl)

            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]
            current_score = scores.get(current_speaker, -1.0)

            if best_speaker != current_speaker and (best_score - current_score) >= min_diff:
                corrected[idx]['speaker'] = best_speaker
                corrected[idx]['correction'] = 'low_overlap_voiceprint'
                corrected[idx]['original_speaker'] = current_speaker
                corrected[idx]['similarity_diff'] = best_score - current_score
                correction_count += 1

                diagnostic_log.append(
                    f"✓ 修正 '{word_info.get('word')}' [{word_info['start']:.2f}-{word_info['end']:.2f}s] "
                    f"{current_speaker} -> {best_speaker} (diff={best_score - current_score:.3f})"
                )
            else:
                diagnostic_log.append(
                    f"✗ 保持 '{word_info.get('word')}' [{word_info['start']:.2f}-{word_info['end']:.2f}s] "
                    f"{current_speaker} (best={best_speaker}, diff={best_score - current_score:.3f})"
                )

        diagnostic_log.append(f"复核完成：修正 {correction_count} 个词")
        diagnostic_log.append("="*80 + "\n")

        if log_path:
            self._append_diagnostic_log(log_path, diagnostic_log)

        return corrected
    
    def _append_diagnostic_log(self, log_path: str, diagnostic_log: List[str]) -> None:
        """追加诊断日志到识别日志文件
        
        Args:
            log_path: 日志文件路径
            diagnostic_log: 诊断日志行列表
        """
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                for line in diagnostic_log:
                    f.write(line + '\n')
        except Exception as e:
            self.logger.warning(f"追加诊断日志失败: {e}")
    
    def _get_speaker_template(self, speaker_id: str) -> Optional[np.ndarray]:
        """获取说话人的注册模板
        
        Args:
            speaker_id: 说话人ID
        
        Returns:
            注册模板的embedding，如果获取失败则返回None
        """
        if not self.vector_storage or speaker_id == 'unknown':
            return None
        
        try:
            user_data = self.vector_storage.get_user(speaker_id)
            if user_data is None:
                return None
            return user_data.get('embedding')
        except Exception as e:
            self.logger.warning(f"获取说话人 {speaker_id} 的模板失败: {e}")
            return None
    
    def _extract_word_embedding(self, word: Dict[str, Any]) -> Optional[np.ndarray]:
        """按需提取词级别声纹特征
        
        Args:
            word: 词信息字典，包含start、end、word等字段
        
        Returns:
            词的声纹特征向量，如果提取失败则返回None
        """
        if self.audio_tensor is None or self.speaker_model is None:
            return None
        
        # 时长过滤（<0.2s的词跳过）
        duration = word['end'] - word['start']
        if duration < 0.2:
            self.logger.debug(f"词 '{word.get('word')}' 时长过短 ({duration:.2f}s)，跳过特征提取")
            return None
        
        try:
            sr = 16000
            start_idx = int(word['start'] * sr)
            end_idx = int(word['end'] * sr)
            
            # 边界保护
            start_idx = max(0, start_idx)
            end_idx = min(len(self.audio_tensor), end_idx)
            
            if start_idx >= end_idx:
                return None
            
            word_audio = self.audio_tensor[start_idx:end_idx]
            
            # 调用模型提取特征
            embedding = self.speaker_model.extract_embedding(word_audio)
            
            return embedding
            
        except Exception as e:
            self.logger.warning(f"词 '{word.get('word')}' 特征提取失败: {e}")
            return None
    
    def _aggregate_with_confidence(
        self,
        aligned_words: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """步骤2: 置信度感知的聚合
        
        将连续的、相同说话人的词语合并成句子。
        使用锚点思想：低置信度的词，如果前后都是高置信度的同一说话人，则修正其身份。
        
        Args:
            aligned_words: 带说话人标签的词列表
        
        Returns:
            高质量对话流列表
        """
        if not aligned_words:
            return []
        
        # 应用锚点修正
        corrected_words = self._apply_anchor_correction(aligned_words)
        
        # 合并连续的同说话人词语
        final_transcript = []
        current_segment = None
        
        for word_info in corrected_words:
            speaker = word_info['speaker']
            
            # 如果是新的说话人，或者是第一个词
            if current_segment is None or current_segment['speaker'] != speaker:
                # 保存上一个片段
                if current_segment is not None:
                    final_transcript.append(current_segment)
                
                # 开始新片段
                current_segment = {
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'speaker': speaker,
                    'transcript': word_info['word'],
                    'z_scores': [word_info['z_score']],
                    'word_count': 1
                }
            else:
                # 继续当前片段
                current_segment['end'] = word_info['end']
                current_segment['transcript'] += word_info['word']
                current_segment['z_scores'].append(word_info['z_score'])
                current_segment['word_count'] += 1
        
        # 保存最后一个片段
        if current_segment is not None:
            final_transcript.append(current_segment)
        
        # 计算平均Z-score
        for segment in final_transcript:
            z_scores = segment.pop('z_scores')
            # 过滤掉0值（unknown的z_score）
            valid_z_scores = [z for z in z_scores if z > 0]
            if valid_z_scores:
                segment['avg_z_score'] = round(np.mean(valid_z_scores), 2)
            else:
                segment['avg_z_score'] = 0.0
            
            segment['duration'] = round(segment['end'] - segment['start'], 2)
        
        self.logger.info(f"聚合完成: {len(final_transcript)} 个对话片段")
        
        return final_transcript
    
    def _find_speaker_for_timestamp(
        self,
        word_start: float,
        word_end: float,
        sv_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """查找时间戳对应的说话人（V3.4优化：IoU/Center/Score三级决策）
        
        V3.4核心优化：解决词跨段问题（一个词一半在片段A，一半在片段B）
        - 步骤1（IoU规则）：计算词与SV片段的IoU，选择IoU最大的片段
        - 步骤2（Center规则）：如果IoU相同，选择包含词中心点的片段
        - 步骤3（Score规则）：如果都不满足，选择Z-score最高的片段
        
        三级决策逻辑确保在各种边缘情况下都能做出合理的归属决策。
        
        Args:
            word_start: 词开始时间
            word_end: 词结束时间
            sv_results: SV片段列表
        
        Returns:
            结果字典: {speaker, z_score, overlap, embedding, status, decision_level, iou}
        """
        if not sv_results:
            return {
                'speaker': 'unknown',
                'z_score': 0.0,
                'overlap': 0.0,
                'embedding': None,
                'status': 'found',
                'decision_level': 'no_segments',
                'iou': 0.0
            }
        
        word_duration = word_end - word_start
        word_center = (word_start + word_end) / 2
        
        # 计算每个SV片段与词的IoU
        candidates = []
        for sv_seg in sv_results:
            sv_start = sv_seg['start']
            sv_end = sv_seg['end']
            
            # 计算交集（Intersection）
            intersection_start = max(word_start, sv_start)
            intersection_end = min(word_end, sv_end)
            intersection = max(0, intersection_end - intersection_start)
            
            # 计算并集（Union）
            union_start = min(word_start, sv_start)
            union_end = max(word_end, sv_end)
            union = union_end - union_start
            
            # 计算IoU
            iou = intersection / union if union > 0 else 0.0
            
            # 检查词中心点是否在SV片段内
            center_in_segment = (sv_start <= word_center <= sv_end)
            
            candidates.append({
                'sv_seg': sv_seg,
                'iou': iou,
                'intersection': intersection,
                'center_in_segment': center_in_segment,
                'speaker': sv_seg.get('speaker', sv_seg.get('user_id', 'unknown')),
                'z_score': sv_seg.get('z_score', sv_seg.get('raw_z_score', 0.0)),
                'embedding': sv_seg.get('embedding', None)
            })
        
        # 三级决策逻辑
        
        # 步骤1：IoU规则 - 选择IoU最大的片段
        max_iou = max(c['iou'] for c in candidates)
        iou_candidates = [c for c in candidates if c['iou'] == max_iou]
        
        if len(iou_candidates) == 1:
            # 只有一个最大IoU，直接选择
            best = iou_candidates[0]
            return {
                'speaker': best['speaker'],
                'z_score': best['z_score'],
                'overlap': best['intersection'],
                'embedding': best['embedding'],
                'status': 'found',
                'decision_level': 'iou',  # 通过IoU规则决策
                'iou': best['iou']
            }
        
        # 步骤2：Center规则 - 如果IoU相同，选择包含词中心点的片段
        center_candidates = [c for c in iou_candidates if c['center_in_segment']]
        
        if len(center_candidates) == 1:
            # 只有一个包含词中心点，选择它
            best = center_candidates[0]
            return {
                'speaker': best['speaker'],
                'z_score': best['z_score'],
                'overlap': best['intersection'],
                'embedding': best['embedding'],
                'status': 'found',
                'decision_level': 'center',  # 通过Center规则决策
                'iou': best['iou']
            }
        elif len(center_candidates) > 1:
            # 多个包含词中心点，在这些候选中选择Z-score最高的
            best = max(center_candidates, key=lambda c: c['z_score'])
            return {
                'speaker': best['speaker'],
                'z_score': best['z_score'],
                'overlap': best['intersection'],
                'embedding': best['embedding'],
                'status': 'found',
                'decision_level': 'center+score',  # 通过Center+Score规则决策
                'iou': best['iou']
            }
        
        # 步骤3：Score规则 - 如果都不满足，选择Z-score最高的片段
        best = max(iou_candidates, key=lambda c: c['z_score'])
        return {
            'speaker': best['speaker'],
            'z_score': best['z_score'],
            'overlap': best['intersection'],
            'embedding': best['embedding'],
            'status': 'found',
            'decision_level': 'score',  # 通过Score规则决策
            'iou': best['iou']
        }
    
    def _calculate_cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """计算两个声纹向量的余弦相似度
        
        Args:
            embedding1: 声纹向量1
            embedding2: 声纹向量2
        
        Returns:
            余弦相似度（0-1）
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # 确保是numpy数组
        emb1 = np.array(embedding1).flatten()
        emb2 = np.array(embedding2).flatten()
        
        # 计算余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def _apply_anchor_correction(
        self,
        aligned_words: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """应用锚点修正
        
        对于低置信度的词，如果其前后都是高置信度的同一说话人，则修正其身份。
        
        Args:
            aligned_words: 带说话人标签的词列表
        
        Returns:
            修正后的词列表
        """
        if len(aligned_words) < 3:
            return aligned_words
        
        corrected_words = aligned_words.copy()
        
        for i in range(1, len(corrected_words) - 1):
            current = corrected_words[i]
            prev = corrected_words[i - 1]
            next_word = corrected_words[i + 1]
            
            # 如果当前词是低置信度
            if current['z_score'] < self.low_confidence_threshold:
                # 检查前后是否是同一个高置信度说话人
                if (prev['z_score'] >= self.low_confidence_threshold and
                    next_word['z_score'] >= self.low_confidence_threshold and
                    prev['speaker'] == next_word['speaker'] and
                    prev['speaker'] != 'unknown'):
                    
                    # 修正当前词的说话人
                    old_speaker = current['speaker']
                    new_speaker = prev['speaker']
                    
                    self.logger.debug(
                        f"锚点修正: 词'{current['word']}' "
                        f"{old_speaker} -> {new_speaker} "
                        f"(前后锚点: {prev['speaker']}, z={prev['z_score']:.2f}/{next_word['z_score']:.2f})"
                    )
                    
                    corrected_words[i]['speaker'] = new_speaker
                    corrected_words[i]['corrected_by_anchor'] = True
        
        return corrected_words
    
    def _append_diarization_log(
        self,
        aligned_words: List[Dict[str, Any]],
        final_transcript: List[Dict[str, Any]],
        log_path: str
    ) -> None:
        """追加 Diarization Engine 的详细日志到现有日志文件
        
        Args:
            aligned_words: 对齐后的词列表
            final_transcript: 最终对话流
            log_path: 日志文件路径
        """
        try:
            log_lines = []
            log_lines.append("")
            log_lines.append("=" * 80)
            log_lines.append("DiarizationEngine 处理详情")
            log_lines.append("=" * 80)
            log_lines.append("")
            
            # 词-说话人对齐结果（完整列举）
            log_lines.append(f"📌 词-说话人对齐结果 (共 {len(aligned_words)} 个词):")
            log_lines.append("-" * 80)
            for i, word_info in enumerate(aligned_words):
                log_lines.append(
                    f"  [{i+1}] '{word_info['word']}' [{word_info['start']:.2f}s-{word_info['end']:.2f}s] "
                    f"-> speaker={word_info['speaker']}, z_score={word_info['z_score']:.2f}, "
                    f"overlap={word_info.get('overlap_duration', 0):.2f}s"
                )
            log_lines.append("")
            
            # 最终对话流
            log_lines.append(f"🎯 最终对话流 (共 {len(final_transcript)} 个片段):")
            log_lines.append("-" * 80)
            for i, seg in enumerate(final_transcript):
                log_lines.append(
                    f"  [{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s) | "
                    f"speaker={seg['speaker']}, z_score={seg['avg_z_score']:.2f}, "
                    f"words={seg['word_count']}"
                )
                log_lines.append(f"      文本: {seg.get('transcript', '')}")
            log_lines.append("")
            log_lines.append("=" * 80)
            
            # 追加到文件
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_lines))
            
            self.logger.info(f"Diarization详细日志已追加到: {log_path}")
            
        except Exception as e:
            self.logger.warning(f"追加Diarization日志失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置
        
        Returns:
            配置字典
        """
        return {
            'time_threshold': self.time_threshold,
            'similarity_threshold': self.similarity_threshold,
            'low_confidence_threshold': self.low_confidence_threshold,
            'max_merge_duration': self.max_merge_duration
        }

