#!/usr/bin/env python3
"""
音频文件存储管理
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class AudioStorage:
    """音频存储管理器"""
    
    def __init__(self, storage_dir: Path, metadata_file: Path):
        """
        初始化音频存储管理器
        
        Args:
            storage_dir: 音频文件存储目录
            metadata_file: 元数据文件路径
        """
        self.storage_dir = storage_dir
        self.metadata_file = metadata_file
        
        # 确保目录存在
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载元数据
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载元数据失败: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    def save_audio(
        self,
        speaker_id: str,
        audio_path: Path,
        overwrite: bool = False
    ) -> Optional[Path]:
        """
        保存音频文件
        
        Args:
            speaker_id: 说话人ID
            audio_path: 音频文件路径
            overwrite: 是否覆盖已存在的文件
            
        Returns:
            保存后的文件路径，失败返回None
        """
        try:
            # 检查是否已存在
            if speaker_id in self.metadata and not overwrite:
                print(f"说话人 {speaker_id} 已存在")
                return None
            
            # 生成目标文件名
            target_filename = f"{speaker_id}.wav"
            target_path = self.storage_dir / target_filename
            
            # 复制文件
            import shutil
            shutil.copy2(audio_path, target_path)
            
            # 更新元数据
            self.metadata[speaker_id] = {
                "audio_path": str(target_path.relative_to(self.storage_dir.parent)),
                "registered_at": datetime.now().isoformat(),
                "file_size": target_path.stat().st_size,
            }
            
            self._save_metadata()
            
            return target_path
            
        except Exception as e:
            print(f"保存音频失败: {e}")
            return None
    
    def get_audio_path(self, speaker_id: str) -> Optional[Path]:
        """获取说话人音频路径"""
        if speaker_id not in self.metadata:
            return None
        
        audio_path = self.metadata[speaker_id].get("audio_path")
        if audio_path:
            full_path = self.storage_dir.parent / audio_path
            if full_path.exists():
                return full_path
        
        return None
    
    def get_all_speakers(self, model_path: Optional[Path] = None) -> List[Dict]:
        """
        获取所有已注册的说话人列表
        
        Args:
            model_path: 模型文件路径，如果提供则从模型文件读取用户列表
        
        Returns:
            说话人列表
        """
        speakers = []
        speaker_ids = set()
        
        # 从模型文件读取用户列表
        if model_path and model_path.exists():
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                    speaker_ids.update(model_data.keys())
            except Exception as e:
                print(f"读取模型文件失败: {e}")
        
        # 从metadata读取用户列表（可能有些用户只在metadata中）
        speaker_ids.update(self.metadata.keys())
        
        # 合并信息
        for speaker_id in sorted(speaker_ids):
            info = self.metadata.get(speaker_id, {})
            speakers.append({
                "speaker_id": speaker_id,
                "audio_path": info.get("audio_path"),
                "registered_at": info.get("registered_at", "未知"),
                "file_size": info.get("file_size", 0),
            })
        
        return speakers
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """删除说话人"""
        try:
            if speaker_id not in self.metadata:
                return False
            
            # 删除音频文件
            audio_path = self.get_audio_path(speaker_id)
            if audio_path and audio_path.exists():
                audio_path.unlink()
            
            # 删除元数据
            del self.metadata[speaker_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"删除说话人失败: {e}")
            return False
    
    def speaker_exists(self, speaker_id: str) -> bool:
        """检查说话人是否存在"""
        return speaker_id in self.metadata

