"""优化的缓冲区管理 - 高效的环形缓冲区实现"""
from __future__ import annotations

import numpy as np
from typing import Optional


class OptimizedBuffer:
    """优化的环形缓冲区 - 避免频繁的内存分配"""
    
    def __init__(self, max_size: int, feature_dim: int, dtype=np.float32):
        """
        初始化环形缓冲区
        
        Args:
            max_size: 缓冲区最大容量（帧数）
            feature_dim: 特征维度
            dtype: 数据类型
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.dtype = dtype
        
        # 预分配内存
        self.buffer = np.zeros((max_size, feature_dim), dtype=dtype)
        self.current_idx = 0
        self.is_full = False
        self._size = 0
    
    def append(self, features: np.ndarray) -> None:
        """
        添加新的特征到缓冲区
        
        Args:
            features: 特征数组，形状应为 (feature_dim,)
        """
        if features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[0]}"
            )
        
        # 直接写入预分配的内存位置
        self.buffer[self.current_idx] = features.astype(self.dtype, copy=False)
        self.current_idx = (self.current_idx + 1) % self.max_size
        
        if self.current_idx == 0:
            self.is_full = True
        
        if not self.is_full:
            self._size += 1
    
    def get_sequence(self) -> np.ndarray:
        """
        获取缓冲区中的序列数据（按时间顺序）
        
        Returns:
            形状为 (size, feature_dim) 的数组
        """
        if not self.is_full:
            # 缓冲区未满，返回已填充部分
            return self.buffer[:self.current_idx].copy()
        
        # 缓冲区已满，返回正确顺序的环形数据
        # 最早的数据在 current_idx，最新的数据在 current_idx-1
        return np.vstack([
            self.buffer[self.current_idx:],
            self.buffer[:self.current_idx]
        ])
    
    def get_sequence_no_copy(self) -> np.ndarray:
        """
        获取缓冲区视图（不复制数据，仅用于只读操作）
        
        Warning:
            返回的数组不应被修改，修改会影响缓冲区内容
            
        Returns:
            缓冲区数据的视图
        """
        if not self.is_full:
            return self.buffer[:self.current_idx]
        return np.vstack([
            self.buffer[self.current_idx:],
            self.buffer[:self.current_idx]
        ])
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.current_idx = 0
        self.is_full = False
        self._size = 0
        # 不需要清零内存，下次写入会覆盖
    
    def size(self) -> int:
        """获取当前缓冲区中的有效数据数量"""
        return self.max_size if self.is_full else self._size
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self._size == 0 and not self.is_full
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return self.size()
    
    def __repr__(self) -> str:
        return (
            f"OptimizedBuffer(max_size={self.max_size}, "
            f"feature_dim={self.feature_dim}, "
            f"current_size={self.size()}, "
            f"is_full={self.is_full})"
        )


class AudioFrameBuffer:
    """音频帧缓冲区 - 用于管理音频帧序列"""
    
    def __init__(self, max_frames: int, frame_size: int, dtype=np.int16):
        """
        初始化音频帧缓冲区
        
        Args:
            max_frames: 最大帧数
            frame_size: 每帧的样本数
            dtype: 数据类型
        """
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.dtype = dtype
        
        # 预分配缓冲区
        self.buffer = np.zeros((max_frames, frame_size), dtype=dtype)
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
    
    def put(self, frame: np.ndarray) -> bool:
        """
        写入一帧数据
        
        Args:
            frame: 音频帧数据
            
        Returns:
            成功返回True，缓冲区满返回False
        """
        if self.count >= self.max_frames:
            return False  # 缓冲区满
        
        if len(frame) != self.frame_size:
            raise ValueError(
                f"Frame size mismatch: expected {self.frame_size}, got {len(frame)}"
            )
        
        self.buffer[self.write_idx] = frame
        self.write_idx = (self.write_idx + 1) % self.max_frames
        self.count += 1
        return True
    
    def get(self) -> Optional[np.ndarray]:
        """
        读取一帧数据
        
        Returns:
            音频帧数据，缓冲区空返回None
        """
        if self.count == 0:
            return None
        
        frame = self.buffer[self.read_idx].copy()
        self.read_idx = (self.read_idx + 1) % self.max_frames
        self.count -= 1
        return frame
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.count >= self.max_frames
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self.count == 0
    
    def available_space(self) -> int:
        """返回可用空间"""
        return self.max_frames - self.count
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
