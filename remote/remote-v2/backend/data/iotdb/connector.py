"""IoTDB 连接器

提供与 IoTDB 时序数据库的连接和数据操作功能。
用于存储音频事件和查询视觉上下文。
"""
from typing import List, Dict, Optional, Any
import time

from backend.utils.logger import get_logger
from backend.core.config import get_config

logger = get_logger()


class IoTDBConnector:
    """IoTDB 连接器
    
    负责：
    1. 写入音频事件（说话人、文本、时间戳）
    2. 查询视觉上下文（用于多模态融合）
    3. 管理时序数据的存储和检索
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """初始化 IoTDB 连接器
        
        Args:
            host: IoTDB 服务器地址，默认从配置文件读取
            port: IoTDB 端口，默认从配置文件读取
            username: 用户名，默认从配置文件读取
            password: 密码，默认从配置文件读取
        """
        self._iotdb_available = False
        try:
            from iotdb.Session import Session
            from iotdb.utils.IoTDBConstants import TSDataType
            
            self.TSDataType = TSDataType
            self.Session = Session
            self._iotdb_available = True
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"IoTDB 客户端不可用: {e}")
            logger.warning("将使用模拟模式运行（数据不会实际写入 IoTDB）")
            self.TSDataType = None
            self.Session = None
        
        # 获取配置
        config = get_config()
        
        # 使用参数或配置文件中的值
        self.host = host or getattr(config, 'iotdb_host', '127.0.0.1')
        self.port = port or getattr(config, 'iotdb_port', 6667)
        self.username = username or getattr(config, 'iotdb_username', 'root')
        self.password = password or getattr(config, 'iotdb_password', 'root')
        
        # 是否启用 IoTDB
        self.enabled = getattr(config, 'iotdb_enabled', False)
        
        self.session = None
        self._connected = False
        
        if not self._iotdb_available:
            logger.info("📁 IoTDB 客户端不可用，使用模拟模式")
            self.enabled = False
        elif self.enabled:
            self._connect()
        else:
            logger.info("📁 IoTDB 未启用（配置文件中 fusion.iotdb.enabled=false）")
    
    def _connect(self):
        """建立 IoTDB 连接"""
        try:
            from iotdb.Session import Session
            
            self.session = Session(
                self.host,
                self.port,
                self.username,
                self.password
            )
            self.session.open(False)
            self._connected = True
            logger.info(f"✅ IoTDB 连接成功: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"❌ IoTDB 连接失败: {e}")
            logger.warning("将使用模拟模式运行（数据不会实际写入 IoTDB）")
            self._connected = False
    
    def insert_audio_event(
        self,
        device_id: str,
        timestamp: int,
        speaker: str,
        text: str,
        confidence: Optional[float] = None
    ) -> bool:
        """写入音频事件
        
        Args:
            device_id: 设备ID（如 "dev01"）
            timestamp: Unix 时间戳（毫秒）
            speaker: 说话人ID
            text: 识别的文本内容
            confidence: 置信度（可选）
        
        Returns:
            是否写入成功
        
        示例路径: root.store_01.audio.dev01
        """
        if not self.enabled or not self._connected:
            logger.debug(f"[模拟模式] 音频事件: {device_id} | {speaker} | {text[:20]}...")
            return True
        
        try:
            device_path = f"root.store_01.audio.{device_id}"
            
            if confidence is not None:
                measurements = ["speaker", "text", "confidence"]
                types = [self.TSDataType.TEXT, self.TSDataType.TEXT, self.TSDataType.FLOAT]
                values = [speaker, text, confidence]
            else:
                measurements = ["speaker", "text"]
                types = [self.TSDataType.TEXT, self.TSDataType.TEXT]
                values = [speaker, text]
            
            self.session.insert_record(
                device_path,
                timestamp,
                measurements,
                types,
                values
            )
            
            logger.debug(f"✅ 音频事件已写入 IoTDB: {device_id} @ {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 写入音频事件失败: {e}")
            return False
    
    def insert_dialogue_transcript(
        self,
        device_id: str,
        dialogue_segments: List[Dict[str, Any]],
        base_timestamp: int
    ) -> bool:
        """批量写入高质量对话流（V3.0）
        
        这是V3.0方案的核心写入接口，将DiarizationEngine生成的
        高质量对话流批量写入IoTDB。
        
        Args:
            device_id: 设备ID（如 "dev01"）
            dialogue_segments: 对话片段列表
                格式: [{
                    "start": 0.0,
                    "end": 2.5,
                    "speaker": "zlh",
                    "transcript": "现在在测试一个项目",
                    "avg_z_score": 5.2,
                    "word_count": 6
                }, ...]
            base_timestamp: 基准时间戳（毫秒），音频开始的绝对时间
        
        Returns:
            是否全部写入成功
        """
        if not self.enabled or not self._connected:
            logger.debug(
                f"[模拟模式] 批量写入对话流: {device_id} | "
                f"{len(dialogue_segments)} 个片段"
            )
            for seg in dialogue_segments:
                logger.debug(
                    f"  [{seg['start']:.1f}s-{seg['end']:.1f}s] "
                    f"{seg['speaker']}: {seg['transcript'][:30]}..."
                )
            return True
        
        try:
            device_path = f"root.store_01.audio.{device_id}"
            success_count = 0
            
            for segment in dialogue_segments:
                # 计算绝对时间戳（毫秒）
                segment_timestamp = base_timestamp + int(segment['start'] * 1000)
                
                # 准备数据
                measurements = [
                    "speaker",
                    "text",
                    "z_score",
                    "word_count",
                    "duration"
                ]
                types = [
                    self.TSDataType.TEXT,
                    self.TSDataType.TEXT,
                    self.TSDataType.FLOAT,
                    self.TSDataType.INT32,
                    self.TSDataType.FLOAT
                ]
                values = [
                    segment['speaker'],
                    segment['transcript'],
                    segment['avg_z_score'],
                    segment['word_count'],
                    segment.get('duration', segment['end'] - segment['start'])
                ]
                
                # 写入单条记录
                self.session.insert_record(
                    device_path,
                    segment_timestamp,
                    measurements,
                    types,
                    values
                )
                
                success_count += 1
            
            logger.info(
                f"✅ 批量写入对话流完成: {device_id} | "
                f"{success_count}/{len(dialogue_segments)} 个片段"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ 批量写入对话流失败: {e}")
            return False
    
    def query_visual_context(
        self,
        start_ts: int,
        end_ts: int,
        expand_seconds: int = 5
    ) -> List[str]:
        """查询视觉上下文（给融合引擎使用）
        
        Args:
            start_ts: 开始时间戳（毫秒）
            end_ts: 结束时间戳（毫秒）
            expand_seconds: 向前向后扩展的秒数，默认5秒
        
        Returns:
            视觉标签列表（去重后），如 ['customer', 'menu', 'goods']
        
        示例路径: root.store_01.visual
        """
        if not self.enabled or not self._connected:
            logger.debug(f"[模拟模式] 查询视觉上下文: {start_ts} - {end_ts}")
            return []
        
        try:
            # 向前向后各扩展指定秒数
            q_start = start_ts - (expand_seconds * 1000)
            q_end = end_ts + (expand_seconds * 1000)
            
            sql = f"SELECT class FROM root.store_01.visual WHERE time >= {q_start} AND time <= {q_end}"
            
            objects = []
            with self.session.execute_query_statement(sql) as session_data_set:
                while session_data_set.has_next():
                    row = session_data_set.next()
                    # 假设第一列是 class
                    class_name = row.get_fields()[0]
                    objects.append(class_name)
            
            # 去重返回
            unique_objects = list(set(objects))
            logger.debug(f"✅ 查询到视觉标签: {unique_objects}")
            return unique_objects
            
        except Exception as e:
            logger.error(f"❌ 查询视觉上下文失败: {e}")
            return []
    
    def query_audio_events(
        self,
        device_id: str,
        start_ts: int,
        end_ts: int
    ) -> List[Dict[str, Any]]:
        """查询音频事件
        
        Args:
            device_id: 设备ID
            start_ts: 开始时间戳（毫秒）
            end_ts: 结束时间戳（毫秒）
        
        Returns:
            音频事件列表
        """
        if not self.enabled or not self._connected:
            logger.debug(f"[模拟模式] 查询音频事件: {device_id} | {start_ts} - {end_ts}")
            return []
        
        try:
            device_path = f"root.store_01.audio.{device_id}"
            sql = f"SELECT speaker, text FROM {device_path} WHERE time >= {start_ts} AND time <= {end_ts}"
            
            events = []
            with self.session.execute_query_statement(sql) as session_data_set:
                while session_data_set.has_next():
                    row = session_data_set.next()
                    fields = row.get_fields()
                    events.append({
                        "timestamp": row.get_timestamp(),
                        "speaker": fields[0],
                        "text": fields[1]
                    })
            
            logger.debug(f"✅ 查询到 {len(events)} 条音频事件")
            return events
            
        except Exception as e:
            logger.error(f"❌ 查询音频事件失败: {e}")
            return []
    
    def close(self):
        """关闭连接"""
        if self._connected and self.session:
            try:
                self.session.close()
                logger.info("✅ IoTDB 连接已关闭")
            except Exception as e:
                logger.error(f"❌ 关闭 IoTDB 连接失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    def get_status(self) -> Dict[str, Any]:
        """获取连接状态信息
        
        Returns:
            状态信息字典
        """
        return {
            "enabled": self.enabled,
            "connected": self._connected,
            "host": self.host,
            "port": self.port
        }

