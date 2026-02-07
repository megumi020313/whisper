"""IoTDB 集成测试脚本

【功能说明】
测试 IoTDB 连接器的基本功能，验证：
- IoTDB连接建立
- 音频事件写入
- 识别结果写入
- 数据查询功能
- 连接状态管理

【启动方式】
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_iotdb.py

【前置条件】
- IoTDB服务已启动（或使用模拟模式）
- 配置文件中IoTDB参数正确

【预期输出】
- IoTDB连接状态
- 数据写入测试结果
- 数据查询测试结果
- 各项功能测试通过/失败状态
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.data.iotdb import IoTDBConnector
from backend.utils.logger import get_logger
import time

logger = get_logger()


def test_iotdb_connection():
    """测试 IoTDB 连接"""
    print("\n" + "="*60)
    print("🧪 测试 IoTDB 连接")
    print("="*60)
    
    try:
        # 创建连接器（模拟模式）
        connector = IoTDBConnector()
        
        # 获取状态
        status = connector.get_status()
        print(f"\n📊 IoTDB 状态:")
        print(f"   - 启用: {status['enabled']}")
        print(f"   - 连接: {status['connected']}")
        print(f"   - 地址: {status['host']}:{status['port']}")
        
        if status['enabled'] and status['connected']:
            print("\n✅ IoTDB 连接成功")
        else:
            print("\n⚠️  IoTDB 未启用或未连接（模拟模式）")
        
        return connector
        
    except Exception as e:
        print(f"\n❌ IoTDB 连接失败: {e}")
        return None


def test_insert_audio_event(connector: IoTDBConnector):
    """测试写入音频事件"""
    print("\n" + "="*60)
    print("🧪 测试写入音频事件")
    print("="*60)
    
    try:
        # 当前时间戳（毫秒）
        timestamp = int(time.time() * 1000)
        
        # 写入测试数据
        success = connector.insert_audio_event(
            device_id="dev01",
            timestamp=timestamp,
            speaker="user001",
            text="这是一条测试语音识别文本",
            confidence=0.95
        )
        
        if success:
            print(f"\n✅ 音频事件写入成功")
            print(f"   - 设备: dev01")
            print(f"   - 时间戳: {timestamp}")
            print(f"   - 说话人: user001")
            print(f"   - 文本: 这是一条测试语音识别文本")
            print(f"   - 置信度: 0.95")
        else:
            print(f"\n❌ 音频事件写入失败")
        
        return success
        
    except Exception as e:
        print(f"\n❌ 写入测试失败: {e}")
        return False


def test_query_visual_context(connector: IoTDBConnector):
    """测试查询视觉上下文"""
    print("\n" + "="*60)
    print("🧪 测试查询视觉上下文")
    print("="*60)
    
    try:
        # 查询最近10秒的视觉数据
        timestamp = int(time.time() * 1000)
        start_ts = timestamp - 10000  # 10秒前
        end_ts = timestamp
        
        visual_tags = connector.query_visual_context(
            start_ts=start_ts,
            end_ts=end_ts,
            expand_seconds=5
        )
        
        print(f"\n📊 查询结果:")
        print(f"   - 时间范围: {start_ts} - {end_ts}")
        print(f"   - 扩展: 前后各5秒")
        print(f"   - 视觉标签: {visual_tags if visual_tags else '(无数据)'}")
        
        if visual_tags:
            print(f"\n✅ 查询到 {len(visual_tags)} 个视觉标签")
        else:
            print(f"\n⚠️  未查询到视觉数据（可能 P4 尚未写入数据）")
        
        return visual_tags
        
    except Exception as e:
        print(f"\n❌ 查询测试失败: {e}")
        return []


def test_query_audio_events(connector: IoTDBConnector):
    """测试查询音频事件"""
    print("\n" + "="*60)
    print("🧪 测试查询音频事件")
    print("="*60)
    
    try:
        # 查询最近10秒的音频数据
        timestamp = int(time.time() * 1000)
        start_ts = timestamp - 10000  # 10秒前
        end_ts = timestamp
        
        events = connector.query_audio_events(
            device_id="dev01",
            start_ts=start_ts,
            end_ts=end_ts
        )
        
        print(f"\n📊 查询结果:")
        print(f"   - 设备: dev01")
        print(f"   - 时间范围: {start_ts} - {end_ts}")
        print(f"   - 事件数量: {len(events)}")
        
        if events:
            print(f"\n✅ 查询到 {len(events)} 条音频事件:")
            for i, event in enumerate(events[:5], 1):  # 只显示前5条
                print(f"   {i}. [{event['timestamp']}] {event['speaker']}: {event['text'][:30]}...")
        else:
            print(f"\n⚠️  未查询到音频事件")
        
        return events
        
    except Exception as e:
        print(f"\n❌ 查询测试失败: {e}")
        return []


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("🚀 IoTDB 集成测试")
    print("="*60)
    
    # 1. 测试连接
    connector = test_iotdb_connection()
    if not connector:
        print("\n❌ 无法创建 IoTDB 连接器，测试终止")
        return
    
    # 2. 测试写入音频事件
    test_insert_audio_event(connector)
    
    # 3. 测试查询视觉上下文
    test_query_visual_context(connector)
    
    # 4. 测试查询音频事件
    test_query_audio_events(connector)
    
    # 5. 关闭连接
    connector.close()
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)
    print("\n💡 提示:")
    print("   - 如需启用 IoTDB，请在 config/model_config.yaml 中设置:")
    print("     fusion.iotdb.enabled: true")
    print("   - 确保 IoTDB 服务已启动")
    print("   - 确保 P4 视觉数据正在写入 IoTDB")
    print()


if __name__ == "__main__":
    main()

