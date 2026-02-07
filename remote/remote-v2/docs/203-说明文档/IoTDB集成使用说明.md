# IoTDB 集成使用说明

## 概述

IoTDB集成已完成，系统可以在识别完成后自动将对话流写入IoTDB时序数据库，为多模态融合分析提供数据基础。

## 功能特性

- ✅ **自动写入**：识别完成后自动将对话流写入IoTDB
- ✅ **模拟模式**：IoTDB不可用时自动降级到模拟模式，不影响主流程
- ✅ **批量写入**：支持批量写入V3.0对话智能引擎生成的高质量对话流
- ✅ **设备隔离**：支持多设备数据隔离（通过device_id参数）
- ✅ **时间对齐**：精确的时间戳对齐，支持多模态数据融合

## 配置说明

### 1. 配置文件位置

`config/model_config.yaml`

### 2. IoTDB配置项

```yaml
storage:
  iotdb:
    enabled: false              # 是否启用IoTDB（默认false）
    host: 127.0.0.1            # IoTDB服务器地址
    port: 6667                 # IoTDB端口（默认6667）
    username: root             # 用户名（默认root）
    password: root             # 密码（默认root）
    database: root.store_01    # 数据库路径
    timeout: 5                 # 连接超时（秒）
```

### 3. 启用IoTDB

**方法1：修改配置文件**

在 `config/model_config.yaml` 中设置：

```yaml
storage:
  iotdb:
    enabled: true  # 改为 true
```

**方法2：安装IoTDB客户端库**

```bash
pip install iotdb-session
```

**注意**：如果未安装IoTDB客户端库，系统会自动使用模拟模式，不会实际写入数据。

## API使用

### 识别接口

**端点**：`POST /recognize`

**参数**：
- `audio` (File, 必需): 音频文件（WAV格式）
- `threshold` (float, 可选): 相似度阈值
- `device_id` (string, 可选): 设备ID（用于IoTDB存储，默认dev01）

**示例**：

```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "audio=@test.wav" \
  -F "device_id=dev01"
```

### 响应格式

```json
{
  "success": true,
  "data": {
    "speaker_id": "zlh",
    "confidence": 95.5
  },
  "details": {
    "mode": "v3_diarization",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "user": "zlh",
        "text": "现在在测试一个项目",
        "raw_z_score": 5.2,
        "score": 65.0,
        "word_count": 6
      }
    ]
  }
}
```

## 数据格式

### IoTDB时序路径

```
root.store_01.audio.{device_id}
```

### 数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| timestamp | INT64 | 片段开始的绝对时间戳（毫秒） |
| speaker | TEXT | 说话人ID |
| text | TEXT | 完整对话文本 |
| z_score | FLOAT | 平均Z-score置信度 |
| word_count | INT32 | 词语数量 |
| duration | FLOAT | 片段时长（秒） |

### 示例数据

```
root.store_01.audio.dev01
├─ timestamp: 1703568000000
├─ speaker: "zlh"
├─ text: "现在在测试一个项目"
├─ z_score: 5.2
├─ word_count: 6
└─ duration: 2.5
```

## 工作流程

```
1. 用户调用 /recognize API
   ↓
2. 系统执行V3.0并行感知（SV + ASR）
   ↓
3. DiarizationEngine生成高质量对话流
   ↓
4. 检查IoTDB是否启用
   ├─ 是 → 写入IoTDB
   └─ 否 → 跳过（模拟模式）
   ↓
5. 返回识别结果给用户
```

## 测试

### 1. 单元测试

测试IoTDB连接器基本功能：

```bash
cd /home/swufe/Project/zhoulonghao/remote/remote-v2
conda activate vioce
python scripts/test/test_iotdb.py
```

### 2. 集成测试

测试完整的识别和写入流程：

```bash
python scripts/test/test_iotdb_integration.py
```

### 3. API测试

使用curl测试API调用：

```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "audio=@test.wav" \
  -F "device_id=dev01"
```

检查日志输出，应该看到：

```
✅ 对话流已写入IoTDB: dev01 | 3 个片段 | 基准时间: 2025-12-26T15:30:12.493000
```

## 模拟模式

当IoTDB未启用或不可用时，系统会自动切换到模拟模式：

- ✅ **不中断主流程**：识别功能正常工作
- ✅ **日志记录**：所有写入操作都会记录到日志
- ✅ **无副作用**：不会影响其他功能

模拟模式下的日志示例：

```
[模拟模式] 批量写入对话流: dev01 | 3 个片段
  [0.0s-2.5s] zlh: 现在在测试一个项目...
  [2.5s-5.8s] unknown: 这是一个多说话人场景...
  [5.8s-8.2s] zlh: 测试IoTDB集成功能...
```

## 故障排查

### 问题1：IoTDB写入失败

**症状**：日志显示"IoTDB写入失败"

**解决方案**：
1. 检查IoTDB服务是否启动
2. 检查配置文件中的连接参数是否正确
3. 检查网络连接是否正常
4. 查看详细错误日志

### 问题2：模拟模式

**症状**：日志显示"模拟模式"

**原因**：
- IoTDB未启用（`storage.iotdb.enabled: false`）
- IoTDB客户端库未安装（`pip install iotdb-session`）
- IoTDB服务未启动或连接失败

**解决方案**：
1. 安装IoTDB客户端库：`pip install iotdb-session`
2. 启动IoTDB服务
3. 在配置文件中启用IoTDB：`storage.iotdb.enabled: true`

### 问题3：数据未写入

**症状**：API调用成功，但IoTDB中查询不到数据

**检查步骤**：
1. 检查日志中是否有IoTDB写入成功的消息
2. 检查设备ID是否正确
3. 检查时间戳范围是否正确
4. 使用IoTDB客户端工具查询数据

## 性能考虑

- **异步写入**：IoTDB写入不会阻塞API响应
- **批量写入**：一次API调用写入所有对话片段
- **错误处理**：写入失败不会影响识别结果返回
- **连接复用**：IoTDB连接器使用单例模式，可复用

## 下一步

完成IoTDB集成后，可以：

1. **项目封装**：将阶段0+1+2封装为项目一（Audio-Sentry）
2. **多模态融合**：开发项目三（Logic-Flow），从IoTDB读取数据进行LLM分析
3. **监控告警**：添加IoTDB写入监控和告警机制

## 相关文档

- [声纹识别系统 技术白皮书](./声纹识别系统%20技术白皮书.md) - 3.6 IoTDB连接与写入层
- [多模态项目拆分](../1-设计/多模态项目拆分.md) - 项目一：Audio-Sentry
- [多模态 具体实现步骤](../1-设计/多模态%20具体实现步骤.md) - 阶段2：数据层打通

