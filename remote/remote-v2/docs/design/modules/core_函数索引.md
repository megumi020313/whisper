# Core模块函数索引

## 模块说明

`src/core/` 包含核心功能组件：配置管理、异常定义、向量存储。

## 📁 子模块

### config.py - 配置管理

##### Config类（单例）
```python
class Config:
    """全局配置管理器（单例模式）"""
```

**核心配置**:
- **模型路径**: `vad_model_path`, `speaker_model_path`
- **阈值参数**: `low_threshold=2.5`, `high_threshold=4.5`
- **VAD参数**: `min_silence_duration_ms=300`
- **批处理**: `batch_size=64`（V14.2）
- **设备**: `device`（cuda:0/cpu）

---

##### get_config
```python
def get_config() -> Config
```
**功能**: 获取全局配置实例

**返回**: 全局Config单例实例

**使用示例**:
```python
from src.core.config import get_config

config = get_config()
print(f"Device: {config.device}")
print(f"Low threshold: {config.low_threshold}")
```

---

### exceptions.py - 异常定义

##### VoiceprintError
```python
class VoiceprintError(Exception):
    """声纹识别基础异常"""
```

---

##### AudioFormatError
```python
class AudioFormatError(VoiceprintError):
    """音频格式错误"""
```

**使用场景**: 音频加载、验证失败

---

##### ModelLoadError
```python
class ModelLoadError(VoiceprintError):
    """模型加载错误"""
```

**使用场景**: VAD、声纹模型加载失败

---

##### VectorStorageError
```python
class VectorStorageError(VoiceprintError):
    """向量存储错误"""
```

---

##### UserNotFoundError
```python
class UserNotFoundError(VectorStorageError):
    """用户不存在"""
```

---

##### UserAlreadyExistsError
```python
class UserAlreadyExistsError(VectorStorageError):
    """用户已存在"""
```

---

### vector_storage.py - 向量存储与匹配

#### 核心类

##### VectorStorage
```python
class VectorStorage:
    """本地向量数据库管理器（Pickle存储）"""
```

**方法**:

###### register
```python
def register(
    self,
    user_id: str,
    embedding: np.ndarray,
    metadata: Optional[Dict] = None
) -> None
```
**功能**: 注册用户向量

**参数**:
- `user_id`: 用户ID
- `embedding`: 声纹特征向量（192维）
- `metadata`: 元数据（可选）

**异常**: `UserAlreadyExistsError` - 用户已存在

---

###### identify
```python
def identify(
    self,
    query_embedding: np.ndarray,
    threshold: Optional[float] = None
) -> Tuple[Optional[str], float]
```
**功能**: 识别用户（AS-Norm评分）

**参数**:
- `query_embedding`: 查询向量（192维）
- `threshold`: 识别阈值（Z-score），默认使用config.low_threshold

**返回**: `(user_id, z_score)` 或 `(None, best_score)`

**核心算法**: AS-Norm自适应归一化
1. 计算query与所有用户的余弦相似度
2. 计算query与cohort的相似度分布
3. Z-score归一化：`z = (score - mean) / std`
4. 阈值判断：`z >= threshold`

---

###### get_user
```python
def get_user(self, user_id: str) -> Dict[str, Any]
```
**功能**: 获取用户信息

**返回**: 包含embedding和metadata的字典

**异常**: `UserNotFoundError` - 用户不存在

---

###### list_users
```python
def list_users(self) -> List[str]
```
**功能**: 列出所有用户ID

**返回**: 用户ID列表

---

###### delete_user
```python
def delete_user(self, user_id: str) -> None
```
**功能**: 删除用户

**异常**: `UserNotFoundError` - 用户不存在

---

##### RedisVectorStorage
```python
class RedisVectorStorage(VectorStorage):
    """Redis向量数据库管理器"""
```

**说明**: 与VectorStorage接口一致，使用Redis作为后端存储

**配置**:
- `REDIS_HOST`: Redis主机，默认localhost
- `REDIS_PORT`: Redis端口，默认6379
- `REDIS_DB`: Redis数据库，默认0

---

#### 工厂函数

##### get_vector_storage
```python
def get_vector_storage(
    backend: Optional[str] = None,
    storage_path: Optional[Path] = None
) -> VectorStorage
```
**功能**: 根据环境变量或入参选择存储后端

**参数**:
- `backend`: 后端类型，"local"或"redis"
- `storage_path`: 存储路径（仅用于本地后端）

**返回**: VectorStorage实例

**环境变量**: `VECTOR_STORAGE_BACKEND`（默认"local"）

---

#### 核心算法函数

##### compute_as_norm_score
```python
def compute_as_norm_score(
    query_embedding: np.ndarray,
    template_embedding: np.ndarray,
    cohort_embeddings: Optional[np.ndarray] = None
) -> float
```
**功能**: AS-Norm自适应归一化评分

**参数**:
- `query_embedding`: 查询向量（192维）
- `template_embedding`: 模板向量（192维）
- `cohort_embeddings`: Cohort向量集（500×192），可选

**返回**: Z-score归一化分数

**算法原理**:
```
1. raw_score = cosine_similarity(query, template)
2. cohort_scores = [cosine_similarity(query, c) for c in cohort]
3. mean = np.mean(cohort_scores)
4. std = np.std(cohort_scores)
5. z_score = (raw_score - mean) / std
```

**优势**:
- 自适应阈值：不同说话人分布自动归一化
- 鲁棒性强：对噪音、信道变化不敏感
- 可解释性：Z-score直观表示"比平均水平高多少个标准差"

---

## 使用示例

### 配置管理
```python
from src.core.config import get_config

config = get_config()
print(f"VAD阈值: {config.vad_threshold}")
print(f"识别阈值: {config.low_threshold} - {config.high_threshold}")
print(f"设备: {config.device}")
```

### 向量存储
```python
from src.core.vector_storage import get_vector_storage
import numpy as np

# 获取存储实例
storage = get_vector_storage(backend="local")

# 注册用户
embedding = np.random.randn(192).astype(np.float32)
storage.register(
    user_id="user123",
    embedding=embedding,
    metadata={"name": "张三", "created_at": "2025-12-22"}
)

# 识别用户
query_emb = np.random.randn(192).astype(np.float32)
user_id, z_score = storage.identify(query_emb, threshold=2.5)

if user_id:
    print(f"识别为: {user_id}, Z-score: {z_score:.2f}")
else:
    print(f"未识别, 最高分: {z_score:.2f}")

# 列出所有用户
users = storage.list_users()
print(f"注册用户: {users}")

# 删除用户
storage.delete_user("user123")
```

### 异常处理
```python
from src.core.exceptions import UserAlreadyExistsError, UserNotFoundError

try:
    storage.register("user123", embedding)
except UserAlreadyExistsError:
    print("用户已存在，请使用其他ID")

try:
    user_info = storage.get_user("nonexistent")
except UserNotFoundError:
    print("用户不存在")
```

---

## 依赖关系

```
core/
├── config.py (无依赖)
├── exceptions.py (无依赖)
└── vector_storage.py
    ├── → config.py
    ├── → exceptions.py
    └── → utils/logger.py
```

---

**最后更新**: 2025-12-22  
**维护者**: 项目开发团队

