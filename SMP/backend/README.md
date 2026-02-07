# Backend 模块说明

## 目录结构

根据模块划分规范，所有后端模块统一在 `backend/` 文件夹下：

```
backend/
├── config/          # 配置管理
├── data/            # 数据存储访问，持久化层（SQLAlchemy模型）
├── utils/           # 纯工具函数，无状态，不依赖业务逻辑
├── common/          # 业务通用组件，多模块复用
├── modules/         # 完整业务功能组合
├── scheduler/       # 系统协调管理，运行时任务调度
├── controller/     # 控制器层（MVC架构）
├── service/         # 服务层（MVC架构）
├── schema/          # 数据验证层（MVC架构）
├── api/             # API接口层（FastAPI路由）
├── core/            # 核心业务层
├── models/          # 模型层（深度学习模型）
├── pipeline/        # 推理流水线
└── training/        # 训练相关（可选，可保留在根目录）
```

## 模块定位规则

### backend/utils/ - 纯工具函数
- **定位**：纯工具函数，无状态，不依赖业务逻辑
- **特点**：可被任何模块使用，不包含业务逻辑
- **示例**：日期时间处理、字符串处理、文件操作

### backend/common/ - 业务通用组件
- **定位**：业务通用组件，多模块复用
- **特点**：包含业务逻辑，可依赖utils、data、config
- **示例**：权限验证、工作流引擎、通用服务

### backend/data/ - 数据存储访问
- **定位**：数据存储访问，持久化层（SQLAlchemy模型）
- **特点**：SQLAlchemy模型定义，数据库操作封装
- **示例**：数据模型、数据库操作

### backend/modules/ - 完整业务功能组合
- **定位**：完整业务功能组合，独立可运行
- **特点**：完整业务功能，可依赖data、common、utils
- **示例**：审批模块、集成模块

### backend/scheduler/ - 系统协调管理
- **定位**：系统协调管理，运行时任务调度
- **特点**：定时任务定义，系统级任务调度
- **示例**：定时任务、系统级调度

### backend/controller/ - 控制器层
- **定位**：处理HTTP请求和路由（MVC架构）
- **特点**：路由注册，请求参数验证，调用service层

### backend/service/ - 服务层
- **定位**：业务逻辑处理（MVC架构）
- **特点**：业务逻辑实现，调用data层

### backend/schema/ - 数据验证层
- **定位**：数据验证（MVC架构）
- **特点**：请求参数验证，响应数据格式化

### backend/api/ - API接口层
- **定位**：FastAPI路由和接口定义
- **特点**：API路由注册，依赖注入

### backend/core/ - 核心业务层
- **定位**：核心业务逻辑
- **特点**：核心功能实现

### backend/models/ - 模型层
- **定位**：深度学习模型定义和加载
- **特点**：模型定义、模型加载、模型推理

### backend/pipeline/ - 推理流水线
- **定位**：推理流水线管理
- **特点**：流水线组装、推理流程

## 依赖关系

```
tests → backend/scheduler → backend/modules → {backend/data, backend/common} → backend/utils
backend/controller → backend/service → {backend/data, backend/modules, backend/common} → backend/utils
backend/api → backend/core → backend/service → backend/data
backend/modules → {backend/data, backend/common} → backend/utils
backend/common → {backend/data, backend/utils}
backend/data → backend/utils
```

## 导入路径规范

所有导入必须使用绝对路径：

```python
# ✅ 正确
from backend.utils.logger import setup_logger
from backend.common.config import get_config
from backend.data.user_models import User
from backend.service.user_service import UserService

# ❌ 错误
from .utils import logger
from ..common import config
```

## 参考文档

- [模块划分规范](../../remote/.cursor/rules/模块划分规范.mdc)
- [导入路径规范](../../remote/.cursor/rules/导入路径规范.mdc)

