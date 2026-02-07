# SMP 项目结构

## 项目概览

本项目按照模块划分规范构建，为后续从 `smp/` 目录迁移代码做准备。

## 目录结构

```
SMP/
├── backend/                    # 后端模块（统一在backend/下）
│   ├── config/                # 配置管理
│   ├── data/                  # 数据存储访问（SQLAlchemy模型）
│   ├── utils/                 # 纯工具函数
│   ├── common/                # 业务通用组件
│   ├── modules/               # 完整业务功能组合
│   ├── scheduler/             # 系统协调管理，运行时任务调度
│   ├── controller/            # 控制器层（MVC架构）
│   ├── service/               # 服务层（MVC架构）
│   ├── schema/                # 数据验证层（MVC架构）
│   ├── api/                   # API接口层（FastAPI路由）
│   ├── core/                  # 核心业务层
│   ├── models/                # 模型层（深度学习模型）
│   ├── pipeline/              # 推理流水线
│   └── training/              # 训练相关（可选）
├── scripts/                   # 命令行入口和维护脚本
├── tests/                     # 测试代码
│   ├── unit/                  # 单元测试
│   ├── integration/           # 集成测试
│   └── utils/                 # 测试工具
├── frontend/                  # 前端代码（Vue3 + TypeScript）
│   └── src/
│       ├── utils/             # 纯工具函数
│       ├── common/            # 业务通用组件
│       ├── config/            # 配置管理
│       ├── api/               # API接口封装
│       ├── stores/            # 状态管理（Pinia）
│       ├── components/        # 可复用组件
│       ├── views/             # 页面视图
│       └── routers/           # 路由配置
└── README.md                  # 本文件
```

## 模块定位规则

### 后端模块（backend/）

所有后端模块统一在 `backend/` 文件夹下，保持同一层级：

1. **backend/utils/** - 纯工具函数
   - 无状态，不依赖业务逻辑
   - 可被任何模块使用

2. **backend/common/** - 业务通用组件
   - 包含业务逻辑，多模块复用
   - 可依赖utils、data、config

3. **backend/data/** - 数据存储访问
   - SQLAlchemy模型定义
   - 数据库操作封装

4. **backend/modules/** - 完整业务功能组合
   - 完整业务功能，独立可运行
   - 可依赖data、common、utils

5. **backend/scheduler/** - 系统协调管理
   - 定时任务定义
   - 系统级任务调度

6. **backend/controller/** - 控制器层
   - 处理HTTP请求和路由
   - 调用service层

7. **backend/service/** - 服务层
   - 业务逻辑处理
   - 调用data层

8. **backend/schema/** - 数据验证层
   - 请求参数验证
   - 响应数据格式化

9. **backend/api/** - API接口层
   - FastAPI路由和接口定义
   - 依赖注入

10. **backend/core/** - 核心业务层
    - 核心业务逻辑

11. **backend/models/** - 模型层
    - 深度学习模型定义和加载

12. **backend/pipeline/** - 推理流水线
    - 推理流水线管理

13. **backend/training/** - 训练相关（可选）
    - 训练脚本和工具

### 其他目录

- **scripts/** - 命令行入口和维护脚本
- **tests/** - 测试代码（可依赖所有backend模块）
- **frontend/** - 前端代码（Vue3 + TypeScript）

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

## 迁移计划

### 从 smp/ 目录迁移代码

1. **核心模块迁移**：
   - `smp/code/software/core/` → `SMP/backend/core/`
   - `smp/code/software/utils/` → `SMP/backend/utils/`（如果有）

2. **数据层迁移**：
   - `smp/code/software/data/` → `SMP/backend/data/`（如果有）

3. **业务模块迁移**：
   - `smp/code/software/modules/` → `SMP/backend/modules/`（如果有）
   - `smp/code/software/deployment/` → `SMP/backend/modules/deployment/`

4. **训练相关迁移**：
   - `smp/code/software/training/` → `SMP/backend/training/`

5. **Web服务迁移**：
   - `smp/code/software/web/app.py` → `SMP/backend/api/app.py`
   - `smp/code/software/web/static/` → `SMP/frontend/src/`
   - `smp/code/software/web/templates/` → `SMP/frontend/src/views/`

6. **配置迁移**：
   - `smp/config/Config.py` → `SMP/backend/config/Config.py`
   - `smp/code/software/web/config.py` → `SMP/backend/config/web_config.py`

7. **测试迁移**：
   - `smp/code/software/test/` → `SMP/tests/`

## 参考文档

- [模块划分规范](../remote/.cursor/rules/模块划分规范.mdc)
- [导入路径规范](../remote/.cursor/rules/导入路径规范.mdc)
- [数据与配置规范](../remote/.cursor/rules/数据与配置规范.mdc)

## 注意事项

1. **迁移顺序**：建议先迁移utils和common，再迁移其他模块
2. **依赖检查**：迁移时注意检查依赖关系，确保符合规范
3. **导入路径**：迁移后统一使用绝对路径导入
4. **测试验证**：迁移后运行测试确保功能正常

