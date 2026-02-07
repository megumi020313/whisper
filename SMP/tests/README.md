# Tests 目录说明

## 目录定位

**tests/** - 测试代码，包括单元测试、集成测试、测试工具

## 目录结构

```
tests/
├── unit/              # 单元测试
├── integration/       # 集成测试
├── utils/             # 测试工具和辅助函数
│   ├── test_helpers/  # 测试辅助工具
│   └── test_data/     # 测试数据
│       ├── configs/   # 测试配置文件
│       ├── fixtures/  # 测试固定数据
│       └── samples/   # 测试样本数据
└── scripts/           # 独立测试脚本（可选）
```

## 特点

- ✅ 可依赖所有backend模块
- ✅ 测试工具和辅助函数
- ❌ 不应被backend模块依赖

## 导入规范

```python
# ✅ 正确：使用绝对路径导入
from backend.service.user_service import UserService
from backend.data.user_models import User
from tests.utils.test_helpers import mock_factory

# ❌ 错误：不应使用相对路径
# from ..backend import something
```

## 参考文档

- [工具与测试文件夹规范](../../remote/.cursor/rules/工具与测试文件夹规范.mdc)

