# Scripts 目录说明

## 目录定位

**scripts/** - 命令行入口，参数解析和维护脚本

## 特点

- ✅ 项目维护脚本
- ✅ 数据库初始化
- ✅ 用户管理
- ✅ 可依赖backend模块
- ✅ 独立运行的工具脚本

## 使用场景

- 数据库初始化脚本
- 用户管理脚本
- 数据迁移脚本
- 系统维护脚本

## 导入规范

```python
# ✅ 正确：使用绝对路径导入backend模块
from backend.data.user_models import User
from backend.service.user_service import UserService

# ❌ 错误：不应使用相对路径
# from ..backend import something
```

