# Frontend 目录说明

## 目录定位

**frontend/** - 前端代码（Vue3 + TypeScript + Vite）

## 目录结构

```
frontend/
└── src/
    ├── utils/         # 纯工具函数（对应后端utils）
    ├── common/        # 业务通用组件（对应后端common）
    ├── config/        # 配置管理（与后端config对应）
    ├── api/           # API接口封装
    ├── stores/        # 状态管理（Pinia）
    ├── components/    # 可复用组件
    ├── views/         # 页面视图
    └── routers/       # 路由配置
```

## 模块说明

### frontend/src/utils/ - 纯工具函数
- **定位**：纯工具函数，对应后端utils
- **特点**：纯函数，无副作用，无状态

### frontend/src/common/ - 业务通用组件
- **定位**：业务通用组件，对应后端common
- **特点**：UI通用组件，多页面复用

### frontend/src/config/ - 配置管理
- **定位**：客户端配置，与后端config对应
- **特点**：UI主题配置、路由配置、API地址配置

### frontend/src/api/ - API接口封装
- **定位**：API接口封装，与后端通信
- **特点**：HTTP请求封装、类型定义、错误处理

### frontend/src/stores/ - 状态管理
- **定位**：Pinia状态管理
- **特点**：全局状态、用户状态、应用状态

### frontend/src/components/ - 可复用组件
- **定位**：可复用的Vue组件
- **特点**：UI组件、业务组件

### frontend/src/views/ - 页面视图
- **定位**：页面视图
- **特点**：路由对应的页面组件

### frontend/src/routers/ - 路由配置
- **定位**：路由配置
- **特点**：路由定义、路由守卫

## 参考文档

- [模块划分规范](../../remote/.cursor/rules/模块划分规范.mdc)
- [数据与配置规范](../../remote/.cursor/rules/数据与配置规范.mdc)

