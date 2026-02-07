# Brain-Core 项目文档

## 文档目录结构

本项目文档按照《文档管理规范》组织：

```
docs/
├── 200-需求管理/                    # 需求管理
│   └── README.md                    # 需求管理主索引
│
├── 201-Bug修复任务管理/             # Bug修复任务管理
│   └── README.md                    # Bug修复任务管理主索引
│
├── 202-优化任务管理/                 # 优化任务管理
│   └── README.md                    # 优化任务管理主索引
│
└── changelog/                       # 变更日志
    └── CHANGLOG_TEMPLATE.md         # 变更日志模板
```

## 项目目录结构

本项目按照《模块划分规范》组织：

```
brain_core/
├── backend/                         # 后端代码（统一在backend/下）
│   ├── config/                     # 配置管理
│   ├── utils/                      # 纯工具函数
│   ├── common/                     # 业务通用组件
│   ├── modules/                    # 完整业务功能组合
│   │   └── llm_engine/             # LLM引擎模块
│   ├── service/                    # 服务层
│   ├── controller/                 # 控制器层
│   └── schema/                     # 数据验证层
│
├── tests/                           # 测试代码
├── scripts/                         # 命令行入口和维护脚本
├── models/                          # 模型存储（方案二：项目内相对路径）
│   └── llm/                         # LLM模型目录
├── docs/                            # 文档目录
├── requirements.txt                 # Python依赖
├── Dockerfile                       # Docker镜像构建文件
└── README.md                        # 项目说明文档
```

## 规范符合性

### ✅ 文档管理规范
- [x] 文档目录结构符合规范（200/201/202系列）
- [x] 变更日志目录已创建
- [x] 各目录索引文件已创建

### ✅ 模块划分规范
- [x] 后端模块统一在 `backend/` 下
- [x] 模块层级清晰（utils/common/modules/service/controller/schema）
- [x] 目录结构支持依赖关系规范

### ✅ 导入路径规范
- [x] 目录结构支持绝对路径导入（`from backend.xxx import yyy`）
- [x] 无相对路径导入需求

### ✅ 函数规范
- [x] 目录结构支持函数规范要求
- [x] 模块划分清晰，便于函数复用

## 相关规范文档

- [文档管理规范](../../.cursor/rules/文档管理规范.mdc)
- [模块划分规范](../../.cursor/rules/模块划分规范.mdc)
- [导入路径规范](../../.cursor/rules/导入路径规范.mdc)
- [函数规范](../../.cursor/rules/函数规范.mdc)

