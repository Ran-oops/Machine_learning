# 贡献指南

感谢您对 Machine Learning 项目的关注！本指南将帮助您了解如何参与项目贡献。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)

## 🤝 行为准则

- 尊重所有参与者
- 欢迎新人和各种背景的贡献者
- 保持友善和建设性的沟通
- 专注于对项目和社区最有利的事情

## 🚀 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议：

1. 先搜索 [Issues](https://github.com/yourusername/machine-learning/issues) 确认问题未被报告
2. 如果问题不存在，创建新 Issue 并提供以下信息：
   - 问题描述
   - 复现步骤
   - 期望行为
   - 实际行为
   - 环境信息（Python版本、操作系统等）

### 提交代码

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m "feat: add amazing feature"`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 💻 开发环境设置

### 前提条件

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) - Python 包管理器

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/machine-learning.git
cd machine-learning

# 设置环境（使用 just）
just init

# 或使用脚本
python scripts/setup.py
```

### 手动设置

```bash
# 同步依赖
uv sync

# 安装开发工具
uv tool install pre-commit
uv tool install ruff
uv tool install rumdl

# 设置 pre-commit hooks
pre-commit install
```

## 📝 代码规范

### Python 代码风格

我们使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式化和检查：

```bash
# 格式化代码
ruff format src/

# 代码检查
ruff check src/

# 自动修复问题
ruff check --fix src/
```

### 代码规范要点

1. **导入顺序**：标准库 → 第三方库 → 本地模块
2. **命名规范**：
   - 模块名：`lowercase_with_underscores`
   - 类名：`PascalCase`
   - 函数/变量名：`lowercase_with_underscores`
   - 常量名：`UPPERCASE_WITH_UNDERSCORES`
3. **文档字符串**：所有公共函数和类必须包含文档字符串
4. **类型注解**：鼓励使用类型注解

### 示例

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001
) -> Dict[str, List[float]]:
    """
    训练神经网络模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        epochs: 训练轮数
        lr: 学习率
        
    Returns:
        包含训练历史的字典
    """
    ...
```

## 📤 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 提交格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 示例

```bash
# 新功能
feat(notebook): add CNN tutorial notebook

# Bug 修复
fix(utils): fix data loading memory leak

# 文档更新
docs(readme): update installation instructions

# 代码重构
refactor(algorithms): improve BPNN training performance

# 多个范围
feat(utils,algorithms): add visualization tools and FFT algorithms
```

## 🔀 Pull Request 流程

### 准备 PR

1. **更新文档**：如果添加了新功能，请更新 README 和文档
2. **添加测试**：为新功能添加测试
3. **运行检查**：
   ```bash
   just lint      # 代码检查
   just format    # 格式化
   just test      # 运行测试
   ```
4. **更新 CHANGELOG**：在 CHANGELOG.md 中添加您的更改

### PR 模板

```markdown
## 描述
简要描述这个 PR 的目的

## 更改类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 其他

## 检查清单
- [ ] 代码符合项目规范
- [ ] 添加/更新了测试
- [ ] 更新了文档
- [ ] 通过了所有检查

## 相关 Issue
Fixes #(issue number)
```

### 审查流程

1. 维护者会审查您的 PR
2. 根据反馈进行修改
3. PR 被合并后会关闭相关的 Issue

## 🎯 贡献领域

我们特别欢迎以下方面的贡献：

### 📚 教程和文档

- 改进现有 notebook 的注释
- 添加新的学习笔记
- 完善文档和示例代码

### 🧮 算法实现

- 实现经典的机器学习算法
- 优化现有算法的性能
- 添加算法的可视化解释

### 🛠️ 工具开发

- 改进数据加载工具
- 添加可视化功能
- 优化训练工具

### 🐛 Bug 修复

- 修复 notebook 中的错误
- 修复工具函数的问题
- 改进错误处理

## 📞 联系方式

- 创建 [Issue](https://github.com/yourusername/machine-learning/issues)
- 发送邮件到: your-email@example.com

## 📄 许可证

通过贡献代码，您同意您的贡献将在 [MIT 许可证](LICENSE) 下发布。

---

**感谢所有贡献者！** 🎉
