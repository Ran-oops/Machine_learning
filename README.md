# 机器学习项目

基于 PyTorch 的深度学习学习项目，跟随《动手学深度学习》教程实现。

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [学习路线](#学习路线)
- [模块使用](#模块使用)
- [开发工具](#开发工具)
- [贡献指南](#贡献指南)
- [参考资料](#参考资料)

## 🎯 项目简介

本项目是一个面向**机器学习和深度学习初学者**的学习仓库，基于《动手学深度学习》教程，主要使用 **PyTorch** 从零实现各种深度学习算法。

### 核心特点

- 📚 **系统化学习**：从 PyTorch 基础到深度神经网络的完整学习路径
- 📝 **主要学习材料**：**25+ Jupyter Notebook 详细教程**（从 `notebooks/01_basics/` 到 `05_regularization/`）
- 🔧 **辅助 Python 模块**：`src/` 目录包含封装好的可复用代码（模型、损失函数、工具函数）
- 🧮 **从零实现**：手写 BPNN、FFT 等经典算法，深入理解原理
- 🎨 **可视化支持**：丰富的数据可视化和训练过程展示

### 学习方式

**📖 推荐学习顺序**：
1. 打开 `notebooks/` 目录下的 Jupyter Notebook
2. 按编号顺序阅读（01 → 05）
3. 运行代码并观察结果
4. 修改参数进行实验

**💻 `src/` 目录的作用**：
- 封装 Notebook 中常用的工具函数
- 提供标准化的模型和数据加载接口
- 可作为参考实现，但不替代 Notebook 学习

### 内容说明

| 目录 | 内容 | 说明 |
|------|------|------|
| `notebooks/01-05/` | **核心教程**（PyTorch） | 按顺序学习 |
| `notebooks/06_experiments/` | 实验笔记（sklearn） | 早期探索，可选读 |
| `src/machine_learning/` | Python 模块 | 封装好的工具代码 |
| `examples/` | 示例脚本 | 快速参考 |

---

## 📁 项目结构

```
Machine_learning/
├── notebooks/              # Jupyter Notebook 教程（25+个）
│   ├── 01_basics/          # 基础知识（PyTorch张量、梯度）
│   ├── 02_linear_regression/ # 线性回归
│   ├── 03_softmax/         # Softmax 回归与图像分类
│   ├── 04_mlp/             # 多层感知机（神经网络）
│   ├── 05_regularization/  # 正则化技术（Dropout、权重衰减）
│   └── 06_experiments/     # 实验笔记（sklearn算法探索）
│       └── README.md       # 实验笔记说明
│
├── src/                    # Python 源码
│   ├── machine_learning/   # PyTorch 深度学习核心实现
│   │   ├── algorithms/     # 算法（BPNN、FFT等）
│   │   ├── models/         # 模型（线性回归、Softmax、MLP）
│   │   ├── losses/         # 损失函数
│   │   ├── metrics/        # 评估指标
│   │   └── optim/          # 优化器
│   │
│   └── utils/              # 工具模块
│       ├── data_utils.py   # 数据加载
│       ├── training_utils.py # 训练工具
│       └── visualization.py  # 可视化
│
├── examples/               # 示例代码
│   ├── linear_regression_example.py
│   └── mlp_example.py
│
├── tests/                  # 测试代码
│   ├── test_algorithms.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── quick_test.py       # 快速测试脚本
│
├── data/ # 数据集（Fashion-MNIST）
├── docs/ # 项目文档
├── scripts/ # 实用脚本
└── README.md # 本文件
```

## 🚀 快速开始

### 环境要求

- Python >= 3.12
- CUDA >= 11.8 (推荐用于 GPU 加速)

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 初始化项目

```bash
# 使用 just
just init

# 或使用脚本
python scripts/setup.py
```

### 运行 Notebook

```bash
# 使用 just
just notebook

# 或直接启动
jupyter notebook
```

## 📚 学习路线

### 🎯 如何学习

本项目通过 **Jupyter Notebook 交互式教程** 进行学习，建议按以下顺序：

1. 启动 Jupyter Notebook：`just notebook` 或 `jupyter notebook`
2. 进入 `notebooks/` 目录，按编号顺序打开文件（01 → 05）
3. 阅读文档、运行代码、观察结果
4. 尝试修改参数，理解原理
5. 必要时参考 `src/` 中的封装代码

### 📖 阶段一：基础（notebooks/01_basics/）
- PyTorch 张量创建和操作
- 自动微分与梯度计算
- 高级梯度计算技巧

### 📖 阶段二：线性回归（notebooks/02_linear_regression/）
- 线性回归理论
- 从零实现线性回归
- PyTorch 简洁实现

### 📖 阶段三：Softmax 回归（notebooks/03_softmax/）
- Softmax 介绍
- Fashion-MNIST 数据集
- 图像分类实现

### 📖 阶段四：多层感知机（notebooks/04_mlp/）
- 激活函数（ReLU、Sigmoid、Tanh）
- MLP 从零实现
- 模型选择与过拟合

### 📖 阶段五：正则化（notebooks/05_regularization/）
- Dropout 正则化
- 权重衰减（L2正则化）

### 🧪 补充学习（notebooks/06_experiments/）
- 早期实验笔记（使用 scikit-learn）
- 可作为额外参考资料
- 详情参见 [实验说明](notebooks/06_experiments/README.md)

### 💡 辅助代码（src/）

`src/` 目录包含封装好的 Python 模块，用于：
- 在 Notebook 中复用代码
- 参考标准实现
- 快速搭建实验环境

**学习时无需深入阅读，Notebook 会按需导入使用。**

## 📦 辅助模块（src/）

`src/` 目录提供封装好的工具代码，**主要用于支持 Notebook 中的示例**，不是学习的核心内容。以下是模块的主要功能：

### 核心模块

| 模块 | 功能 | 在 Notebook 中的用途 |
|------|------|---------------------|
| `src.machine_learning.models` | 线性回归、Softmax、MLP 模型 | 快速搭建实验模型 |
| `src.machine_learning.losses` | 损失函数（MSE、交叉熵等） | 训练时使用 |
| `src.machine_learning.metrics` | 准确率、F1分数等指标 | 模型评估 |
| `src.machine_learning.optim` | 学习率调度器 | 训练优化 |
| `src.utils.data_utils` | 数据加载（Fashion-MNIST等） | 数据预处理 |
| `src.utils.training_utils` | 训练循环、Animator | 训练可视化 |
| `src.utils.visualization` | 绘图函数 | 结果展示 |

### 示例用法

Notebook 中会用到这些模块：

```python
from src.utils import load_fashion_mnist, train_epoch, Animator

# 加载数据
train_iter, test_iter = load_fashion_mnist(batch_size=256)

# 训练可视化
animator = Animator(xlabel='epoch', legend=['train loss', 'train acc'])
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
    animator.add(epoch + 1, (train_loss, train_acc))
```

### 🔧 开发工具

本项目使用 [just](https://github.com/casey/just) 作为任务运行器：
train_iter, test_iter = load_fashion_mnist(batch_size=256)

# 创建合成回归数据
data = SyntheticRegressionData(
    w=torch.tensor([2.0, -3.4]), 
    b=4.2, 
    num_examples=1000
)
train_loader = data.get_dataloader(train=True)

# 使用训练工具
animator = Animator(
    xlabel='epoch', 
    legend=['train loss', 'train acc', 'test acc']
)

# 训练模型
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
    animator.add(epoch + 1, (train_loss, train_acc, test_acc))
```

## 🛠️ 开发工具

本项目使用 [just](https://github.com/casey/just) 作为任务运行器：

```bash
# 查看所有命令
just --list

# 常用命令
just init       # 初始化项目环境
just test       # 运行测试
just format     # 格式化代码
just lint       # 代码检查
just fix        # 自动修复代码问题
just clean      # 清理临时文件
just check      # 项目健康检查
just notebook   # 启动 Jupyter Notebook
just stats      # 代码统计
```

## 🧪 测试

```bash
# 运行所有测试
just test
# 或
python test.py

# 运行特定测试
python -c "from test import test_fft; test_fft()"
```

测试内容包括：
- ✅ PyTorch 线性回归
- ✅ 排序算法（冒泡、归并）
- ✅ 反向传播神经网络（BPNN）
- ✅ FFT 算法（递归、矩阵、迭代）
- ✅ 数据加载工具

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

快速开始：

```bash
# Fork 并克隆项目
git clone https://github.com/yourusername/machine-learning.git
cd machine-learning

# 创建功能分支
git checkout -b feature/amazing-feature

# 提交更改
git commit -m "feat: add amazing feature"

# 推送并创建 PR
git push origin feature/amazing-feature
```

## 📄 项目规范

### 代码风格

- 使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式化和检查
- 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 提交规范
- 使用类型注解
- 所有公共函数和类必须包含文档字符串

### 提交类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关

## 📚 参考资料

- [《动手学深度学习》(Dive into Deep Learning)](https://zh.d2l.ai/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [scikit-learn 文档](https://scikit-learn.org/)

## 📖 相关文档

- [PyTorch 适用范围概述](docs/overview.md)
- [更新日志](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md)

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

## 🙏 致谢

感谢《动手学深度学习》作者李沐等人为机器学习教育做出的贡献。

---

**Star ⭐ 这个项目，如果它对你有帮助！**
