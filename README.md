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

本项目是一个面向机器学习和深度学习初学者的学习仓库，主要特点：

- 📚 **系统化学习**：从基础到进阶的完整学习路径
- 📝 **丰富的笔记**：25+ Jupyter Notebook 详细教程
- 🔧 **实用工具**：封装的数据加载、训练和可视化工具
- 🧮 **算法实现**：从零实现经典算法（BPNN、FFT等）
- 🎨 **可视化支持**：内置多种数据可视化方法

## 📁 项目结构

```
Machine_learning/
├── notebooks/                  # Jupyter Notebook 教程（25个）
│   ├── 01_basics/              # 基础知识（4个）
│   │   ├── 01_tensor_creation.ipynb
│   │   ├── 02_tensor_operations.ipynb
│   │   ├── 03_gradient_computation.ipynb
│   │   └── 04_gradient_advanced.ipynb
│   │
│   ├── 02_linear_regression/   # 线性回归（5个）
│   │   ├── 01_linear_regression_theory.ipynb
│   │   ├── 02_linear_regression_from_scratch.ipynb
│   │   ├── 03_linear_regression_detailed.ipynb
│   │   ├── 04_linear_regression_concise.ipynb
│   │   └── 05_linear_regression_pytorch.ipynb
│   │
│   ├── 03_softmax/             # Softmax 回归（5个）
│   │   ├── 01_softmax_introduction.ipynb
│   │   ├── 02_fashion_mnist_dataset.ipynb
│   │   ├── 03_softmax_from_scratch.ipynb
│   │   ├── 04_softmax_regression_full.ipynb
│   │   └── 05_softmax_concise.ipynb
│   │
│   ├── 04_mlp/                 # 多层感知机（5个）
│   │   ├── 01_activation_functions.ipynb
│   │   ├── 02_mlp_from_scratch.ipynb
│   │   ├── 03_mlp_detailed.ipynb
│   │   ├── 04_mlp_concise.ipynb
│   │   └── 05_model_selection.ipynb
│   │
│   ├── 05_regularization/      # 正则化技术（2个）
│   │   ├── 01_dropout.ipynb
│   │   └── 02_weight_decay.ipynb
│   │
│   └── 06_experiments/         # 实验笔记（4个）
│       ├── 01_one_hot_encoding.ipynb
│       ├── 02_scratch_experiment.ipynb
│       ├── 03_scratch_experiment1.ipynb
│       └── 04_scratch_experiment2.ipynb
│
├── src/                        # Python 源码（模块化设计）
│   ├── machine_learning/       # 主模块
│   │   ├── algorithms/         # 算法实现
│   │   │   ├── fft.py         # 快速傅里叶变换（5种实现）
│   │   │   ├── neural_networks.py  # 神经网络（BPNN）
│   │   │   └── sorting.py     # 排序算法（冒泡、归并）
│   │   ├── __init__.py
│   │   └── py.typed
│   │
│   └── utils/                  # 工具模块
│       ├── data_utils.py      # 数据加载工具
│       ├── training_utils.py  # 训练工具（Animator、Accumulator）
│       └── visualization.py   # 可视化工具
│
├── scripts/                    # 实用脚本
│   ├── setup.py               # 环境设置
│   ├── clean.py               # 清理临时文件
│   └── check_project.py       # 项目健康检查
│
├── docs/                       # 项目文档
│   ├── overview.md            # PyTorch 适用范围概述
│   ├── README.md              # 文档说明
│   └── 数学.pdf                # 数学基础
│
├── assets/                     # 静态资源
│   └── images/                 # 图片资源（激活函数图等）
│
├── data/                       # 数据集（Fashion-MNIST）
├── data1/                      # 附加数据
│
├── test.py                     # 测试脚本
├── justfile                    # 任务运行器配置
├── pyproject.toml              # 项目配置（元数据、依赖）
├── CHANGELOG.md                # 更新日志
├── CONTRIBUTING.md             # 贡献指南
├── .gitignore                  # Git 忽略配置
└── README.md                   # 本文件
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

### 第一阶段：基础（notebooks/01_basics/）
- PyTorch 张量创建和操作
- 自动微分与梯度计算
- 高级梯度计算技巧

### 第二阶段：线性回归（notebooks/02_linear_regression/）
- 线性回归理论
- 从零实现线性回归
- PyTorch 简洁实现

### 第三阶段：Softmax 回归（notebooks/03_softmax/）
- Softmax 介绍
- Fashion-MNIST 数据集
- 图像分类实现

### 第四阶段：多层感知机（notebooks/04_mlp/）
- 激活函数（ReLU、Sigmoid、Tanh）
- MLP 从零实现
- 模型选择与过拟合

### 第五阶段：正则化（notebooks/05_regularization/）
- Dropout 正则化
- 权重衰减（L2正则化）

## 📦 模块使用

### 算法模块

```python
from src.machine_learning import BPNN, fft, merge_sort

# 使用反向传播神经网络
model = BPNN(input_size=2, hidden_size=4, output_size=1)
model.train(X, y, lr=0.5, epochs=1000)
predictions = model.predict(X_test)

# 使用FFT（5种实现方式）
import torch
x = torch.randn(8)
result = fft(x)           # 递归实现
result = fft_matrix(x)    # 矩阵实现
result = fft_iter(x)      # 迭代实现

# 使用排序算法
sorted_arr = merge_sort([3, 1, 4, 1, 5, 9, 2, 6])
sorted_arr = bubble_sort([64, 34, 25, 12, 22, 11, 90])
```

### 工具模块

```python
from src.utils import (
    load_fashion_mnist,
    SyntheticRegressionData,
    train_epoch,
    evaluate_accuracy,
    Animator,
    plot_images
)

# 加载Fashion-MNIST数据
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
