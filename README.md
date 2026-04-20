# 机器学习项目

基于 PyTorch 的深度学习学习项目，跟随《动手学深度学习》教程实现。

## 📁 项目结构

```
Machine_learning/
├── notebooks/                  # Jupyter Notebook 教程
│   ├── 01_basics/              # 基础知识
│   │   ├── 01_tensor_creation.ipynb
│   │   ├── 02_tensor_operations.ipynb
│   │   ├── 03_gradient_computation.ipynb
│   │   └── 04_gradient_advanced.ipynb
│   │
│   ├── 02_linear_regression/   # 线性回归
│   │   ├── 01_linear_regression_theory.ipynb
│   │   ├── 02_linear_regression_from_scratch.ipynb
│   │   ├── 03_linear_regression_detailed.ipynb
│   │   ├── 04_linear_regression_concise.ipynb
│   │   └── 05_linear_regression_pytorch.ipynb
│   │
│   ├── 03_softmax/             # Softmax 回归
│   │   ├── 01_softmax_introduction.ipynb
│   │   ├── 02_fashion_mnist_dataset.ipynb
│   │   ├── 03_softmax_from_scratch.ipynb
│   │   ├── 04_softmax_regression_full.ipynb
│   │   └── 05_softmax_concise.ipynb
│   │
│   ├── 04_mlp/                 # 多层感知机
│   │   ├── 01_activation_functions.ipynb
│   │   ├── 02_mlp_from_scratch.ipynb
│   │   ├── 03_mlp_detailed.ipynb
│   │   ├── 04_mlp_concise.ipynb
│   │   └── 05_model_selection.ipynb
│   │
│   ├── 05_regularization/      # 正则化技术
│   │   ├── 01_dropout.ipynb
│   │   └── 02_weight_decay.ipynb
│   │
│   └── 06_experiments/         # 实验笔记
│       ├── 01_one_hot_encoding.ipynb
│       ├── 02_scratch_experiment.ipynb
│       ├── 03_scratch_experiment1.ipynb
│       └── 04_scratch_experiment2.ipynb
│
├── src/                        # Python 源码
│   ├── machine_learning/       # 主模块
│   │   ├── algorithms/         # 算法实现
│   │   │   ├── __init__.py
│   │   │   ├── fft.py         # 快速傅里叶变换
│   │   │   ├── neural_networks.py  # 神经网络
│   │   │   └── sorting.py     # 排序算法
│   │   ├── __init__.py
│   │   └── py.typed
│   │
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       ├── data_utils.py      # 数据加载工具
│       ├── training_utils.py  # 训练工具
│       └── visualization.py   # 可视化工具
│
├── assets/                     # 静态资源
│   └── images/                 # 图片资源
│
├── data/                       # 数据集
├── data1/                      # 附加数据
├── test.py                     # 测试脚本
├── pyproject.toml              # 项目配置
├── .gitignore                  # Git 忽略配置
└── README.md                   # 项目说明
```
Machine_learning/
├── notebooks/              # Jupyter Notebook 教程
│   ├── 01_basics/         # 基础知识
│   │   ├── 01_tensor_creation.ipynb
│   │   ├── 02_tensor_operations.ipynb
│   │   ├── 03_gradient_computation.ipynb
│   │   └── 04_gradient_advanced.ipynb
│   │
│   ├── 02_linear_regression/    # 线性回归
│   │   ├── 01_linear_regression_theory.ipynb
│   │   ├── 02_linear_regression_from_scratch.ipynb
│   │   ├── 03_linear_regression_detailed.ipynb
│   │   ├── 04_linear_regression_concise.ipynb
│   │   └── 05_linear_regression_pytorch.ipynb
│   │
│   ├── 03_softmax/        # Softmax 回归
│   │   ├── 01_softmax_introduction.ipynb
│   │   ├── 02_fashion_mnist_dataset.ipynb
│   │   ├── 03_softmax_from_scratch.ipynb
│   │   ├── 04_softmax_regression_full.ipynb
│   │   └── 05_softmax_concise.ipynb
│   │
│   ├── 04_mlp/            # 多层感知机
│   │   ├── 01_activation_functions.ipynb
│   │   ├── 02_mlp_from_scratch.ipynb
│   │   ├── 03_mlp_detailed.ipynb
│   │   ├── 04_mlp_concise.ipynb
│   │   └── 05_model_selection.ipynb
│   │
│   ├── 05_regularization/ # 正则化技术
│   │   ├── 01_dropout.ipynb
│   │   └── 02_weight_decay.ipynb
│   │
│   └── 06_experiments/    # 实验笔记
│       ├── 01_one_hot_encoding.ipynb
│       ├── 02_scratch_experiment.ipynb
│       ├── 03_scratch_experiment1.ipynb
│       └── 04_scratch_experiment2.ipynb
│
├── src/                   # Python 源码
│   ├── machine_learning/
│   │   ├── __init__.py
│   │   └── py.typed
│   └── utils/             # 工具函数
│
├── assets/                # 静态资源
│   ├── images/           # 图片资源
│   └── data/             # 数据文件
│
├── data/                  # 数据集
├── data1/                 # 附加数据
├── test.py               # 测试脚本
├── pyproject.toml        # 项目配置
├── .gitignore            # Git 忽略配置
└── README.md             # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python >= 3.12
- CUDA >= 11.8 (推荐用于 GPU 加速)

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

### 运行 Notebook

```bash
jupyter notebook
```

## 📚 学习路线

1. **基础知识** (notebooks/01_basics/)
   - PyTorch 张量创建
   - 张量操作
   - 自动微分与梯度计算

2. **线性回归** (notebooks/02_linear_regression/)
   - 线性回归理论
   - 从零实现线性回归
   - PyTorch 简洁实现

3. **Softmax 回归** (notebooks/03_softmax/)
   - Softmax 介绍
   - Fashion-MNIST 数据集
   - 图像分类实现

4. **多层感知机 (MLP)** (notebooks/04_mlp/)
   - 激活函数
   - MLP 从零实现
   - 模型选择与过拟合

5. **正则化技术** (notebooks/05_regularization/)
   - Dropout
   - 权重衰减

## 🔧 主要依赖

- PyTorch >= 2.7.1
- d2l >= 0.17.0 (动手学深度学习库)
- matplotlib >= 3.10.8
- numpy >= 2.4.3
- pandas >= 3.0.1

## 📦 模块使用

### 算法模块

```python
from src.machine_learning import BPNN, fft, merge_sort

# 使用反向传播神经网络
model = BPNN(input_size=2, hidden_size=4, output_size=1)
model.train(X, y, lr=0.5, epochs=1000)

# 使用FFT
import torch
x = torch.randn(8)
result = fft(x)

# 使用排序算法
sorted_arr = merge_sort([3, 1, 4, 1, 5, 9, 2, 6])
```

### 工具模块

```python
from src.utils import (
    load_fashion_mnist,
    SyntheticRegressionData,
    train_epoch,
    evaluate_accuracy,
    Animator
)

# 加载Fashion-MNIST数据
train_iter, test_iter = load_fashion_mnist(batch_size=256)

# 创建合成回归数据
data = SyntheticRegressionData(w=torch.tensor([2.0]), b=4.2)
train_loader = data.get_dataloader(train=True)

# 使用训练工具
animator = Animator(xlabel='epoch', legend=['train loss', 'train acc'])
```

## 🧪 运行测试

```bash
python test.py
```

这将运行所有模块的测试，包括：
- PyTorch 线性回归
- 排序算法
- 反向传播神经网络
- FFT 算法
- 数据工具

## 📝 参考资料

- 《动手学深度学习》(Dive into Deep Learning)
- PyTorch 官方文档

## 📄 许可证

MIT License
