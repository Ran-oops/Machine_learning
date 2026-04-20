# 机器学习项目

基于 PyTorch 的深度学习学习项目，跟随《动手学深度学习》教程实现。

## 📁 项目结构

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

## 📝 参考资料

- 《动手学深度学习》(Dive into Deep Learning)
- PyTorch 官方文档

## 📄 许可证

MIT License
