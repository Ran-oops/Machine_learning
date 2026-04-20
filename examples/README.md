# 示例代码

本目录包含使用 `machine_learning` 模块的各种示例代码。

## 📁 文件说明

| 文件 | 描述 |
|------|------|
| `linear_regression_example.py` | 线性回归完整示例（PyTorch版 + Scratch版） |
| `mlp_example.py` | 多层感知机分类示例 |
| `softmax_example.py` | Softmax 回归示例（待添加） |

## 🚀 快速开始

### 运行线性回归示例

```bash
cd examples
python linear_regression_example.py
```

**输出示例：**
```
============================================================
示例 1: PyTorch Linear Regression
============================================================

1. 准备数据...
2. 创建模型...
3. 训练模型...
  Epoch [20/100], Loss: 0.0234
  Epoch [40/100], Loss: 0.0156
  ...
4. 评估模型...
  真实参数: w=2.0000, b=3.0000
  学习参数: w=1.9876, b=2.9876

5. 测试预测...
  输入 x=5.0
  预测值: 12.9256
  真实值: 13.0000
```

### 运行 MLP 示例

```bash
cd examples
python mlp_example.py
```

**输出示例：**
```
============================================================
示例 1: Basic MLP Training
============================================================

1. 加载数据...
2. 创建模型...
   模型结构:
   - 输入维度: 784
   - 隐藏维度: 256
   - 输出维度: 10
   - Dropout: 0.2

3. 训练模型（10 epochs）...
   Epoch [2/10] Loss: 0.5234, Train Acc: 0.8234, Test Acc: 0.8123
   Epoch [4/10] Loss: 0.3456, Train Acc: 0.8765, Test Acc: 0.8654
   ...

4. 最终测试准确率: 0.8845

5. 绘制训练曲线...
   图表已保存到: mlp_training.png
```

## 📊 生成的图表

运行示例后会生成以下图表文件：

- `linear_regression_example.png` - 线性回归训练过程和拟合结果
- `mlp_training.png` - MLP 训练曲线
- `activation_comparison.png` - 不同激活函数性能对比

## 📝 示例说明

### 线性回归示例

展示了两种实现方式：

1. **PyTorch 版本** - 使用 `nn.Module` 的标准实现
2. **Scratch 版本** - 从零实现，展示底层原理

涵盖内容：
- 数据生成
- 模型定义
- 训练循环
- 参数更新
- 可视化

### MLP 示例

展示多层感知机的完整训练流程：

1. **基础 MLP** - 单隐藏层网络
2. **简洁 MLP** - 多隐藏层网络
3. **激活函数对比** - ReLU vs Sigmoid vs Tanh

涵盖内容：
- Fashion-MNIST 数据加载
- 模型配置（激活函数、Dropout）
- 训练和评估
- 过拟合分析

## 🔧 自定义示例

您可以基于这些示例创建自己的实验：

```python
import sys
sys.path.insert(0, '../src')

from machine_learning.models import MLP
from machine_learning.losses import cross_entropy_loss
from machine_learning.metrics import accuracy

# 您的代码...
```

## 📚 相关文档

- [项目主页](../README.md)
- [API 文档](../docs/)
- [源代码](../src/)
