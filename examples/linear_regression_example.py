"""
线性回归示例

展示如何使用项目中的线性回归模型
"""

import torch
from torch import nn
import matplotlib.pyplot as plt

# 导入项目模块
import sys
sys.path.insert(0, '../src')

from machine_learning.models import LinearRegression, LinearRegressionScratch
from machine_learning.losses import squared_loss
from machine_learning.metrics import mse, r2_score
from utils.data_utils import SyntheticRegressionData


def example_1_pytorch_version():
    """
    示例 1: 使用 PyTorch 实现的线性回归
    """
    print("=" * 60)
    print("示例 1: PyTorch Linear Regression")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    w_true = torch.tensor([2.0])
    b_true = 3.0
    data = SyntheticRegressionData(
        w=w_true, 
        b=b_true, 
        num_examples=100,
        noise=0.1
    )
    
    # 2. 创建模型
    print("2. 创建模型...")
    model = LinearRegression(input_dim=1)
    
    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    
    # 4. 训练
    print("3. 训练模型...")
    losses = []
    for epoch in range(100):
        train_iter = data.get_dataloader(train=True)
        
        for X, y in train_iter:
            # 前向传播
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
            losses.append(loss.item())
    
    # 5. 评估
    print("\n4. 评估模型...")
    print(f"  真实参数: w={w_true.item():.4f}, b={b_true:.4f}")
    print(f"  学习参数: w={model.weight.item():.4f}, b={model.bias.item():.4f}")
    
    # 6. 测试
    print("\n5. 测试预测...")
    x_test = torch.tensor([[5.0]])
    y_pred = model(x_test)
    y_true = w_true * x_test + b_true
    print(f"  输入 x=5.0")
    print(f"  预测值: {y_pred.item():.4f}")
    print(f"  真实值: {y_true.item():.4f}")
    
    print("\n" + "=" * 60 + "\n")


def example_2_scratch_version():
    """
    示例 2: 从零实现的线性回归
    """
    print("=" * 60)
    print("示例 2: Scratch Linear Regression")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    w_true = torch.tensor([2.0])
    b_true = 3.0
    data = SyntheticRegressionData(
        w=w_true, 
        b=b_true, 
        num_examples=100,
        noise=0.1
    )
    
    # 2. 创建模型
    print("2. 创建模型...")
    model = LinearRegressionScratch(input_dim=1, lr=0.03)
    
    # 3. 训练
    print("3. 训练模型...")
    for epoch in range(100):
        train_iter = data.get_dataloader(train=True)
        
        for X, y in train_iter:
            # 前向传播
            y_pred = model.forward(X)
            loss = model.loss(y_pred, y)
            
            # 反向传播（手动）
            loss.sum().backward()
            
            # 更新参数（手动）
            with torch.no_grad():
                model.w -= model.lr * model.w.grad / len(X)
                model.b -= model.lr * model.b.grad / len(X)
                model.w.grad.zero_()
                model.b.grad.zero_()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/100], Loss: {loss.mean().item():.4f}")
    
    # 4. 评估
    print("\n4. 评估模型...")
    print(f"  真实参数: w={w_true.item():.4f}, b={b_true:.4f}")
    print(f"  学习参数: w={model.w.item():.4f}, b={model.b.item():.4f}")
    
    print("\n" + "=" * 60 + "\n")


def example_3_visualization():
    """
    示例 3: 可视化训练过程
    """
    print("=" * 60)
    print("示例 3: Visualization")
    print("=" * 60)
    
    # 准备数据
    w_true = torch.tensor([2.0])
    b_true = 3.0
    data = SyntheticRegressionData(
        w=w_true, 
        b=b_true, 
        num_examples=100,
        noise=0.5
    )
    
    # 创建模型
    model = LinearRegression(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    
    # 训练并记录
    losses = []
    for epoch in range(200):
        train_iter = data.get_dataloader(train=True)
        
        for X, y in train_iter:
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # 数据与拟合线
    plt.subplot(1, 2, 2)
    X_plot = data.X.numpy()
    y_plot = data.y.numpy()
    plt.scatter(X_plot, y_plot, alpha=0.5, label='Data')
    
    # 拟合线
    x_line = torch.linspace(X_plot.min(), X_plot.max(), 100).reshape(-1, 1)
    y_line = model(x_line).detach().numpy()
    plt.plot(x_line.numpy(), y_line, 'r-', label='Fitted Line', linewidth=2)
    
    # 真实线
    y_true = w_true * x_line + b_true
    plt.plot(x_line.numpy(), y_true, 'g--', label='True Line', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_example.png', dpi=150)
    print("\n可视化已保存到: linear_regression_example.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 运行示例
    example_1_pytorch_version()
    example_2_scratch_version()
    example_3_visualization()
    
    print("\n✅ 所有示例运行完成！")
