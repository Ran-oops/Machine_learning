"""
线性回归模型实现
"""

import torch
from torch import nn


class LinearRegression(nn.Module):
    """
    线性回归模型
    
    简单的线性回归实现，包含权重和偏置
    
    Attributes:
        weight: 权重参数
        bias: 偏置参数
    """
    
    def __init__(self, input_dim: int):
        """
        初始化线性回归模型
        
        Args:
            input_dim: 输入特征维度
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            输出预测，形状为 (batch_size, 1)
        """
        return torch.matmul(x, self.weight) + self.bias


class LinearRegressionScratch:
    """
    从零实现的线性回归（不使用 PyTorch nn.Module）
    
    用于教学目的，展示底层实现细节
    """
    
    def __init__(self, input_dim: int, lr: float = 0.03, sigma: float = 0.01):
        """
        初始化模型参数
        
        Args:
            input_dim: 输入特征维度
            lr: 学习率
            sigma: 权重初始化标准差
        """
        self.lr = lr
        self.w = torch.normal(0, sigma, (input_dim, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return torch.matmul(X, self.w) + self.b
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算均方误差损失"""
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    
    def configure_optimizers(self):
        """配置优化器（SGD）"""
        return torch.optim.SGD([self.w, self.b], lr=self.lr)


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    均方误差损失函数
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
