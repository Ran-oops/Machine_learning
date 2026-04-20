"""
Softmax 回归模型实现
"""

import torch
from torch import nn


class SoftmaxRegression(nn.Module):
    """
    Softmax 回归模型（多类逻辑回归）
    
    用于多分类任务的线性模型
    
    Attributes:
        flatten: 展平层（处理图像输入）
        fc: 全连接层
    """
    
    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        """
        初始化 Softmax 回归模型
        
        Args:
            input_dim: 输入特征维度
            output_dim: 类别数量
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            未归一化的 logits
            （在使用 CrossEntropyLoss 时，不需要手动应用 softmax）
        """
        x = self.flatten(x)
        return self.fc(x)


class SoftmaxRegressionScratch:
    """
    从零实现的 Softmax 回归
    
    不使用 PyTorch nn.Module，展示底层实现
    """
    
    def __init__(self, num_inputs: int, num_outputs: int, lr: float = 0.1, sigma: float = 0.01):
        """
        初始化模型参数
        
        Args:
            num_inputs: 输入特征数
            num_outputs: 输出类别数
            lr: 学习率
            sigma: 权重初始化标准差
        """
        self.lr = lr
        self.w = torch.normal(0, sigma, (num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
        self.params = [self.w, self.b]
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            X: 输入张量
            
        Returns:
            softmax 概率分布
        """
        X = X.reshape((-1, self.w.shape[0]))
        return self.softmax(torch.matmul(X, self.w) + self.b)
    
    @staticmethod
    def softmax(X: torch.Tensor) -> torch.Tensor:
        """
        Softmax 函数
        
        Args:
            X: 输入 logits
            
        Returns:
            概率分布
        """
        X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition
    
    def cross_entropy(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        交叉熵损失
        
        Args:
            y_hat: 预测概率
            y: 真实标签索引
            
        Returns:
            损失值
        """
        return -torch.log(y_hat[range(len(y_hat)), y])
