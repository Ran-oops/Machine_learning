"""
多层感知机 (MLP) 实现
"""

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    多层感知机模型
    
    包含输入层、隐藏层和输出层的全连接神经网络
    
    Attributes:
        flatten: 展平层
        fc1: 第一个全连接层
        fc2: 第二个全连接层
    """
    
    def __init__(
        self, 
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        初始化 MLP 模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
            dropout: Dropout 比率
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, name: str):
        """获取激活函数"""
        activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        return activations.get(name, F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出 logits
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def init_weights(self, std: float = 0.01):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SimpleMLP(nn.Module):
    """
    简洁版 MLP
    
    使用 Sequential 的简洁实现
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [256, 128],
        output_dim: int = 10,
        activation: str = "relu"
    ):
        """
        初始化简洁版 MLP
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            activation: 激活函数类型
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation_layer(activation)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(
            nn.Flatten(),
            *layers
        )
    
    def _get_activation_layer(self, name: str):
        """获取激活函数层"""
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)
