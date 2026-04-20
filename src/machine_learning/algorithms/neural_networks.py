"""
神经网络算法实现
"""

import numpy as np


class BPNN:
    """
    反向传播神经网络 (Back Propagation Neural Network)
    
    使用NumPy实现的多层感知机，支持自定义隐藏层大小
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化神经网络
        
        Args:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        
        # 初始化偏置
        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.output_size)
        
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid导数"""
        return x * (1 - x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, input_size)
            
        Returns:
            输出预测，形状为 (batch_size, output_size)
        """
        # 第一层
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, x: np.ndarray, y: np.ndarray, 
                 y_pred: np.ndarray, lr: float):
        """
        反向传播
        
        Args:
            x: 输入数据
            y: 真实标签
            y_pred: 预测输出
            lr: 学习率
        """
        m = x.shape[0]  # 样本数
        
        # 输出层梯度
        dz2 = y_pred - y
        self.W2_gradient = np.dot(self.a1.T, dz2) / m
        self.b2_gradient = np.sum(dz2, axis=0) / m
        
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        self.W1_gradient = np.dot(x.T, dz1) / m
        self.b1_gradient = np.sum(dz1, axis=0) / m
        
        # 更新权重和偏置
        self.W2 -= lr * self.W2_gradient
        self.W1 -= lr * self.W1_gradient
        self.b2 -= lr * self.b2_gradient
        self.b1 -= lr * self.b1_gradient
    
    def train(self, x: np.ndarray, y: np.ndarray, 
              lr: float = 0.1, epochs: int = 1000, 
              verbose: bool = False):
        """
        训练模型
        
        Args:
            x: 训练数据
            y: 训练标签
            lr: 学习率
            epochs: 训练轮数
            verbose: 是否打印训练进度
        """
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y, y_pred, lr)
            
            if verbose and (epoch + 1) % 100 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测输出
        
        Args:
            x: 输入数据
            
        Returns:
            预测结果
        """
        return self.forward(x)
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算均方误差损失
        
        Args:
            x: 输入数据
            y: 真实标签
            
        Returns:
            均方误差
        """
        y_pred = self.forward(x)
        return np.mean((y_pred - y) ** 2)
