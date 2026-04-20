"""
数据加载和处理工具函数
"""

import torch
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np


def get_dataloader_workers():
    """
    获取数据加载器的工作进程数
    
    在非交互式环境中使用4个工作进程
    在交互式环境（Jupyter）中使用0个工作进程
    """
    try:
        # 检测是否在Jupyter环境中
        shell = get_ipython().__class__.__name__
        if shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']:
            return 0
    except NameError:
        pass
    return 4


def load_fashion_mnist(batch_size: int, resize: tuple = None) -> tuple:
    """
    加载Fashion-MNIST数据集
    
    Args:
        batch_size: 批次大小
        resize: 可选的图像大小调整，例如 (224, 224)
        
    Returns:
        (train_iter, test_iter): 训练集和测试集的数据迭代器
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    mnist_train = datasets.FashionMNIST(
        root="./data", 
        train=True, 
        transform=trans, 
        download=True
    )
    mnist_test = datasets.FashionMNIST(
        root="./data", 
        train=False, 
        transform=trans, 
        download=True
    )
    
    num_workers = get_dataloader_workers()
    
    train_iter = data.DataLoader(
        mnist_train, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_iter = data.DataLoader(
        mnist_test, 
        batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_iter, test_iter


def get_fashion_mnist_labels(labels: torch.Tensor) -> list:
    """
    返回Fashion-MNIST数据集的文本标签
    
    Args:
        labels: 数字标签的张量
        
    Returns:
        对应的文本标签列表
    """
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


class SyntheticRegressionData:
    """
    合成线性回归数据集
    
    用于生成用于回归任务的人工数据
    """
    
    def __init__(
        self, 
        w: torch.Tensor, 
        b: float, 
        num_examples: int = 1000,
        noise: float = 0.01, 
        num_inputs: int = 2,
        batch_size: int = 10
    ):
        """
        Args:
            w: 真实的权重向量
            b: 真实的偏置项
            num_examples: 样本数量
            noise: 噪声标准差
            num_inputs: 输入特征数量
            batch_size: 批次大小
        """
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        
        # 生成特征
        self.X = torch.randn(num_examples, num_inputs)
        
        # 生成带有噪声的标签
        self.y = torch.matmul(self.X, w) + b
        self.y += torch.randn(num_examples, 1) * noise
        
        self.y = self.y.reshape((-1, 1))
        
    def get_dataloader(self, train: bool = True):
        """
        获取数据加载器
        
        Args:
            train: 是否为训练集
            
        Returns:
            数据迭代器
        """
        if train:
            indices = list(range(0, len(self.y)))
            # 随机打乱
            np.random.shuffle(indices)
        else:
            # 测试集不打乱
            indices = list(range(len(self.y)))
            
        num_workers = get_dataloader_workers()
        
        # 创建自定义数据集
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        
        return data.DataLoader(
            dataset, 
            self.batch_size, 
            shuffle=train, 
            num_workers=num_workers
        )


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor):
    """
    简单数据迭代器（不使用DataLoader）
    
    Args:
        batch_size: 批次大小
        features: 特征张量
        labels: 标签张量
        
    Yields:
        (X, y): 特征和标签的批次
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]
