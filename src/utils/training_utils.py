"""
训练相关工具函数
"""

import torch
from torch import nn
from IPython import display
import matplotlib.pyplot as plt
import numpy as np


class Animator:
    """
    在动画中绘制数据
    
    用于Jupyter Notebook中实时显示训练过程
    """
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, 
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """
        Args:
            xlabel: x轴标签
            ylabel: y轴标签
            legend: 图例列表
            xlim: x轴范围
            ylim: y轴范围
            xscale: x轴刻度类型
            yscale: y轴刻度类型
            fmts: 线条格式列表
            nrows: 子图行数
            ncols: 子图列数
            figsize: 图表大小
        """
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        """
        向图表中添加多个数据点
        
        Args:
            x: x坐标或x坐标列表
            y: y坐标或y坐标列表
        """
        # 判断y是否包含多个列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        # 判断x是否包含多个列表
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(float(a))
                self.Y[i].append(float(b))
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_epoch(net: nn.Module, train_iter, loss_fn, updater):
    """
    训练模型一个迭代周期
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss_fn: 损失函数
        updater: 优化器
        
    Returns:
        (train_loss, train_acc): 训练损失和准确率
    """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    
    # 损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            loss.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            loss.sum().backward()
            updater(X.shape[0])
            
        metric.add(float(loss.sum()), accuracy(y_hat, y), y.numel())
        
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_accuracy(net: nn.Module, data_iter):
    """
    计算模型在指定数据集上的精度
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        
    Returns:
        准确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
    
    metric = Accumulator(2)  # 正确预测数、预测总数
    
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            
    return metric[0] / metric[1]


def evaluate_loss(net: nn.Module, data_iter, loss_fn):
    """
    计算模型在指定数据集上的损失
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        loss_fn: 损失函数
        
    Returns:
        平均损失
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    metric = Accumulator(2)  # 损失总和、样本数
    
    with torch.no_grad():
        for X, y in data_iter:
            loss = loss_fn(net(X), y)
            metric.add(float(loss.sum()), y.numel())
            
    return metric[0] / metric[1]


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算预测正确的数量
    
    Args:
        y_hat: 预测结果，形状为 (batch_size, num_classes)
        y: 真实标签，形状为 (batch_size,)
        
    Returns:
        预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """
    在n个变量上累加
    """
    
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(net: nn.Module, train_iter, test_iter, loss_fn, num_epochs, 
          updater, plot=True):
    """
    训练模型
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss_fn: 损失函数
        num_epochs: 训练轮数
        updater: 优化器
        plot: 是否绘制训练过程
        
    Returns:
        (train_losses, train_accs, test_accs): 训练历史
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss_fn, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        
        train_losses.append(train_metrics[0])
        train_accs.append(train_metrics[1])
        test_accs.append(test_acc)
        
        if plot:
            animator.add(epoch + 1, train_metrics + (test_acc,))
            
    train_loss, train_acc = train_metrics
    
    if not plot:
        print(f'最终损失 {train_loss:.3f}, 训练准确率 {train_acc:.3f}, '
              f'测试准确率 {test_acc:.3f}')
              
    return train_losses, train_accs, test_accs


def predict(net: nn.Module, test_iter, n: int = 9):
    """
    预测标签并可视化前n个样本
    
    Args:
        net: 神经网络模型
        test_iter: 测试数据迭代器
        n: 显示的样本数
    """
    from .data_utils import get_fashion_mnist_labels
    from .visualization import plot_images
    
    for X, y in test_iter:
        break
    
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [f'true: {t}\npred: {p}' for t, p in zip(trues, preds)]
    plot_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
