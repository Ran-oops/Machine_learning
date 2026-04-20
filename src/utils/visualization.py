"""
可视化工具函数
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib_inline import backend_inline


def set_figure_size(figsize: tuple = (3.5, 2.5)):
    """
    设置matplotlib的图表大小
    
    Args:
        figsize: 图表大小元组 (宽度, 高度)
    """
    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def plot_images(imgs: torch.Tensor, num_rows: int, num_cols: int, 
                titles: list = None, scale: float = 2):
    """
    绘制图像网格
    
    Args:
        imgs: 图像张量，形状为 (N, C, H, W) 或 (N, H, W)
        num_rows: 行数
        num_cols: 列数
        titles: 可选的标题列表
        scale: 图像缩放比例
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 张量图像，转换为numpy并调整维度
            img = img.numpy()
            if img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
        else:
            # PIL图像
            img = np.array(img)
            
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        if titles:
            ax.set_title(titles[i])
            
    plt.tight_layout()
    plt.show()


def plot_metrics(train_metrics: list, test_metrics: list = None,
                 xlabel: str = 'epoch', ylabel: str = 'loss',
                 legend: list = None):
    """
    绘制训练指标曲线
    
    Args:
        train_metrics: 训练指标列表
        test_metrics: 测试指标列表（可选）
        xlabel: x轴标签
        ylabel: y轴标签
        legend: 图例列表
    """
    set_figure_size()
    
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'b-', label='train')
    
    if test_metrics:
        plt.plot(epochs, test_metrics, 'r--', label='test')
        
    if legend:
        plt.legend(legend)
    else:
        plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_activation_functions(x: torch.Tensor = None):
    """
    绘制常用激活函数
    
    Args:
        x: 输入值张量，如果为None则自动生成
    """
    if x is None:
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # ReLU
    y_relu = torch.relu(x).detach()
    axes[0, 0].plot(x.detach(), y_relu)
    axes[0, 0].set_title('ReLU')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sigmoid
    y_sigmoid = torch.sigmoid(x).detach()
    axes[0, 1].plot(x.detach(), y_sigmoid)
    axes[0, 1].set_title('Sigmoid')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tanh
    y_tanh = torch.tanh(x).detach()
    axes[1, 0].plot(x.detach(), y_tanh)
    axes[1, 0].set_title('Tanh')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ReLU导数
    y_relu.backward(torch.ones_like(x), retain_graph=True)
    axes[1, 1].plot(x.detach(), x.grad.detach())
    axes[1, 1].set_title('ReLU derivative')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_heatmaps(matrices: list, xlabel: str, ylabel: str, 
                  titles: list = None, figsize: tuple = (2.5, 2.5), 
                  cmap: str = 'Reds'):
    """
    显示热图
    
    Args:
        matrices: 矩阵列表
        xlabel: x轴标签
        ylabel: y轴标签
        titles: 标题列表
        figsize: 图表大小
        cmap: 颜色映射
    """
    num_rows, num_cols = 1, len(matrices)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    if num_cols == 1:
        axes = [axes]
        
    for i, (matrix, ax) in enumerate(zip(matrices, axes)):
        im = ax.imshow(matrix.detach().numpy(), cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if titles:
            ax.set_title(titles[i])
            
    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.tight_layout()
    plt.show()


def plot_gradients(gradients: list, labels: list = None):
    """
    绘制梯度大小分布
    
    Args:
        gradients: 梯度张量列表
        labels: 梯度标签列表
    """
    if labels is None:
        labels = [f'Layer {i}' for i in range(len(gradients))]
        
    means = [g.abs().mean().item() for g in gradients]
    stds = [g.abs().std().item() for g in gradients]
    
    x = range(len(gradients))
    plt.figure(figsize=(10, 4))
    plt.bar(x, means, yerr=stds, alpha=0.7)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Gradient magnitude')
    plt.title('Gradient Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
