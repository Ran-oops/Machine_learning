"""
分类损失函数
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    交叉熵损失
    
    注意：y_hat 应该是 logits（未经过 softmax），
    PyTorch 的 cross_entropy 内部会应用 softmax
    
    Args:
        y_hat: 预测 logits，形状 (batch_size, num_classes)
        y: 真实标签索引，形状 (batch_size,)
        reduction: 归约方式
        
    Returns:
        交叉熵损失
    """
    return F.cross_entropy(y_hat, y, reduction=reduction)


def softmax_cross_entropy(
    y_hat: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Softmax 交叉熵（手动实现版）
    
    Args:
        y_hat: 预测 logits
        y: 真实标签索引
        
    Returns:
        损失张量
    """
    # 手动实现 softmax
    y_hat_softmax = F.softmax(y_hat, dim=1)
    
    # 选取真实标签对应的概率
    y_hat_selected = y_hat_softmax[range(len(y_hat)), y]
    
    # 负对数似然
    return -torch.log(y_hat_selected + 1e-7)


def binary_cross_entropy(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    二元交叉熵损失
    
    Args:
        y_hat: 预测概率（经过 sigmoid）
        y: 真实标签（0 或 1）
        reduction: 归约方式
        
    Returns:
        BCE 损失
    """
    return F.binary_cross_entropy(y_hat, y, reduction=reduction)


def binary_cross_entropy_with_logits(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    带 logits 的二元交叉熵
    
    Args:
        y_hat: 预测 logits（未经过 sigmoid）
        y: 真实标签
        reduction: 归约方式
        
    Returns:
        BCE 损失
    """
    return F.binary_cross_entropy_with_logits(y_hat, y, reduction=reduction)


def nll_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    负对数似然损失（Negative Log Likelihood）
    
    Args:
        y_hat: log softmax 输出
        y: 真实标签
        reduction: 归约方式
        
    Returns:
        NLL 损失
    """
    return F.nll_loss(y_hat, y, reduction=reduction)


def focal_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Focal Loss
    
    解决类别不平衡问题，降低易分类样本的权重
    
    Args:
        y_hat: 预测 logits
        y: 真实标签
        alpha: 权重因子
        gamma: 聚焦参数
        reduction: 归约方式
        
    Returns:
        Focal 损失
    """
    ce_loss = F.cross_entropy(y_hat, y, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss
