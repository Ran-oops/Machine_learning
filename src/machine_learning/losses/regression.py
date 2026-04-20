"""
回归损失函数
"""

import torch
import torch.nn.functional as F


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    均方误差损失（基础版）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        损失张量
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def mean_squared_error(
    y_hat: torch.Tensor, 
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    均方误差损失（MSE）
    
    Args:
        y_hat: 预测值
        y: 真实值
        reduction: 归约方式 ('mean', 'sum', 'none')
        
    Returns:
        MSE 损失
    """
    return F.mse_loss(y_hat, y, reduction=reduction)


def mean_absolute_error(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    平均绝对误差损失（MAE / L1 Loss）
    
    Args:
        y_hat: 预测值
        y: 真实值
        reduction: 归约方式
        
    Returns:
        MAE 损失
    """
    return F.l1_loss(y_hat, y, reduction=reduction)


def smooth_l1_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
    beta: float = 1.0
) -> torch.Tensor:
    """
    Smooth L1 Loss（Huber Loss）
    
    在 |x| < beta 时使用 L2 损失，否则使用 L1 损失
    对异常值更鲁棒
    
    Args:
        y_hat: 预测值
        y: 真实值
        reduction: 归约方式
        beta: 阈值参数
        
    Returns:
        Smooth L1 损失
    """
    return F.smooth_l1_loss(y_hat, y, reduction=reduction, beta=beta)
