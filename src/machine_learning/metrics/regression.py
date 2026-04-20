"""
回归评估指标
"""

import torch


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    均方误差（Mean Squared Error）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        MSE 值
    """
    return ((y_hat - y) ** 2).mean().item()


def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    均方根误差（Root Mean Squared Error）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        RMSE 值
    """
    return torch.sqrt(((y_hat - y) ** 2).mean()).item()


def mae(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    平均绝对误差（Mean Absolute Error）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        MAE 值
    """
    return (y_hat - y).abs().mean().item()


def r2_score(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    R² 决定系数（Coefficient of Determination）
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        R² 值（范围通常在 0-1 之间）
    """
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    
    if ss_tot == 0:
        return 0.0
    
    return (1 - ss_res / ss_tot).item()


def mape(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    平均绝对百分比误差（Mean Absolute Percentage Error）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        MAPE 值
    """
    return (torch.abs((y - y_hat) / (y + 1e-8))).mean().item() * 100


def explained_variance(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    解释方差（Explained Variance）
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        解释方差值
    """
    return (1 - (y - y_hat).var() / y.var()).item()
