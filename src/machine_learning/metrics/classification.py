"""
分类评估指标
"""

import torch
import numpy as np
from typing import Tuple


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算分类准确率
    
    Args:
        y_hat: 预测结果，形状 (batch_size, num_classes) 或 (batch_size,)
        y: 真实标签，形状 (batch_size,)
        
    Returns:
        准确率（0-1之间）
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


def precision(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = None
) -> torch.Tensor:
    """
    计算精确率（Precision）
    
    Precision = TP / (TP + FP)
    
    Args:
        y_hat: 预测标签
        y: 真实标签
        num_classes: 类别数
        
    Returns:
        各类别的精确率
    """
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(dim=1)
    
    if num_classes is None:
        num_classes = int(max(y.max(), y_hat.max()).item()) + 1
    
    precisions = []
    for c in range(num_classes):
        true_positive = ((y_hat == c) & (y == c)).sum().float()
        predicted_positive = (y_hat == c).sum().float()
        
        if predicted_positive > 0:
            precisions.append(true_positive / predicted_positive)
        else:
            precisions.append(torch.tensor(0.0))
    
    return torch.tensor(precisions)


def recall(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = None
) -> torch.Tensor:
    """
    计算召回率（Recall）
    
    Recall = TP / (TP + FN)
    
    Args:
        y_hat: 预测标签
        y: 真实标签
        num_classes: 类别数
        
    Returns:
        各类别的召回率
    """
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(dim=1)
    
    if num_classes is None:
        num_classes = int(max(y.max(), y_hat.max()).item()) + 1
    
    recalls = []
    for c in range(num_classes):
        true_positive = ((y_hat == c) & (y == c)).sum().float()
        actual_positive = (y == c).sum().float()
        
        if actual_positive > 0:
            recalls.append(true_positive / actual_positive)
        else:
            recalls.append(torch.tensor(0.0))
    
    return torch.tensor(recalls)


def f1_score(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = None,
    average: str = "macro"
) -> float:
    """
    计算 F1 分数
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_hat: 预测标签
        y: 真实标签
        num_classes: 类别数
        average: 平均方式 ('macro', 'micro', 'weighted')
        
    Returns:
        F1 分数
    """
    prec = precision(y_hat, y, num_classes)
    rec = recall(y_hat, y, num_classes)
    
    # 避免除以零
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    
    if average == "macro":
        return f1.mean().item()
    elif average == "micro":
        # Micro F1 等于准确率
        return accuracy(y_hat, y)
    
    return f1.mean().item()


def confusion_matrix(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = None
) -> torch.Tensor:
    """
    计算混淆矩阵
    
    Args:
        y_hat: 预测标签
        y: 真实标签
        num_classes: 类别数
        
    Returns:
        混淆矩阵 (num_classes x num_classes)
    """
    if len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(dim=1)
    
    if num_classes is None:
        num_classes = int(max(y.max(), y_hat.max()).item()) + 1
    
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    for t, p in zip(y.view(-1), y_hat.view(-1)):
        matrix[t.long(), p.long()] += 1
    
    return matrix
