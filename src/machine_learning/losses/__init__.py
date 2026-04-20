"""
损失函数模块

包含各种损失函数的实现
"""

from .regression import squared_loss, mean_squared_error
from .classification import (
    cross_entropy_loss,
    softmax_cross_entropy,
    binary_cross_entropy,
)

__all__ = [
    "squared_loss",
    "mean_squared_error",
    "cross_entropy_loss",
    "softmax_cross_entropy",
    "binary_cross_entropy",
]
