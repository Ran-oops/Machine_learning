"""
评估指标测试
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from machine_learning.metrics import (
    accuracy,
    confusion_matrix,
    mse,
    mae,
    r2_score,
)


def test_accuracy():
    """测试准确率"""
    y_hat = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([1, 0, 1])
    
    acc = accuracy(y_hat, y)
    assert acc == 1.0


def test_confusion_matrix():
    """测试混淆矩阵"""
    y_hat = torch.tensor([0, 1, 1, 0, 1, 0])
    y = torch.tensor([0, 1, 0, 0, 1, 1])
    
    cm = confusion_matrix(y_hat, y, num_classes=2)
    
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # True negative
    assert cm[1, 1] == 2  # True positive


def test_mse():
    """测试 MSE"""
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.5, 2.5, 3.5])
    
    error = mse(y_hat, y)
    assert error == 0.25


def test_mae():
    """测试 MAE"""
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.5, 2.5, 3.5])
    
    error = mae(y_hat, y)
    assert error == 0.5


def test_r2_score():
    """测试 R2 分数"""
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    
    score = r2_score(y_hat, y)
    assert score == 1.0


if __name__ == "__main__":
    print("Running metrics tests...")
    
    test_accuracy()
    print("Accuracy test passed")
    
    test_confusion_matrix()
    print("Confusion matrix test passed")
    
    test_mse()
    print("MSE test passed")
    
    test_mae()
    print("MAE test passed")
    
    test_r2_score()
    print("R2 score test passed")
    
    print("All metrics tests passed")
