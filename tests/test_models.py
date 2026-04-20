"""
模型测试
"""

import torch
import pytest
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from machine_learning.models import (
    LinearRegression,
    LinearRegressionScratch,
    MLP,
    SimpleMLP,
    SoftmaxRegression,
)


def test_linear_regression():
    """测试线性回归模型"""
    model = LinearRegression(input_dim=2)
    
    # 测试前向传播
    x = torch.randn(10, 2)
    y = model(x)
    
    assert y.shape == (10, 1)
    assert not torch.isnan(y).any()


def test_linear_regression_scratch():
    """测试从零实现的线性回归"""
    model = LinearRegressionScratch(input_dim=2, lr=0.01)
    
    # 测试前向传播
    x = torch.randn(10, 2)
    y = model.forward(x)
    
    assert y.shape == (10, 1)


def test_mlp():
    """测试 MLP 模型"""
    model = MLP(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        activation='relu'
    )
    
    # 测试前向传播
    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    
    assert y.shape == (5, 10)
    assert not torch.isnan(y).any()


def test_simple_mlp():
    """测试简洁版 MLP"""
    model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10
    )
    
    # 测试前向传播
    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    
    assert y.shape == (5, 10)


def test_softmax_regression():
    """测试 Softmax 回归"""
    model = SoftmaxRegression(input_dim=784, output_dim=10)
    
    # 测试前向传播
    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    
    assert y.shape == (5, 10)


def test_mlp_activations():
    """测试不同激活函数"""
    for activation in ['relu', 'sigmoid', 'tanh']:
        model = MLP(
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            activation=activation
        )
        
        x = torch.randn(2, 1, 28, 28)
        y = model(x)
        
        assert y.shape == (2, 10)
        assert not torch.isnan(y).any()


if __name__ == "__main__":
    # 运行测试
    print("Running model tests...")
    
    test_linear_regression()
    print("✅ LinearRegression test passed")
    
    test_linear_regression_scratch()
    print("✅ LinearRegressionScratch test passed")
    
    test_mlp()
    print("✅ MLP test passed")
    
    test_simple_mlp()
    print("✅ SimpleMLP test passed")
    
    test_softmax_regression()
    print("✅ SoftmaxRegression test passed")
    
    test_mlp_activations()
    print("✅ MLP activations test passed")
    
    print("\n✨ All tests passed!")
