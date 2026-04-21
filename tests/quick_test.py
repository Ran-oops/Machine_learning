"""
测试脚本

包含机器学习项目的各种测试和演示代码
"""

import torch
import torch.nn as nn
from src.machine_learning import BPNN, bubble_sort, merge_sort, fft, fft_matrix
from src.utils.data_utils import load_fashion_mnist, SyntheticRegressionData
from src.utils.training_utils import train_epoch, evaluate_accuracy, train
import numpy as np


def test_pytorch_linear():
    """测试 PyTorch 线性回归模型"""
    print("=" * 50)
    print("测试 PyTorch 线性回归")
    print("=" * 50)
    
    # 定义输入和输出数据
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[2.0], [4.0], [6.0]])
    
    # 定义模型
    model = nn.Linear(1, 1)
    
    # 定义损失函数
    loss_fn = nn.MSELoss()
    
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    print("训练模型...")
    for t in range(1000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 测试模型
    x_test = torch.tensor([[4.0]])
    y_test = model(x_test)
    print(f"预测 x=4 的结果: {y_test.item():.3f} (期望: 8.0)")
    print()


def test_sorting_algorithms():
    """测试排序算法"""
    print("=" * 50)
    print("测试排序算法")
    print("=" * 50)
    
    # 测试冒泡排序
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"冒泡排序输入: {arr}")
    result = bubble_sort(arr)
    print(f"冒泡排序结果: {result}")
    
    # 测试归并排序
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"归并排序输入: {arr}")
    result = merge_sort(arr)
    print(f"归并排序结果: {result}")
    print()


def test_bpnn():
    """测试反向传播神经网络"""
    print("=" * 50)
    print("测试 BPNN")
    print("=" * 50)
    
    # 创建简单数据集：异或问题
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 创建和训练模型
    model = BPNN(input_size=2, hidden_size=4, output_size=1)
    print("训练 XOR 问题...")
    model.train(X, y, lr=0.5, epochs=5000, verbose=False)
    
    # 测试
    print("预测结果:")
    for i in range(len(X)):
        pred = model.predict(X[i:i+1])
        print(f"  {X[i]} -> {pred[0][0]:.3f} (期望: {y[i][0]})")
    print()


def test_fft():
    """测试 FFT 算法"""
    print("=" * 50)
    print("测试 FFT 算法")
    print("=" * 50)
    
    # 创建测试信号
    N = 8
    t = torch.arange(N, dtype=torch.float32)
    x = torch.sin(2 * np.pi * t / N)
    
    print(f"输入信号: {x.numpy()}")
    
    # 递归 FFT
    result = fft(x)
    print(f"FFT 结果: {result.numpy()}")
    
    # 验证与 torch.fft 的一致性
    expected = torch.fft.fft(x)
    print(f"期望结果: {expected.numpy()}")
    print(f"误差: {torch.abs(result - expected).max().item():.6f}")
    print()


def test_data_utils():
    """测试数据工具"""
    print("=" * 50)
    print("测试数据工具")
    print("=" * 50)
    
    # 测试合成回归数据
    print("创建合成回归数据...")
    w = torch.tensor([2.0, -3.4])
    b = 4.2
    data = SyntheticRegressionData(w=w, b=b, num_examples=1000)
    train_iter = data.get_dataloader(train=True)
    
    print(f"数据批次数量: {len(list(train_iter))}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Machine Learning 项目测试套件")
    print("=" * 60 + "\n")
    
    test_pytorch_linear()
    test_sorting_algorithms()
    test_bpnn()
    test_fft()
    test_data_utils()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
