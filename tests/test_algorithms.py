"""
算法测试
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from machine_learning.algorithms import (
    bubble_sort,
    merge_sort,
    fft,
    fft_matrix,
    fft_iter,
    BPNN,
)


def test_bubble_sort():
    """测试冒泡排序"""
    arr = [64, 34, 25, 12, 22, 11, 90]
    result = bubble_sort(arr)
    
    assert result == [11, 12, 22, 25, 34, 64, 90]
    assert arr == [64, 34, 25, 12, 22, 11, 90]  # 原数组不应改变


def test_merge_sort():
    """测试归并排序"""
    arr = [38, 27, 43, 3, 9, 82, 10]
    result = merge_sort(arr)
    
    assert result == [3, 9, 10, 27, 38, 43, 82]


def test_fft():
    """测试 FFT 算法"""
    # 创建测试信号
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    
    # 测试递归 FFT
    result = fft(x)
    
    # 验证与 torch.fft 的一致性
    expected = torch.fft.fft(torch.view_as_complex(x))
    assert torch.allclose(result, expected, atol=1e-5)


def test_fft_matrix():
    """测试矩阵版 FFT"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    
    result = fft_matrix(x)
    expected = torch.fft.fft(torch.view_as_complex(x))
    
    assert torch.allclose(result, expected, atol=1e-5)


def test_fft_iter():
    """测试迭代版 FFT"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    
    result = fft_iter(x)
    expected = torch.fft.fft(torch.view_as_complex(x))
    
    assert torch.allclose(result, expected, atol=1e-5)


def test_bpnn():
    """测试 BPNN"""
    # XOR 问题
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    model = BPNN(input_size=2, hidden_size=4, output_size=1)
    
    # 训练
    model.train(X, y, lr=0.5, epochs=5000, verbose=False)
    
    # 预测
    predictions = model.predict(X)
    
    # 验证准确率
    accuracy = np.mean((predictions > 0.5) == y)
    assert accuracy > 0.9, f"Accuracy should be > 0.9, got {accuracy}"


def test_merge():
    """测试归并函数"""
    from machine_learning.algorithms.sorting import merge
    
    left = [1, 3, 5]
    right = [2, 4, 6]
    
    result = merge(left, right)
    assert result == [1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    print("Running algorithm tests...")
    
    test_bubble_sort()
    print("✅ Bubble sort test passed")
    
    test_merge_sort()
    print("✅ Merge sort test passed")
    
    test_fft()
    print("✅ FFT test passed")
    
    test_fft_matrix()
    print("✅ FFT matrix test passed")
    
    test_fft_iter()
    print("✅ FFT iter test passed")
    
    test_bpnn()
    print("✅ BPNN test passed")
    
    test_merge()
    print("✅ Merge test passed")
    
    print("\n✨ All algorithm tests passed!")
