"""
快速傅里叶变换 (FFT) 实现

包含递归和非递归实现，以及NumPy版本
"""

import torch
import numpy as np
from typing import Union


def dft_matrix(N: int, use_torch: bool = True) -> Union[torch.Tensor, np.ndarray]:
    """
    生成离散傅里叶变换矩阵
    
    Args:
        N: 矩阵大小
        use_torch: 是否返回PyTorch张量
        
    Returns:
        N x N 的DFT矩阵
    """
    n = np.arange(N).reshape(N, 1)
    k = np.arange(N).reshape(1, N)
    M = np.exp(-2j * np.pi * k * n / N)
    
    if use_torch:
        return torch.from_numpy(M).to(torch.complex64)
    return M.astype(np.complex64)


def fft(x: torch.Tensor) -> torch.Tensor:
    """
    递归实现快速傅里叶变换 (Cooley-Tukey算法)
    
    时间复杂度: O(n log n)
    
    Args:
        x: 输入序列，PyTorch张量
        
    Returns:
        FFT结果
    """
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    # 分治
    even = fft(x[::2])
    odd = fft(x[1::2])
    
    # 合并
    factor = torch.exp(-2j * np.pi * torch.arange(N) / N)
    return torch.cat([
        even + factor[:N // 2] * odd,
        even + factor[N // 2:] * odd
    ])


def fft_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    使用矩阵乘法的FFT实现
    
    Args:
        x: 输入序列
        
    Returns:
        FFT结果
    """
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    M = dft_matrix(N)
    return torch.matmul(M, x)


def fft_iter(x: torch.Tensor) -> torch.Tensor:
    """
    迭代实现快速傅里叶变换（不使用递归）
    
    Args:
        x: 输入序列
        
    Returns:
        FFT结果
    """
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    if (N & (N - 1)) != 0:
        raise ValueError("输入长度必须是2的幂次")
    
    # 位逆序重排
    x = bit_reverse_copy(x)
    
    # 迭代FFT
    for s in range(1, int(np.log2(N)) + 1):
        m = 2 ** s
        omega_m = np.exp(-2j * np.pi / m)
        
        for k in range(0, N, m):
            omega = 1
            for j in range(m // 2):
                t = omega * x[k + j + m // 2]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + m // 2] = u - t
                omega *= omega_m
    
    return x


def bit_reverse_copy(x: torch.Tensor) -> torch.Tensor:
    """位逆序复制"""
    N = x.shape[0]
    result = torch.zeros_like(x)
    
    for i in range(N):
        j = reverse_bits(i, int(np.log2(N)))
        result[j] = x[i]
    
    return result


def reverse_bits(x: int, bits: int) -> int:
    """反转整数x的bits位二进制表示"""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result


def fft_numpy(x: np.ndarray) -> np.ndarray:
    """
    使用NumPy实现的递归FFT
    
    Args:
        x: 输入序列，NumPy数组
        
    Returns:
        FFT结果
    """
    x = np.asarray(x, dtype=np.complex64)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    even = fft_numpy(x[::2])
    odd = fft_numpy(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    return np.concatenate([
        even + factor[:N // 2] * odd,
        even + factor[N // 2:] * odd
    ])


def fft_numpy_matrix(x: np.ndarray) -> np.ndarray:
    """
    使用NumPy实现的矩阵乘法FFT
    
    Args:
        x: 输入序列
        
    Returns:
        FFT结果
    """
    x = np.asarray(x, dtype=np.complex64)
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    M = dft_matrix(N, use_torch=False)
    return np.matmul(M, x)
