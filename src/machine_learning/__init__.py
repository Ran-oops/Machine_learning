"""
Machine Learning 项目主模块

基于 PyTorch 的深度学习学习项目
"""

from .algorithms import (
    BPNN,
    bubble_sort,
    merge_sort,
    fft,
    fft_matrix,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Neural Networks
    "BPNN",
    # Algorithms
    "bubble_sort",
    "merge_sort",
    "fft",
    "fft_matrix",
]
