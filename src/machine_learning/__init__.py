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

from .models import (
    LinearRegression,
    LinearRegressionScratch,
    MLP,
    SimpleMLP,
    SoftmaxRegression,
    SoftmaxRegressionScratch,
)

from .losses import (
    squared_loss,
    mean_squared_error,
    cross_entropy_loss,
    softmax_cross_entropy,
    binary_cross_entropy,
)

from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    mse,
    rmse,
    mae,
    r2_score,
)

from .optim import (
    StepLR,
    CosineAnnealingLR,
    ExponentialLR,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Algorithms
    "BPNN",
    "bubble_sort",
    "merge_sort",
    "fft",
    "fft_matrix",
    # Models
    "LinearRegression",
    "LinearRegressionScratch",
    "MLP",
    "SimpleMLP",
    "SoftmaxRegression",
    "SoftmaxRegressionScratch",
    # Losses
    "squared_loss",
    "mean_squared_error",
    "cross_entropy_loss",
    "softmax_cross_entropy",
    "binary_cross_entropy",
    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    # Optimizers
    "StepLR",
    "CosineAnnealingLR",
    "ExponentialLR",
]
