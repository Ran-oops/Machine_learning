"""
机器学习项目工具模块

提供数据加载、可视化、模型评估等常用工具函数
"""

from .data_utils import (
    load_fashion_mnist,
    get_dataloader_workers,
    SyntheticRegressionData,
)

from .visualization import (
    plot_images,
    plot_metrics,
    plot_activation_functions,
    set_figure_size,
)

from .training_utils import (
    train_epoch,
    evaluate_accuracy,
    evaluate_loss,
    Animator,
)

__all__ = [
    # Data utilities
    'load_fashion_mnist',
    'get_dataloader_workers',
    'SyntheticRegressionData',
    # Visualization
    'plot_images',
    'plot_metrics',
    'plot_activation_functions',
    'set_figure_size',
    # Training utilities
    'train_epoch',
    'evaluate_accuracy',
    'evaluate_loss',
    'Animator',
]
