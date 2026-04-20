"""
学习率调度器实现

包含常用的学习率调整策略
"""

import math
from typing import List


class StepLR:
    """
    阶梯式学习率衰减
    
    每隔 step_size 个 epoch，学习率乘以 gamma
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        """
        Args:
            optimizer: 优化器
            step_size: 衰减间隔
            gamma: 衰减系数
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch % self.step_size == 0 and epoch > 0:
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * (self.gamma ** (epoch // self.step_size))
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingLR:
    """
    余弦退火学习率调度
    
    使用余弦函数平滑降低学习率
    """
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0):
        """
        Args:
            optimizer: 优化器
            T_max: 最大迭代次数
            eta_min: 最小学习率
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = -1
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for i, group in enumerate(self.optimizer.param_groups):
            lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * \
                 (1 + math.cos(math.pi * epoch / self.T_max)) / 2
            group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class ExponentialLR:
    """
    指数衰减学习率
    
    每个 epoch 都进行衰减
    """
    
    def __init__(self, optimizer, gamma: float = 0.95):
        """
        Args:
            optimizer: 优化器
            gamma: 衰减系数
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = -1
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * (self.gamma ** epoch)
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class ReduceLROnPlateau:
    """
    根据验证指标自动调整学习率
    
    当指标不再改善时，降低学习率
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 0.0001,
        cooldown: int = 0,
        min_lr: float = 0
    ):
        """
        Args:
            optimizer: 优化器
            mode: 'min' 或 'max'，表示指标应该最小化还是最大化
            factor: 衰减系数
            patience: 容忍次数
            threshold: 改善阈值
            cooldown: 冷却期
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, metrics: float, epoch: int = None):
        """
        根据指标更新学习率
        
        Args:
            metrics: 当前指标值
            epoch: 当前 epoch
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            return
        
        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def is_better(self, a: float, best: float) -> bool:
        """判断当前指标是否更好"""
        if self.mode == 'min':
            return a < best - self.threshold
        return a > best + self.threshold
    
    def _reduce_lr(self):
        """降低学习率"""
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            group['lr'] = new_lr
            print(f"Reducing learning rate from {old_lr:.4f} to {new_lr:.4f}")
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
