# 更新日志

所有显著的更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 创建 `src/utils/` 模块，包含数据、训练和可视化工具
- 创建 `src/machine_learning/algorithms/` 模块，包含：
  - 排序算法（冒泡排序、归并排序）
  - 神经网络（BPNN 反向传播实现）
  - FFT 算法（递归、迭代、矩阵实现）
- 创建 `docs/` 目录，存放项目文档
- 创建 `scripts/` 目录，包含环境设置和项目检查脚本
- 添加代码质量配置文件：
  - `.editorconfig` - 编辑器配置
  - `.pre-commit-config.yaml` - Git hooks
  - `.rumdl.toml` - Markdown linting
  - `ruff.toml` - Python linting
- 添加 `CONTRIBUTING.md` 贡献指南
- 添加 `CHANGELOG.md` 更新日志
- 优化 `pyproject.toml`，添加项目元数据和分类
- 重构 `test.py`，添加结构化的测试函数

### Changed

- 优化 `.gitignore`，改进文件过滤规则
- 更新 `README.md`，添加项目结构和使用示例
- 移动文档文件到 `docs/` 目录

### Removed

- 删除旧的备份 notebook 文件：
  - `notebooks/04_mlp/05_mlp_from_scratch_old.ipynb`
  - `notebooks/04_mlp/06_mlp_concise_old.ipynb`

## [0.1.0] - 2026-04-20

### Added

- 初始化项目结构
- 添加基础知识 notebooks（张量创建、操作、梯度计算）
- 添加线性回归 notebooks（理论、从零实现、PyTorch实现）
- 添加 Softmax 回归 notebooks（介绍、数据集、实现）
- 添加多层感知机 notebooks（激活函数、实现、模型选择）
- 添加正则化技术 notebooks（Dropout、权重衰减）
- 添加实验 notebooks（one-hot编码、实验笔记）
- 配置项目依赖（PyTorch、d2l、matplotlib 等）
- 添加代码质量工具配置
- 创建 `src/machine_learning/` 基础模块

[Unreleased]: https://github.com/yourusername/machine-learning/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/machine-learning/releases/tag/v0.1.0
