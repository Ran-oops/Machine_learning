"""
多层感知机 (MLP) 示例

展示如何使用项目中的 MLP 模型进行分类任务
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../src')

from machine_learning.models import MLP, SimpleMLP
from machine_learning.metrics import accuracy
from utils.data_utils import load_fashion_mnist
from utils.training_utils import train_epoch, evaluate_accuracy


def example_1_basic_mlp():
    """
    示例 1: 基础 MLP 训练
    """
    print("=" * 60)
    print("示例 1: Basic MLP Training")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_iter, test_iter = load_fashion_mnist(batch_size=256)
    
    # 2. 创建模型
    print("2. 创建模型...")
    model = MLP(
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        activation='relu',
        dropout=0.2
    )
    model.init_weights(std=0.01)
    
    print(f"   模型结构:")
    print(f"   - 输入维度: 784")
    print(f"   - 隐藏维度: 256")
    print(f"   - 输出维度: 10")
    print(f"   - Dropout: 0.2")
    
    # 3. 定义损失和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 4. 训练
    print("\n3. 训练模型（10 epochs）...")
    num_epochs = 10
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_iter, loss_fn, optimizer)
        
        # 评估
        test_acc = evaluate_accuracy(model, test_iter)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 2 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
    
    print(f"\n4. 最终测试准确率: {test_acc:.4f}")
    
    # 5. 可视化
    print("\n5. 绘制训练曲线...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # 计算每个epoch的accuracy gap
    gaps = [train - test for train, test in zip(train_accs, test_accs)]
    plt.plot(gaps)
    plt.xlabel('Epoch')
    plt.ylabel('Gap')
    plt.title('Train-Test Accuracy Gap')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mlp_training.png', dpi=150)
    print("   图表已保存到: mlp_training.png")
    
    print("\n" + "=" * 60 + "\n")


def example_2_simple_mlp():
    """
    示例 2: 使用简洁版 MLP
    """
    print("=" * 60)
    print("示例 2: Simple MLP")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载数据...")
    train_iter, test_iter = load_fashion_mnist(batch_size=256)
    
    # 创建简洁版模型
    print("2. 创建简洁版模型...")
    model = SimpleMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        activation='relu'
    )
    
    print(f"   模型结构: 784 -> 256 -> 128 -> 10")
    
    # 训练
    print("\n3. 快速训练（5 epochs）...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_iter, loss_fn, optimizer)
        test_acc = evaluate_accuracy(model, test_iter)
        
        print(f"   Epoch [{epoch+1}/5] "
              f"Loss: {train_loss:.4f}, "
              f"Test Acc: {test_acc:.4f}")
    
    print(f"\n4. 最终测试准确率: {test_acc:.4f}")
    print("\n" + "=" * 60 + "\n")


def example_3_compare_activations():
    """
    示例 3: 比较不同激活函数
    """
    print("=" * 60)
    print("示例 3: Compare Activation Functions")
    print("=" * 60)
    
    activations = ['relu', 'sigmoid', 'tanh']
    results = {}
    
    for act in activations:
        print(f"\n使用 {act} 激活函数:")
        
        # 加载数据
        train_iter, test_iter = load_fashion_mnist(batch_size=256)
        
        # 创建模型
        model = MLP(
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            activation=act,
            dropout=0.0
        )
        model.init_weights(std=0.01)
        
        # 训练
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        train_accs = []
        test_accs = []
        
        for epoch in range(5):
            train_loss, train_acc = train_epoch(model, train_iter, loss_fn, optimizer)
            test_acc = evaluate_accuracy(model, test_iter)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        results[act] = {
            'train': train_accs,
            'test': test_accs
        }
        
        print(f"   最终准确率: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # 可视化比较
    print("\n绘制对比图...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    for act in activations:
        plt.plot(results[act]['train'], label=act.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for act in activations:
        plt.plot(results[act]['test'], label=act.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150)
    print("   图表已保存到: activation_comparison.png")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # 运行示例
    example_1_basic_mlp()
    example_2_simple_mlp()
    example_3_compare_activations()
    
    print("\n✅ 所有示例运行完成！")
