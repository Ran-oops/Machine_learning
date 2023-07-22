import torch

# 定义输入和输出数据
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 定义模型
model = torch.nn.Linear(1, 1)

# 定义损失函数
loss_fn = torch.nn.MSELoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for t in range(1000):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试模型
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print(y_test)