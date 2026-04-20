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

# implement FFT with matrix multiplication
import torch
import numpy as np
import matplotlib.pyplot as plt

def dft_matrix(N):
    """
    N: the size of the matrix
    """
    n = torch.arange(N).reshape(N, 1)
    k = torch.arange(N).reshape(1, N)
    M = torch.exp(-2j * np.pi * k * n / N)
    return M


# 实现一个冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-1-i):
            if arr[j+1] < arr[j]:
                arr[j+1], arr[j] = arr[j], arr[j+1]
    return arr


# 实现一个归并排序
def merge_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# 合并两个有序数组
def merge(left, right):
    res = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        res.append(left[i]) if left[i] < right[j] else res.append(right[j])
        i += 1 if left[i] < right[j] else 0
        j += 1 if left[i] >= right[j] else 0
    res += left[i:] if i < len(left) else right[j:]
    return res

# 实现一个比赛的接口， 可以进行加分，减分等
class Match:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def add_score(self, score):
        self.score += score

    def sub_score(self, score):
        self.score -= score

    def __str__(self):
        return self.name + ' ' + str(self.score)
    
    # 将分数存储到mysql数据库
    def save(self):
        # 1. 连接数据库
        # 2. 插入数据
        # 3. 关闭数据库
        pass
    
# implement a FFT
def fft(x):
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    if N <= 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = torch.exp(-2j * np.pi * torch.arange(N) / N)
    return torch.cat([even + factor[:N//2] * odd, even + factor[N//2:] * odd])

# implement a FFT with matrix multiplication
def fft_matrix(x):
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    if N <= 1:
        return x
    factor = torch.exp(-2j * np.pi * torch.arange(N) / N)
    M = dft_matrix(N)
    return torch.matmul(M, x)

# implement a FFT without recursion
def fft_iter(x):
    x = torch.as_tensor(x, dtype=torch.complex64)
    N = x.shape[0]
    if N <= 1:
        return x
    factor = torch.exp(-2j * np.pi * torch.arange(N) / N)
    M = dft_matrix(N)
    for i in range(int(np.log2(N))):
        for j in range(2**i):
            M[2**i+j] = M[2**i+j] * factor[2**i*j]
    return torch.matmul(M, x)

# implement a FFT without third party library
def fft_numpy(x):
    x = np.asarray(x, dtype=np.complex64)
    N = x.shape[0]
    if N <= 1:
        return x
    even = fft_numpy(x[::2])
    odd = fft_numpy(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N//2] * odd, even + factor[N//2:] * odd])

# implement a FFT with matrix multiplication, and without third party library
def fft_numpy_matrix(x):
    x = np.asarray(x, dtype=np.complex64)
    N = x.shape[0]
    if N <= 1:
        return x
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    M = dft_matrix(N)
    return np.matmul(M, x)

# implement a BPNN class
class BPNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # initialize the weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        # initialize the bias
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y, y_pred, lr):
        self.loss = y_pred - y
        self.y_pred = y_pred
        
        self.W2_gradient = np.dot(self.a1.T, self.loss * self.sigmoid_derivative(self.y_pred))
        self.W1_gradient = np.dot(x.T, np.dot(self.loss * self.sigmoid_derivative(self.y_pred), self.W2.T) * self.sigmoid_derivative(self.a1))
        
        self.b2_gradient = np.sum(self.loss * self.sigmoid_derivative(self.y_pred), axis=0)
        self.b1_gradient = np.sum(np.dot(self.loss * self.sigmoid_derivative(self.y_pred), self.W2.T) * self.sigmoid_derivative(self.a1), axis=0)
        
        self.W2 -= lr * self.W2_gradient
        self.W1 -= lr * self.W1_gradient
        
        self.b2 -= lr * self.b2_gradient
        self.b1 -= lr * self.b1_gradient
        
    def train(self, x, y, lr, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y, y_pred, lr)
            
    def predict(self, x):
        return self.forward(x)
