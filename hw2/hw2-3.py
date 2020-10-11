import numpy as np
import matplotlib.pyplot as plt

# 输入数据
# X = np.array([[1, 0, 0],
#               [1, 0, 1],
#               [1, 1, 0],
#               [1, 1, 1]])
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# 标签
Y = np.array([0, 1, 1, 0])[:, np.newaxis]

print(Y)
# 权值初始化，取值范围-1到1
V = np.random.random((2, 4)) * 2 - 1
W = np.random.random((4, 1)) * 2 - 1
print(V)
print(W)
# 学习率设置
lr = 0.11


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def update():
    pass 
    
    
    


for i in range(20000):
    update()  # 更新权值
    
    

def judge(x):
    if x >= 0.5:
        return 1
    else:
        return 0


for i in map(judge, L2):
    print(i)
