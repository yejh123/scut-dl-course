import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

num_observations = 100
x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)

# 绘制散点图
plt.figure()
plt.title('(x,y) scatter')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, y, label='(x,y)')
plt.legend()
plt.show()


def get_X(x):
    ones = np.ones(len(x))[:,np.newaxis]
    x = x[:,np.newaxis]
    return np.hstack((ones, x))


def get_y(y_data):
    return y_data[:, np.newaxis]


def get_h(theta, X):
    return X @ theta


def get_cost(theta, X, y):
    return np.mean(np.power(get_h(theta, X) - y, 2)) / 2


def gradient(theta, X, y):
    """
    求梯度
    :param theta: (n,1)
    :param X: (m,n)
    :param y: (m,)
    :return:
    """
    m = len(y)
    temp = X.T @ (X @ theta - y)
    return temp / m


def batch_gradient_descent(theta, X, y, lr, epochs, eps):
    _theta = theta.copy()
    adagrad = np.zeros((X.shape[1], 1))
    for i in range(epochs + 1):
        g = gradient(_theta, X, y)
        adagrad += g ** 2
        _theta -= lr * g / np.sqrt(adagrad + eps)
        if i % 50 == 0:
            cost = get_cost(_theta, X, y)
            print(f'第{i}轮cost: {cost}')
    return _theta


def linear_regression(theta, x_train, y_train, lr, epochs, eps):
    final_theta = batch_gradient_descent(theta, x_train, y_train, lr, epochs, eps)
    final_cost = get_cost(final_theta, x_train, y_train)
    return final_theta, final_cost


# 轮数epochs
epochs = 100
lr = 0.1
eps = 1e-8

x_train = get_X(x)
y_train = get_y(y)
theta = np.zeros((x_train.shape[1], 1))
final_theta, final_cost = linear_regression(theta, x_train, y_train, lr, epochs, eps)

print("result:", final_theta, final_cost)

# 绘制散点图
plt.figure()
plt.title('(x,y) regression')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, y, label='(x,y)')
plt.legend()
plt.plot(x, final_theta[1, 0] * x + final_theta[0, 0], 'r', label='regression function w*x+b')
plt.show()
