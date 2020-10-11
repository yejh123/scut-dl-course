from __future__ import print_function
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

num_observations = 100
x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# 绘制散点图
# plt.figure()
# plt.title('(x,y) scatter')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.scatter(x, y, label='(x,y)')
# plt.legend()
# plt.show()

POLY_DEGREE = 3


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(x, y):
    """Builds a batch i.e. (x, f(x)) pair."""
    x = make_features(x)
    y = y.unsqueeze(1)
    return x, y


epoch = 10000
lr = 0.001
loss_list = []
sample_interval = 1
# Define model
fc = torch.nn.Linear(POLY_DEGREE, 1)
# Define optimizer
optimizer = torch.optim.Adam(fc.parameters(), lr=lr)

# Get data
batch_x, batch_y = get_batch(x, y)

for batch_idx in range(epoch):
    # Reset gradients
    # fc.zero_grad()
    optimizer.zero_grad()

    # Forward pass
    batch_predict = fc(batch_x)
    output = F.mse_loss(batch_predict, batch_y)
    loss = output.item()
    print(f"epoch: {batch_idx}, loss: {loss}")

    # Backward pass
    output.backward()

    # Apply gradients
    # for param in fc.parameters():
    #     param.data.add_(-lr * param.grad)
    # replace manual grad with optimizer step
    optimizer.step()

    if len(loss_list) > 0 and loss_list[-1] - loss < 1e-12:
        break

    if (batch_idx + 1) % sample_interval == 0:
        loss_list.append(loss)

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx + 1))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))

w = fc.weight.detach()
b = fc.bias.detach()

# 绘制散点图
plt.figure()
plt.title(f'(x,y) polynomial regression (POLY_DEGREE={POLY_DEGREE}, epoch={epoch})')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, y, label='(x,y)')
plt.plot(x, batch_x @ w.T + b, 'r', label=f'polynomial regression function {poly_desc(fc.weight.view(-1), fc.bias)}')
plt.legend()
plt.show()

# 绘制loss图
loss_x = np.linspace(0, len(loss_list) * sample_interval, len(loss_list) + 1)
plt.figure()
plt.title('loss list')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss_x[1:], loss_list, label='loss')
plt.legend()
plt.show()