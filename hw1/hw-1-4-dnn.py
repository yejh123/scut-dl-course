import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 64
# learning rate
lr = 0.001
# epoch
epoch = 5


class DNN_model(nn.Module):
    def __init__(self):
        super(DNN_model, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 10),
            nn.Sigmoid(),
            nn.Softmax(),
        )

    def forward(self, input):
        input = input.view(len(input), -1)
        return self.main(input)


# Image Augmentation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

# load MNIST dataset
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size)

# have a look at MNIST
# real_batch = next(iter(data_loader_train))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(
#     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Define model
dnn = DNN_model().to(device)

# Define optimizer
optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)

# Define loss function
criterion = nn.CrossEntropyLoss()

######################
#       Train
######################
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []
sample_interval = 1
for e in range(epoch):
    total_train_correct = 0
    total_test_correct = 0
    total_train_loss = 0.0
    total_test_loss = 0.0
    dnn.train()
    for i, (X, y) in enumerate(data_loader_train):
        optimizer.zero_grad()

        X, y = X.to(device), y.to(device)
        predict = dnn(X)
        train_loss = criterion(predict, y)
        total_train_loss = train_loss.item() * len(X)

        # Backward pass
        train_loss.backward()
        optimizer.step()

        # calculate correct
        train_correct = torch.sum(torch.argmax(predict, dim=1) == y.data).item()
        total_train_correct += train_correct

        train_loss_list.append(train_loss)
        train_acc_list.append(train_correct / len(X))

    dnn.eval()
    for i, (X, y) in enumerate(data_loader_test):
        X, y = X.to(device), y.to(device)
        predict = dnn(X)
        test_loss = criterion(predict, y)
        total_test_loss = test_loss.item() * len(X)
        # calculate correct
        predict_label = torch.argmax(predict, dim=1)
        test_correct = torch.sum(predict_label == y.data).item()
        total_test_correct += test_correct

        test_loss_list.append(test_loss)
        test_acc_list.append(test_correct / len(X))

    # train_loss_list.append(total_train_loss / len(data_train))
    # test_loss_list.append(total_test_loss / len(data_train))
    # train_acc_list.append(train_correct / len(data_train))
    # test_acc_list.append(test_correct / len(data_test))
    # print(f"Epoch: {e + 1}/{epoch}, Train Loss is:{total_train_loss / len(data_train)},"
    #       f" Test Loss is:{total_test_loss / len(data_test)},"
    #       f" Train Accuracy is:{100 * total_train_correct / len(data_train)}%,"
    #       f" Test Accuracy is:{100 * total_test_correct / len(data_test)}%")

    print("Epoch: {}/{}, Train Loss is:{:.4f}, Test Loss is:{:.4f}, "
          "Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}"
          .format(e + 1, epoch,
                  total_train_loss / len(data_train),
                  total_test_loss / len(data_test),
                  100 * total_train_correct / len(data_train),
                  100 * total_test_correct / len(data_test)))

# 绘制loss图
img_x_train = np.linspace(0, len(train_loss_list) * sample_interval, len(train_loss_list) + 1)
img_x_test = np.linspace(0, len(test_loss_list) * sample_interval, len(test_loss_list) + 1)
plt.figure()
plt.title('Loss change')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.plot(img_x_train[1:], train_loss_list, label='Train Loss')
plt.plot(img_x_test[1:], test_loss_list, label='Test Loss')
plt.legend()

plt.figure()
plt.title('Correct rate')
plt.xlabel('Round')
plt.ylabel('Correct Rate')
plt.plot(img_x_train[1:], train_acc_list, label='Train Correct Rate')
plt.plot(img_x_test[1:], test_acc_list, label='Test Correct Rate')
plt.legend()
plt.show()
