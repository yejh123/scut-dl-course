import os
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
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 64
# learning rate
lr = 0.0002
# epoch
epoch = 5
# data root
data_dir = '../../data'
# img save dir
img_dir = '../img'
# checkpoints dir
checkpoints_dir = '../checkpoints'
# mkdir
os.makedirs(img_dir,exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)


class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()
        # Define your model here
        self.main = nn.Sequential(
        
        )

    def forward(self, input):
        input = input.view(len(input), -1)
        return self.main(input)


# Image Augmentation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

# load MNIST dataset
data_train = datasets.MNIST(root=data_dir,
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root=data_dir,
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
dnn = MLP_model().to(device)
print(dnn)

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

        # Backward pass

        # calculate correct
    
    # Test 
    dnn.eval()
    for i, (X, y) in enumerate(data_loader_test):
        X, y = X.to(device), y.to(device)
        
        # calculate correct

    print("Epoch: {}/{}, Train Loss is:{:.4f}, Test Loss is:{:.4f}, "
          "Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}"
          .format(e + 1, epoch,
                  total_train_loss / len(data_train),
                  total_test_loss / len(data_test),
                  100 * total_train_correct / len(data_train),
                  100 * total_test_correct / len(data_test)))

# plot loss and acc

# Save your model
torch.save(dnn.state_dict(), os.path.join(checkpoints_dir, f'dnn.pth'))




