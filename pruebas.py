# Importacion de librerias
# %matplotlib inline

import matplotlib.pyplot as plt
import os
import torch
# from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.nn import functional as Fun
import torch.nn as nn

class RNC(nn.Module):
  # Constructor
  def __init__(self, input_size, num_classes):
    super(RNC, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=224, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2)
    self.conv2 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, stride=1, padding=1)
    # A drop layer deletes 20% of the features to help prevent overfitting
    self.drop = nn.Dropout2d(p=0.2)

    self.fc = nn.Linear(in_features=25*25*24, out_features=num_classes)

  def forward(self, x):
    x = Fun.relu(self.conv1(x))
    # x = self.pool(x)
    x = Fun.relu(self.conv2(x))
    # x = self.pool(x)
    x = Fun.relu(self.conv3(x))

    x = Fun.dropout(x, training=self.training)
    # Flatten
    x = x.view(-1, 25*25*24)
    # Feed to fully-connected layer to predict class
    x = self.fc(x)
    return Fun.log_softmax(x, dim=1)

def train(model: RNC, device, train_loader, optimizer, epoch):
  # Set model to training mode
  model.train()
  train_loss = 0
  print("Epoch:", epoch)

  # Process the image in batches
  for batch_idx, (data, target) in enumerate(train_loader):
    # Move the data to the selected device
    data, target = data.to(device), target.to(device)

    # Reset the optimizer
    optimizer.zero_grad()

    # Push the data forward through the model layers
    output = model(data)

    # Get the loss
    loss = Fun.nll_loss(output, target)

    # Keep a running total
    train_loss += loss.item()

    # Backpropagate
    loss.backward()
    optimizer.step()
    # train_loss += Fun.nll_loss(output, target, size_average=False).data.item()

    # return average loss for epoch
    avg_loss = train_loss / (batch_idx+1)
  print('\nTrain set: Average loss: {:.6f}'.format(avg_loss))
  return avg_loss

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define data transformations (if needed)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize if necessary
    transforms.ToTensor(),
])

# Create datasets using ImageFolder
train_dataset = datasets.ImageFolder(root='Cards/Train', transform=data_transforms)
test_dataset = datasets.ImageFolder(root='Cards/Test', transform=data_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

modelo = RNC(input_size=224, num_classes=2)
print(modelo)

avg_loss = train(modelo, 'cpu', train_loader, optim.Adam(modelo.parameters(), lr=0.001), 1)