

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# def get_cifar10_classes():
#   return ('aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# load the cifar10 dataset
def load_cifer10_dataset():
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
  print("batch_size=4, each batch is (50000/4)=12500")

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
  return trainloader, testloader

# train the network
def train(net, trainloader):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    running_count = 0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data # get the inputs
      optimizer.zero_grad() # zero the parameter gradients

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      running_count += 1
      if i%1000 == 0:
        print(".",end="")

    print("Epoch {} iterations {}, average loss {}".format(epoch, running_count, running_loss/running_count))
  print('Finished Training')

# run test images
def test(net, testloader):
  correct, total = 0, 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return correct, total

if __name__ == "__main__":
  trainloader, testloader = load_cifer10_dataset()
  net = Net()
  train(net, trainloader)

  correct, total = test(net, testloader)
  print('Accuracy for 10000 test images: %d %%' % (100 * correct / total))
