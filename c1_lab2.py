import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
  expansion=1
  def __init__(self, in_channels, out_channels, stride=1, downsample = None):
    super().__init__()
    
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample
  
  def forward(self,x):
    identity = x
    
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.downsample is not None:
      identity = self.downsample(x)
    
    out+=identity
    
    return self.relu(out)

class Resnet(nn.Module):
  def __init__(self, block, layers,num_classes=1000):
    super().__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.maxpool = nn.Identity()
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*block.expansion, num_classes)

  def _make_layer(self, block, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or self.in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * block.expansion)
        )

    layers = [block(self.in_channels, out_channels, stride, downsample)]
    self.in_channels = out_channels * block.expansion

    for _ in range(1, blocks):
        layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

  def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataset(batch_size=64):
  transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  return trainloader, testloader


def resnet18(num_classes=1000):
    return Resnet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

import torch.optim as optim 
import deepspeed
import mpi4py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, engine, epoch):
  model.train()
  total_loss=0
  for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    engine.backward(loss)
    engine.step()

    total_loss += loss.item()
    if i%100 ==0:
      print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

def evaluate(model, test_loader):
  model.eval()
  correct = total = 0
  with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")


def main():
    train_loader, test_loader = get_cifar10_dataset(batch_size=64)

    model = resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="/content/config_1.json"
    )

    for epoch in range(10):
        train(model_engine, train_loader, criterion, optimizer, model_engine, epoch)
        evaluate(model_engine, test_loader)

if __name__ == "__main__":
    main()
