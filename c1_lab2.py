import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3,3), stride = stride,padding = 1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (3,3), stride = 1,padding = 1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channels)
   
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(x)
        
        out = self.conv2(x)
        out = self.bn2(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
      
        self.conv1 = nn.Conv2d(3, 64, kernel_size = (3,3), stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # sub group 1
        self.block1_1 = BasicBlock(64,64,stride =1)
        self.block1_2 = BasicBlock(64,64,stride =1)
        # sub group 2
        self.block2_1 = BasicBlock(64,128,stride =2)
        self.block2_2 = BasicBlock(128,128,stride =2)
        # sub group 3
        self.block3_1 = BasicBlock(128,256,stride =2)
        self.block3_2 = BasicBlock(256,256,stride =2)
        # sub group 4
        self.block4_1 = BasicBlock(256,512,stride =2)
        self.block4_2 = BasicBlock(512,512,stride =2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.block1_1(x)
        x = self.block1_2(x)

        x = self.block2_1(x)
        x = self.block2_2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)

        x = self.block4_1(x)
        x = self.block4_2(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def main(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
    ])
    train_datasets = datasets.CIFAR10(root = args.data_path, train=True, download=True,transfrom=transform)
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss+= loss.item()
            _,predicted = output.max(1)
            total+= labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total*100
    print(f'Epoch [{epoch+1}/5], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a CNN with Python')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])

    args = parser.parse_args()
    main(args)
