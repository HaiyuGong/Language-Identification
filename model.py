import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_input_features = 64 * (input_size[0] // 8) * (input_size[1] // 8)
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch,w,h] -> [batch,channel=1,w,h]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.view(-1, self.fc_input_features)  # Flatten
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    input_size = (400, 80)
    num_classes = 2
    model = SimpleCNN(input_size, num_classes)
    input_data=torch.randn(10,400,80)
    outputs = model(input_data)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)