"""
CNN for MNIST dataset
"""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(32*14*14, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(-1, 32*14*14)
        x = self.classifier(x)
        return x