import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(DQN, self).__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(input_size, out_channels=input_size // 2)
        self.conv1_bn = nn.BatchNorm2d(input_size)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(input_size // 2, out_channels=input_size // 2)
        self.conv2_bn = nn.BatchNorm2d(input_size // 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(input_size // 2, out_channels=input_size // 4)
        self.conv3_bn = nn.BatchNorm2d(input_size // 2)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu3(x)

        x = self.fc1(x)
        x = self.fc1_bn(x)

        x = self.fc2(x)
        x = nn.Softmax(self.fc2_bn(x))

        return x
