import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, n_classes, kernel=3):
        super(DQN, self).__init__()
        # ResNet-20 backbone
        self.n_classes = n_classes

        # 2 Blocks of 64 channels
        self.block1 = Block(in_channels=1, out_channels=64, kernel_size=(kernel, kernel))
        self.block2 = Block(in_channels=64, out_channels=64, kernel_size=(kernel, kernel))
        # 2 Blocks of 128 channels
        self.block3 = Block(in_channels=64, out_channels=128, kernel_size=(kernel, kernel))
        self.block4 = Block(in_channels=128, out_channels=128, kernel_size=(kernel, kernel))
        # 2 Blocks of 256 channels
        self.block5 = Block(in_channels=128, out_channels=256, kernel_size=(kernel, kernel))
        self.block6 = Block(in_channels=256, out_channels=256, kernel_size=(kernel, kernel))
        # 2 Blocks of 512 channels
        self.block7 = Block(in_channels=256, out_channels=512, kernel_size=(kernel, kernel))
        self.block8 = Block(in_channels=512, out_channels=512, kernel_size=(kernel, kernel))

        self.fc1 = nn.Linear(n_classes * n_classes * 512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, self.n_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_in = x
        print(x_in.shape)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        print(x.shape)

        x += x_in

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        x = self.softmax(x)

        return x
