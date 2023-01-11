import torch
import torch.nn as nn
import logging


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, inplace=False, bias=False):
        super(Block, self).__init__()
        logging.debug("Constructing Block")
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=inplace)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=inplace)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class DQN(nn.Module):
    def __init__(self, n_classes, kernel=3, inplace=True, bias=False):
        super(DQN, self).__init__()
        logging.info("Constructing DQN")

        # ResNet-20 backbone
        self.n_classes = n_classes

        # 2 Blocks of 64 channels
        self.block1 = Block(in_channels=1, out_channels=8, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block2 = Block(in_channels=8, out_channels=8, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        # 2 Blocks of 128 channels
        self.block3 = Block(in_channels=8, out_channels=16, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block4 = Block(in_channels=16, out_channels=16, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        # 2 Blocks of 256 channels
        self.block5 = Block(in_channels=16, out_channels=32, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block6 = Block(in_channels=32, out_channels=32, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        # 2 Blocks of 512 channels
        self.block7 = Block(in_channels=32, out_channels=64, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block8 = Block(in_channels=64, out_channels=64, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)

        self.fc1 = nn.Linear(n_classes * n_classes * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, self.n_classes)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_in = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x += x_in

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        x = self.softmax(x)

        return x
