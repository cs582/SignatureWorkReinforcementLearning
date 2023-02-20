import torch
import torch.nn as nn
from logs.logger_file import logger_cnn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, inplace=False, bias=False):
        super(Block, self).__init__()
        logger_cnn.debug("Constructing Block")
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=inplace)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        nn.init.xavier_uniform(self.conv2.weight)
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
    def __init__(self, in_size, n_classes, kernel=3, inplace=True, bias=False):
        super(DQN, self).__init__()
        logger_cnn.info("Constructing DQN")

        # backbone
        self.block1 = Block(in_channels=1, out_channels=8, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block2 = Block(in_channels=8, out_channels=16, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block3 = Block(in_channels=16, out_channels=32, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)

        self.fc1 = nn.Linear(in_size[0] * in_size[1] * 32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_classes)

        self.relu = nn.ReLU(inplace=True)

        self.out = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x = self.out(x)

        return x


class DuelingDQN(nn.Module):
    def __init__(self, in_size, n_classes, kernel=3, inplace=True, bias=False):
        super(DuelingDQN, self).__init__()
        logger_cnn.info("Constructing Dueling DQN")

        self.n_classes = n_classes

        # ResNet-20 backbone
        self.block1 = Block(in_channels=1, out_channels=8, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block2 = Block(in_channels=8, out_channels=16, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)
        self.block3 = Block(in_channels=16, out_channels=32, kernel_size=(kernel, kernel), inplace=inplace, bias=bias)

        self.val1 = nn.Linear(in_size[0] * in_size[1] * 32, 128)
        self.val2 = nn.Linear(128, 128)
        self.val3 = nn.Linear(128, 64)
        self.val4 = nn.Linear(64, 1)

        self.adv1 = nn.Linear(in_size[0] * in_size[1] * 32, 256)
        self.adv2 = nn.Linear(256, 128)
        self.adv3 = nn.Linear(128, 64)
        self.adv4 = nn.Linear(64, n_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)

        x_val = self.relu(self.val1(x))
        x_val = self.val2(x_val)
        x_val = self.val3(x_val)
        x_val = self.val4(x_val).repeat(1, self.n_classes)

        x_adv = self.relu(self.adv1(x))
        x_adv = self.adv2(x_adv)
        x_adv = self.adv3(x_adv)
        x_adv = self.adv4(x_adv)

        return x_val.add(x_adv - x_adv.sum()/x_adv.shape[0])
