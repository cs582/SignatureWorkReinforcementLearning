import torch.nn as nn
from logs.logger_file import logger_cnn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=False, bias=False):
        super(Block, self).__init__()
        logger_cnn.debug("Constructing Block")
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, bias=bias)
        nn.init.xavier_uniform(self.conv1.weight)

        self.relu = nn.ReLU(inplace=inplace)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0, stride=1, bias=bias)
        nn.init.xavier_uniform(self.conv2.weight)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.pool(out)
        return out