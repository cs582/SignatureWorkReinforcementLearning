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