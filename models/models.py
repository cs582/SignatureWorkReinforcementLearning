from models.modules import *

import torch
import torch.nn as nn
from logs.logger_file import logger_cnn



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


class ViT(nn.Module):
    def __init__(self, in_size, n_classes, dropout):
        super(ViT, self).__init__()
        n_embeddings, embedding_dim = in_size
        self.pos_embedding = nn.Parameter(torch.randn(1, n_embeddings + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embedding_dim, activation='gelu', nhead=1, dim_feedforward=256),
            nn.TransformerEncoderLayer(d_model=embedding_dim, activation='gelu', nhead=1, dim_feedforward=256),
            nn.TransformerEncoderLayer(d_model=embedding_dim, activation='gelu', nhead=1, dim_feedforward=256),
            nn.TransformerEncoderLayer(d_model=embedding_dim, activation='gelu', nhead=1, dim_feedforward=256)
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, x):
        _, n_vectors, vector_dim = x.shape

        x += self.pos_embedding[:, :(n_vectors + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)