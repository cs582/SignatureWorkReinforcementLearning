from models.modules import *

import torch
import torch.nn as nn
from logs.logger_file import logger_cnn, logger_att


class DQN(nn.Module):
    def __init__(self, in_size, n_classes, inplace=True, bias=False):
        super(DQN, self).__init__()
        logger_cnn.info("Constructing DQN")

        # backbone
        self.block1 = Block(in_channels=1, out_channels=8, inplace=inplace, bias=bias)
        self.block2 = Block(in_channels=8, out_channels=16, inplace=inplace, bias=bias)
        self.block3 = Block(in_channels=16, out_channels=32, inplace=inplace, bias=bias)

        n, m = in_size

        self.fc1 = nn.Linear((n-12) * (m-12) * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)

        self.relu = nn.ReLU(inplace=True)

        self.out = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        print(x.shape)

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
        self.block1 = Block(in_channels=1, out_channels=8, inplace=inplace, bias=bias)
        self.block2 = Block(in_channels=8, out_channels=16, inplace=inplace, bias=bias)
        self.block3 = Block(in_channels=16, out_channels=32, inplace=inplace, bias=bias)

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
    def __init__(self, in_size, n_classes, dropout, vector_size, nhead=1):
        super(ViT, self).__init__()
        logger_att.info("Constructing Vision Transformer")
        n_embeddings, row_dim = in_size

        # Patch Embedding Eq. 1 "An Image is Worth 16x16 Words"
        self.patch_embedding_encoder = nn.Parameter(torch.randn(1, row_dim, vector_size))
        self.class_token = nn.Parameter(torch.randn(1, 1, vector_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_embeddings + 1, vector_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size),
            nn.TransformerEncoderLayer(d_model=vector_size, activation='gelu', nhead=nhead, dim_feedforward=vector_size)
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(vector_size),
            nn.Linear(vector_size, n_classes)
        )

    def forward(self, x):
        N, n_channels, n_vectors, vector_dim = x.shape

        # Since this considers the case when n_channels = 1, simply reshape
        x = x.view(N, n_vectors, vector_dim)

        # Patch Embedding
        x = torch.matmul(x, self.patch_embedding_encoder)
        class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer Encoder Layers
        x = self.transformer(x)

        # Getting class token
        x = x[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)
        return x