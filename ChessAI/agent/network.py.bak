"""Compact residual policy/value network for Stockfish-supervised chess."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.encoding import ACTION_SIZE, INPUT_CHANNELS

RESIDUAL_CHANNELS = 64
RESIDUAL_BLOCKS = 4
POLICY_EMBEDDING = 32
VALUE_HIDDEN = 128


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PolicyHead(nn.Module):
    """Bilinear from-square/to-square head with five promotion planes."""
    def __init__(self, channels=RESIDUAL_CHANNELS, embedding_dim=POLICY_EMBEDDING):
        super().__init__()
        self.from_projection = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.to_projection = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.promotion_weights = nn.Parameter(torch.empty(5, embedding_dim, embedding_dim))
        self.promotion_bias = nn.Parameter(torch.zeros(5))
        nn.init.xavier_uniform_(self.promotion_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        from_features = self.from_projection(x).flatten(2).transpose(1, 2)
        to_features = self.to_projection(x).flatten(2).transpose(1, 2)
        logits = torch.einsum("bfi,pij,bgj->bfgp", from_features, self.promotion_weights, to_features)
        return (logits + self.promotion_bias).reshape(batch_size, ACTION_SIZE)


class ValueHead(nn.Module):
    def __init__(self, channels=RESIDUAL_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, VALUE_HIDDEN)
        self.fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x))).flatten(1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out)).squeeze(-1)


class ChessResNet(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, channels=RESIDUAL_CHANNELS, residual_blocks=RESIDUAL_BLOCKS):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.initial_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(residual_blocks)))
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (self.input_channels, 8, 8):
            raise ValueError(f"Expected [B,{self.input_channels},8,8], got {tuple(x.shape)}")
        out = self.relu(self.initial_bn(self.initial_conv(x)))
        out = self.residual_tower(out)
        return self.policy_head(out), self.value_head(out)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
