import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class MLPChannels(nn.Module):
    def __init__(self, n_channels, bn):
        super().__init__()
        self.layers = []
        self.n_channels = n_channels
        for i in range(len(n_channels) - 1):
            self.layers.append(nn.Linear(n_channels[i], n_channels[i + 1]))
            if i != len(n_channels) - 2:
                if bn:
                    self.layers.append(nn.BatchNorm1d(n_channels[i + 1]))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class MLP(nn.Module):
    def __init__(self, n_layers, n_channel_in, n_channel_out, n_phase_channel, bn=False, last_activation=False):
        super().__init__()
        self.layers = []
        self.need_add_one = last_activation
        for i in range(n_layers):
            n_out = n_channel_out if i == n_layers - 1 else n_channel_in
            self.layers.append(nn.Linear(n_channel_in, n_out))
            if i != n_layers - 1:
                if bn:
                    self.layers.append(nn.BatchNorm1d(n_phase_channel))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
            if last_activation:
                self.layers.append(nn.ELU())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        if self.need_add_one:
            x = x + 1
        return x

class LN_v3(nn.Module):
    def __init__(self, dim, epsilon=1e-8, keep_std=False):
        super().__init__()
        self.epsilon = epsilon
        self.keep_std = keep_std

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        if self.keep_std:
            std = 1.
        else:
            std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        return y