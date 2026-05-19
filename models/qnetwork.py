import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers=[64, 64]):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        layer_sizes = [state_size] + layers + [action_size]
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, state):
        x = state
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        return self.fcs[-1](x)
