
from typing import Sequence

import torch
from torch import Tensor
from torch import nn

from .fc_network import FCNetwork


class QNetwork(nn.Module):
    """Continuous parametric action-value (Q) function with fully-connected (MLP) network."""

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 num_nets: int = 1):
        super().__init__()
        self.nets = nn.ModuleList([
            FCNetwork(input_dim + action_dim, 1,
                      hidden_layers=hidden_layers)
            for i in range(num_nets)
        ])

    def forward(self, input: Tensor, action: Tensor) -> Tensor:
        """Evaluate action value (Q).
        :returns: action value (Q)
        """
        input = torch.cat([input, action], dim=1)
        values = [net(input) for net in self.nets]
        return torch.cat(values, 1)


class DiscreteQNetwork(nn.Module):
    """Discrete action-value (Q) function with fully-connected (MLP) network."""

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 num_nets: int = 1):
        super().__init__()
        self.nets = nn.ModuleList([
            FCNetwork(input_dim, action_dim,
                      hidden_layers=hidden_layers)
            for i in range(num_nets)
        ])

    def forward(self, input: Tensor) -> Tensor:
        """Evaluate action values (Q).
        :returns: action values (Q)
        """
        values = [net(input) for net in self.nets]
        return torch.stack(values, 1)
