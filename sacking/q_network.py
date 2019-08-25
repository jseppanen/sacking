
from typing import Optional, Sequence

import torch
from torch import FloatTensor, LongTensor
from torch import nn

from .fc_network import FCNetwork


class QNetwork(nn.Module):
    """Continuous parametric action-value (Q) function with fully-connected (MLP) network."""

    is_parametric = True

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 num_nets: int = 1):
        super().__init__()
        self.nets = nn.ModuleList([
            FCNetwork(input_dim + action_dim, 1,
                      hidden_layers=hidden_layers)
            for i in range(num_nets)
        ])

    def forward(self, observation: FloatTensor, action: FloatTensor) \
            -> FloatTensor:
        """Evaluate action value (Q).
        :returns: action value (Q)
        """
        observation = torch.cat([observation, action], dim=1)
        values = [net(observation) for net in self.nets]
        return torch.cat(values, 1)

    @property
    def num_nets(self) -> int:
        """Number of Q networks."""
        return len(self.nets)


class DiscreteQNetwork(nn.Module):
    """Discrete action-value (Q) function with fully-connected (MLP) network."""

    is_parametric = False

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 num_nets: int = 1):
        super().__init__()
        self.nets = nn.ModuleList([
            FCNetwork(input_dim, action_dim,
                      hidden_layers=hidden_layers)
            for i in range(num_nets)
        ])

    def forward(self, observation: FloatTensor,
                action: Optional[LongTensor] = None) \
            -> FloatTensor:
        """Evaluate action values (Q).
        :returns: action values (Q)
        """
        values = [net(observation) for net in self.nets]
        values = torch.stack(values, 1)
        if action is not None:
            # gather Q values corresponding to individual actions
            ids = range(len(values))
            values = values[ids, :, action]
        return values

    @property
    def num_nets(self) -> int:
        """Number of Q networks."""
        return len(self.nets)
