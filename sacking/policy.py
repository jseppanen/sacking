
from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal
from torch.nn import init
from torch.nn.functional import softplus

from .typing import Checkpoint, PolicyOutput

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class FCNetwork(nn.Module):
    """Fully-connected (MLP) network."""

    def __init__(self, input_dim: int, output_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        super().__init__()

        sizes: List[int] = [input_dim] + list(hidden_layers) + [output_dim]
        layers: List[nn.Module] = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            fc = nn.Linear(s1, s2)
            # softlearning initialization
            init.xavier_uniform_(fc.weight.data)
            fc.bias.data.fill_(0.0)
            layers.append(fc)
            # softlearning activation
            layers.append(nn.ReLU(inplace=True))
        # remove final ReLU
        del layers[-1]
        self.net = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class GaussianPolicy(nn.Module):
    """Gaussian policy with fully-connected (MLP) network."""

    def __init__(self, input_dim: int,
                 action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 squash: bool = True):
        """Create Gaussian policy.
        :param input_dim: Input dimension
        :param action_dim: Action dimension
        :param squash: Whether to squash actions to [-1, 1] range
        """
        super().__init__()
        self.net = FCNetwork(input_dim, 2 * action_dim,
                             hidden_layers=hidden_layers)
        self._squash = squash

    def forward(self, input: Tensor, *,
                mode: str = 'sample') -> PolicyOutput:
        """Sample action from policy.
        :returns: action and its log-probability
        """
        action_stats = self.net(input)
        action_mean, action_log_std = action_stats.chunk(2, dim=-1)
        if mode == 'sample':
            action_log_std = action_log_std.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX)
            action_dist = Normal(action_mean, action_log_std.exp())
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(1)
        elif mode == 'best':
            action = action_mean.detach()
            log_prob = torch.zeros_like(action)
        else:
            raise ValueError(mode)

        if self._squash:
            if mode == 'sample':
                log_prob = log_prob - 2.0 * (
                    np.log(2.0) - action - softplus(-2.0 * action)
                ).sum(1)
            action = torch.tanh(action)
        return PolicyOutput(action, log_prob)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> 'GaussianPolicy':
        """Restore Gaussian policy from model checkpoint."""
        # FIXME hacky way to recover layers on pytorch 1.1
        # FIXME squash not recovered but hardcoded to True
        state = checkpoint.policy
        num_layers = len(state) // 2
        _, input_dim = state['net.net.0.weight'].shape
        output_dim = len(state[f'net.net.{2 * (num_layers - 1)}.bias'])
        action_dim = output_dim // 2
        hidden_layers = [
            len(state[f'net.net.{2 * i}.bias'])
            for i in range(num_layers - 1)
        ]
        policy = cls(input_dim=input_dim, action_dim=action_dim,
                     hidden_layers=hidden_layers)
        policy.load_state_dict(state)
        return policy


class QNetwork(nn.Module):
    """Action-value (Q) function with fully-connected (MLP) network."""

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        super().__init__()
        self.net = FCNetwork(input_dim + action_dim, 1,
                             hidden_layers=hidden_layers)

    def forward(self, input: Tensor, action: Tensor) -> Tensor:
        """Evaluate action value (Q).
        :returns: action value (Q)
        """
        input = torch.cat([input, action], dim=1)
        value = self.net(input).squeeze(1)
        return value


class StackedModule(nn.Module):
    def __init__(self, modules: Sequence[nn.Module]):
        super().__init__()
        self.submodules = nn.ModuleList(modules)

    def forward(self, *inputs) -> Tensor:
        return torch.stack([m(*inputs) for m in self.submodules])
