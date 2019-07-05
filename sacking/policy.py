
from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal

from .typing import PolicyOutput

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class FCNetwork(nn.Module):
    """Fully-connected (MLP) network."""

    def __init__(self, input_dim: int, output_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        super().__init__()

        sizes = [input_dim] + list(hidden_layers) + [output_dim]
        layers = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s1, s2))
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
q        """
        super().__init__()
        self.net = FCNetwork(input_dim, action_dim,
                             hidden_layers=hidden_layers)
        self._squash = squash

    def forward(self, input: Tensor) -> PolicyOutput:
        """Sample action from policy.
        :returns: action and its log-probability
        """
        action_mean, action_log_std = self.net(input).chunk(2, dim=1)
        action_log_std = action_log_std.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX)
        action_dist = Normal(action_mean, action_log_std.exp())
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action).sum(1)

        if self._squash:
            action = torch.tanh(action)
            log_prob = log_prob - torch.log1p(-(action ** 2) + EPS).sum(1)

        return PolicyOutput(action, log_prob)


class QNetwork(nn.Module):
    """Action-value (Q) function with fully-connected (MLP) network."""

    def __init__(self, input_dim: int, action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        super().__init__(self)
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
        self.modules = nn.ModuleList(modules)

    def forward(self, *inputs) -> Tensor:
        return torch.cat([m(*inputs) for m in self.modules])
