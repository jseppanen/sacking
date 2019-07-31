
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
from torch import FloatTensor, LongTensor, Tensor
from torch import nn
from torch.distributions import Normal, Categorical
from torch.nn.functional import softplus

from .fc_network import FCNetwork
from .typing import Checkpoint

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_CONST = -0.5 * np.log(2.0 * np.pi)


class GaussianAction(NamedTuple):
    """Tuple of policy outputs
    torch.trace only supports tensors and tuples
    """
    action: FloatTensor
    log_prob: Optional[FloatTensor] = None


class GaussianStats(NamedTuple):
    """Tuple of policy outputs
    torch.trace only supports tensors and tuples
    """
    action_mean: FloatTensor
    action_log_std: FloatTensor


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

    def forward(self, observation: Tensor) -> GaussianStats:
        """Sample action from policy.
        :returns: action and its log-probability
        """
        action_stats = self.net(observation)
        action_mean, action_log_std = action_stats.chunk(2, dim=-1)
        return GaussianStats(action_mean, action_log_std)

    def sample_action(self, observation: Tensor) -> GaussianAction:
        """Sample action.
        Implements the reparametrization trick
        """
        action_mean, action_log_std = self.forward(observation)
        action_log_std = action_log_std.clamp(min=LOG_STD_MIN,
                                              max=LOG_STD_MAX)
        latent = torch.randn_like(action_mean)
        action = latent * action_log_std.exp() + action_mean
        log_prob = (
            -0.5 * latent ** 2 - action_log_std + LOG_PROB_CONST
        ).sum(1)
        if self._squash:
            log_prob = log_prob - 2.0 * (
                np.log(2.0) - action - softplus(-2.0 * action)
            ).sum(1)
            action = torch.tanh(action)
        return GaussianAction(action, log_prob)

    def best_action(self, observation: Tensor) -> GaussianAction:
        """Choose deterministic best action."""
        action_mean, action_log_std = self.forward(observation)
        action = action_mean.detach()
        log_prob = torch.zeros_like(action)
        if self._squash:
            action = torch.tanh(action)
        return GaussianAction(action, log_prob)

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


class DiscreteAction(NamedTuple):
    """Tuple of policy outputs
    torch.trace only supports tensors and tuples
    """
    action: LongTensor
    log_prob: Optional[FloatTensor] = None


DiscreteStats = FloatTensor


class DiscretePolicy(nn.Module):
    """Discrete policy with fully-connected (MLP) network."""

    def __init__(self, input_dim: int,
                 action_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        """Create discrete policy.
        :param input_dim: Input dimension
        :param action_dim: Action dimension
        """
        super().__init__()
        self.net = FCNetwork(input_dim, action_dim,
                             hidden_layers=hidden_layers)

    def forward(self, input: Tensor) -> DiscreteStats:
        """Sample action from policy.
        :returns: action and its log-probability
        """
        logits = self.net(input)
        logprobs = logits - logits.logsumexp(dim=-1, keepdim=True)
        return logprobs

    def sample_action(self, observation: Tensor) -> DiscreteAction:
        """Sample action.
        """
        action_log_probs = self.forward(observation)
        dist = Categorical(logits=action_log_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return DiscreteAction(action, log_prob)

    def best_action(self, observation: Tensor) -> DiscreteAction:
        """Choose deterministic best action."""
        action_log_probs = self.forward(observation)
        _, action = action_log_probs.max(dim=1)
        log_prob = torch.zeros_like(action)
        return DiscreteAction(action, log_prob)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> 'DiscretePolicy':
        """Restore discrete policy from model checkpoint."""
        # FIXME hacky way to recover layers on pytorch 1.1
        state = checkpoint.policy
        num_layers = len(state) // 2
        _, input_dim = state['net.net.0.weight'].shape
        action_dim = len(state[f'net.net.{(num_layers - 1)}.bias'])
        hidden_layers = [
            len(state[f'net.net.{2 * i}.bias'])
            for i in range(num_layers - 1)
        ]
        policy = cls(input_dim=input_dim, action_dim=action_dim,
                     hidden_layers=hidden_layers)
        policy.load_state_dict(state)
        return policy
