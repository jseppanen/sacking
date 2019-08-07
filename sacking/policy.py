
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus

from .fc_network import FCNetwork
from .typing import Checkpoint, PolicyOutput

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_CONST = -0.5 * np.log(2.0 * np.pi)


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
            action_log_std = action_log_std.clamp(min=LOG_STD_MIN,
                                                  max=LOG_STD_MAX)
            latent = torch.randn_like(action_mean)
            action = latent * action_log_std.exp() + action_mean
            log_prob = (
                -0.5 * latent ** 2 - action_log_std + LOG_PROB_CONST
            ).sum(1)
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

    def choose_action(self, observation: np.ndarray, *, mode: str = 'sample') \
            -> np.ndarray:
        """Choose action from policy for one observation."""
        with torch.no_grad():
            pt_obs = torch.from_numpy(observation).unsqueeze(0)
            pt_action, _ = self.forward(pt_obs)
            action = pt_action.squeeze(0).numpy()
            return action

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
