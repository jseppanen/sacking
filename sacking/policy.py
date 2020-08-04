
from typing import NamedTuple, Optional, Sequence, Tuple
from typing_extensions import Protocol

import numpy as np
import torch
from torch import FloatTensor, LongTensor, Tensor
from torch import nn
from torch.distributions import Normal, Categorical
from torch.nn.functional import softplus

from .fc_network import FCNetwork
from .typing import Checkpoint
from .measurements import Measurements

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_CONST = -0.5 * np.log(2.0 * np.pi)


class PolicyOutput(Protocol):
    action: Optional[Tensor] = None
    log_prob: FloatTensor

    def average(self, x: FloatTensor) -> FloatTensor:
        """Calculate expected value under action distribution."""
        ...


class SquashNormalSample(NamedTuple):
    """Reparametrized sample from squashed normal action distribution"""
    action: FloatTensor
    log_prob: FloatTensor

    def average(self, x: FloatTensor) -> FloatTensor:
        return x

    def entropy(self) -> FloatTensor:
        return -self.log_prob


class SquashNormalDistribution(NamedTuple):
    """Multivariate normal action distribution with optional tanh squashing."""

    action_mean: FloatTensor
    action_log_std: FloatTensor
    squash: bool

    def sample_action(self, *, measure: bool = False) -> SquashNormalSample:
        action_log_std = self.action_log_std.clamp(
            min=LOG_STD_MIN, max=LOG_STD_MAX)
        latent = torch.randn_like(self.action_mean)
        action = latent * action_log_std.exp() + self.action_mean
        if measure:
            Measurements.update({
                "policy/location": self.action_mean.flatten(),
                "policy/scale": action_log_std.exp().flatten(),
                "policy/raw_action": action.flatten(),
            })
        log_prob = (
            -0.5 * latent ** 2 - action_log_std + LOG_PROB_CONST
        ).sum(1)
        if self.squash:
            log_prob = log_prob - 2.0 * (
                np.log(2.0) - action - softplus(-2.0 * action)
            ).sum(1)
            action = torch.tanh(action)
        return SquashNormalSample(action, log_prob)

    def greedy_action(self) -> SquashNormalSample:
        action = self.action_mean.detach()
        log_prob = torch.zeros_like(action)
        if self.squash:
            action = torch.tanh(action)
        return SquashNormalSample(action, log_prob)

    def reparameterize(self, *, measure: bool = False) -> PolicyOutput:
        # reparameterization trick
        return self.sample_action(measure=measure)

    def log_prob(self, action: FloatTensor) -> FloatTensor:
        if self.squash:
            action = atanh(action)
        action_log_std = self.action_log_std.clamp(
            min=LOG_STD_MIN, max=LOG_STD_MAX)
        latent = (action - self.action_mean) / action_log_std.exp()
        log_prob = (
            -0.5 * latent ** 2 - action_log_std + LOG_PROB_CONST
        ).sum(1)
        if self.squash:
            log_prob = log_prob - 2.0 * (
                np.log(2.0) - action - softplus(-2.0 * action)
            ).sum(1)
        return log_prob


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

    def _action_distribution(self, observation: Tensor) \
            -> SquashNormalDistribution:
        """Calculate action distribution given observation"""
        stats = self.net(observation)
        mean, log_std = stats.chunk(2, dim=-1)
        return SquashNormalDistribution(mean, log_std, self._squash)

    def forward(self, observation: Tensor) -> SquashNormalDistribution:
        """Get .
        :returns: action and its log-probability
        """
        return self._action_distribution(observation)

    @torch.no_grad()
    def choose_action(self, observation: np.ndarray, *, greedy: bool = False) \
            -> Tuple[np.ndarray, np.float32]:
        """Choose action from policy for one observation."""
        pt_obs = torch.from_numpy(observation).unsqueeze(0).float()
        dist = self._action_distribution(pt_obs)
        sample = dist.greedy_action() if greedy else dist.sample_action()
        action = sample.action.squeeze(0).numpy()
        logp = sample.log_prob.numpy()[0]
        return action, logp

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


class DiscreteSample(NamedTuple):
    """Sample from discrete action distribution"""
    action: LongTensor
    log_prob: FloatTensor


class DiscreteDistribution(NamedTuple):
    """Discrete action distribution.
    """
    log_probs: FloatTensor

    def sample_action(self) -> DiscreteSample:
        dist = Categorical(logits=self.log_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return DiscreteSample(action, log_prob)

    def greedy_action(self) -> DiscreteSample:
        tmp, action = self.log_probs.max(dim=1)
        log_prob = torch.zeros_like(tmp)
        return DiscreteSample(action, log_prob)

    def reparameterize(self, *, measure: bool = False) -> "DiscreteParameters":
        return DiscreteParameters(*self)

    def log_prob(self, action: LongTensor) -> FloatTensor:
        ii = torch.arange(action.shape[0])
        return self.log_probs[ii, action]


class DiscreteParameters(NamedTuple):
    """Action distribution from discrete policy
    torch.trace only supports tensors and tuples
    """
    log_prob: FloatTensor

    def average(self, x: FloatTensor) -> FloatTensor:
        """Calculate expected value under action distribution."""
        return (self.log_prob.exp() * x).sum(1)

    def entropy(self) -> FloatTensor:
        return -self.average(self.log_prob)


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

    def _action_distribution(self, observation: Tensor) \
            -> DiscreteDistribution:
        """Calculate action distribution given observation"""
        logits = self.net(observation)
        logprobs = logits - logits.logsumexp(dim=-1, keepdim=True)
        return DiscreteDistribution(logprobs)

    def forward(self, observation: Tensor) -> DiscreteDistribution:
        """Get action distribution given observation.
        :returns: action distribution
        """
        return self._action_distribution(observation)

    @torch.no_grad()
    def choose_action(self, observation: np.ndarray, *, greedy: bool = False) \
            -> Tuple[np.ndarray, np.float32]:
        """Choose action from policy for one observation."""
        pt_obs = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0)
        dist = self._action_distribution(pt_obs)
        sample = dist.greedy_action() if greedy else dist.sample_action()
        action = sample.action.squeeze(0).numpy()
        logp = sample.log_prob.numpy()[0]
        return action, logp

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> 'DiscretePolicy':
        """Restore discrete policy from model checkpoint."""
        # FIXME hacky way to recover layers on pytorch 1.1
        state = checkpoint.policy
        num_layers = len(state) // 2
        _, input_dim = state['net.net.0.weight'].shape
        output_dim = len(state[f'net.net.{2 * (num_layers - 1)}.bias'])
        action_dim = output_dim
        hidden_layers = [
            len(state[f'net.net.{2 * i}.bias'])
            for i in range(num_layers - 1)
        ]
        policy = cls(input_dim=input_dim, action_dim=action_dim,
                     hidden_layers=hidden_layers)
        policy.load_state_dict(state)
        return policy


def atanh(x: FloatTensor) -> FloatTensor:
    return 0.5 * (x.log1p() - (-x).log1p())
