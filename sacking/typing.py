
from typing import NamedTuple, Optional, Union
from typing_extensions import Protocol

import numpy as np
import torch
from torch import FloatTensor
from torch.nn import Module
from torch.optim import Optimizer


# actions can be discrete or continuous
Action = Union[int, float]


class Env(Protocol):

    def reset(self) -> 'EnvStep':
        """Start new episode
        """
        ...

    def seed(self, seed: Optional[int]) -> None:
        """Seed random number generator."""
        ...

    def step(self, action: Action) -> 'EnvStep':
        """Make one action in environment.
        """
        ...


class EnvStep(NamedTuple):
    """Tuple of environment step output"""
    observation: np.ndarray
    reward: float
    done: bool


class PolicyOutput(NamedTuple):
    """Tuple of policy outputs
    torch.trace only supports tensors and tuples
    """
    action: FloatTensor
    log_prob: Optional[FloatTensor] = None


class Transition(NamedTuple):
    """Environment interaction example(s)."""
    observation: np.ndarray  # float32
    action: np.ndarray  # float32
    reward: np.ndarray  # float32
    next_observation: np.ndarray  # float32
    done: np.ndarray  # bool


class Checkpoint(NamedTuple):
    """Policy checkpoint."""
    policy: Module
    q_network: Module
    log_alpha: np.float32
    policy_optimizer: Optimizer
    q_network_optimizer: Optimizer
    alpha_optimizer: Optimizer

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        state = {
            'policy': self.policy.state_dict(),
            'q_network': self.q_network.state_dict(),
            'log_alpha': self.log_alpha,
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_network_optimizer': self.q_network_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }
        torch.save(state, path)
