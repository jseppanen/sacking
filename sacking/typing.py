
from typing import NamedTuple, Optional, Union
from typing_extensions import Protocol

import numpy as np
from torch import FloatTensor


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
