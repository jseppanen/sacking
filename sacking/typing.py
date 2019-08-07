
from typing import Dict, NamedTuple, Optional, Union
from typing_extensions import Protocol

import numpy as np
import torch
from torch import FloatTensor


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
    terminal: np.ndarray  # bool


class Checkpoint(NamedTuple):
    """Policy checkpoint."""
    policy: Dict
    q_network: Dict
    log_alpha: np.float32
    policy_optimizer: Dict
    q_network_optimizer: Dict
    alpha_optimizer: Dict

    def save(self, path: str) -> None:
        """Save model checkpoint to disk."""
        state = self._asdict()
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> 'Checkpoint':
        """Load model checkpoint from disk."""
        state = torch.load(path)
        return cls(**state)
