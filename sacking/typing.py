
from typing import Dict, NamedTuple

import numpy as np
import torch


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
