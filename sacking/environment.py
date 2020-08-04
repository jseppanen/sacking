
from typing import Dict, NamedTuple, Optional
from typing_extensions import Protocol

import gym
import numpy as np


class Env(Protocol):

    def reset(self) -> np.ndarray:
        """Start new episode
        """
        ...

    def seed(self, seed: Optional[int]) -> None:
        """Seed random number generator."""
        ...

    def step(self, action: np.ndarray) -> 'EnvStep':
        """Make one action in environment.
        """
        ...


class EnvStep(NamedTuple):
    """Tuple of environment step output"""
    observation: np.ndarray
    reward: float
    terminal: bool
    info: Dict


def load_env(full_name: str) -> Env:
    """Load environment"""
    assert '/' in full_name
    env_ns, env_name = full_name.split('/', 1)
    if env_ns == 'gym':
        return gym.make(env_name)
    elif env_ns == 'roboschool':
        import roboschool
        return gym.make(env_name)
    else:
        raise ValueError(f'unknown env: {full_name}')


class NormalizedActionEnv(gym.Wrapper):
    """Normalize continuous actions to (-1, 1)."""

    def __init__(self, env) -> None:
        super().__init__(env)
        spc = env.action_space
        if isinstance(spc, gym.spaces.Box):
            dtype = spc.dtype
            shape = spc.shape
            ones = np.ones(shape, dtype)
            self.action_space = gym.spaces.Box(-ones, ones, shape, dtype)

    def step(self, action: np.ndarray) -> "EnvStep":
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = 0.5 * (action + 1.0)
            action = action * self.env.action_space.high + (1.0 - action) * self.env.action_space.low
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return self.env.step(action)
