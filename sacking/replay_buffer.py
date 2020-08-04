import random
from collections import deque
from functools import singledispatch
from typing import Dict, Generator, NamedTuple, Optional, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from .environment import Env
from .measurements import Measurements
from .policy import GaussianPolicy

Batch = Dict[str, Tensor]


class Transition(NamedTuple):
    """Environment interaction example(s)."""
    observation: np.ndarray  # float32
    action: np.ndarray  # float32
    reward: np.ndarray  # float32
    next_observation: np.ndarray  # float32
    terminal: np.ndarray  # bool
    log_propensity: np.ndarray  # float32


class ReplayBuffer(deque):
    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen=maxlen)

    def initialize(self, sampler: "Datagen",
                   num_initial_exploration_episodes: int,
                   num_initial_exploration_steps: int) -> None:
        """Fill replay buffer with episodes from random policy."""

        num_episodes = 0
        while (num_episodes < num_initial_exploration_episodes
                or len(self) < num_initial_exploration_steps):
            self.extend(sampler.sample_episode())
            num_episodes += 1

    @torch.no_grad()
    def sample_batch(self, size: int) -> Batch:
        """Sample batch of transitions from replay buffer."""
        batch = random.sample(self, size)
        batch = map(np.stack, zip(*batch))  # transpose
        batch = {k: torch.from_numpy(v)
                 for k, v in zip(Transition._fields, batch)}
        for k in ['reward', 'terminal', "log_propensity"]:
            batch[k] = batch[k].squeeze(1)
        batch['terminal'] = batch['terminal'].bool()
        return batch


class Datagen:
    """Training data generator, runs policy on env to get transition samples.

    Also known as "experience collection."
    """

    def __init__(self, env: Env):
        self._env = env
        self._observation = self._env.reset()
        self._ongoing_episode_length = 0
        self._ongoing_episode_return = 0.0
        self.total_episodes = 0
        self.total_steps = 0

    def sample_episode(self, policy: Optional[GaussianPolicy] = None) \
            -> Generator[Transition, None, None]:
        """Sample one episode/trajectory/rollout from environment.
        """
        done = False
        while not done:
            tr, done = self._sample(policy)
            yield tr

    def sample_transition(self, policy: Optional[GaussianPolicy] = None) \
            -> Transition:
        """Sample one transition/example from environment.
        """
        tr, _ = self._sample(policy)
        return tr

    def _sample(self, policy: Optional[GaussianPolicy]) \
            -> Tuple[Transition, bool]:
        # sample action from the policy
        if policy:
            action, logp = policy.choose_action(self._observation)
        else:
            # sample random action
            action, logp = random_action(self._env.action_space)
        # sample transition from the environment
        next_observation, reward, done, _ = self._env.step(action)
        self.total_steps += 1
        self._ongoing_episode_length += 1
        self._ongoing_episode_return += reward
        if self._ongoing_episode_length == self._env.spec.max_episode_steps:
            assert done
            # reaching time limit is not true episode termination
            terminal = False
        else:
            terminal = done
        tr = Transition(self._observation.astype(np.float32),
                        action,
                        np.array([reward], dtype=np.float32),
                        next_observation.astype(np.float32),
                        np.array([terminal], dtype=bool),
                        np.array([logp], dtype=np.float32),
        )
        if done:
            self._observation = self._env.reset()
            self.total_episodes += 1
            Measurements.update({
                "datagen/episode_length": self._ongoing_episode_length,
                "datagen/episode_return": self._ongoing_episode_return,
            })
            self._ongoing_episode_length = 0
            self._ongoing_episode_return = 0.0
        else:
            self._observation = next_observation
        return tr, done


@singledispatch
def random_action(space: gym.spaces.Space) -> Tuple[np.ndarray, np.float32]:
    """Sample random action from Gym action space."""
    raise NotImplementedError(f"not supported: {type(space)}")


@random_action.register
def _(space: gym.spaces.Box) -> Tuple[np.ndarray, np.float32]:
    assert space.is_bounded()
    sample = np.random.uniform(space.low, space.high, space.shape).astype(space.dtype)
    logprob = -np.log(space.high - space.low).sum().astype(np.float32)
    return sample, logprob


@random_action.register
def _(space: gym.spaces.Discrete) -> Tuple[np.ndarray, np.float32]:
    sample = np.random.randint(space.n)
    logprob = -np.log(space.n).astype(np.float32)
    return sample, logprob
