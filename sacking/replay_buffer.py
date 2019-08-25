
import random
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .environment import Env
from .policy import GaussianPolicy
from .typing import Transition

Batch = Dict[str, Tensor]

@torch.no_grad()
def sample_batch(replaybuf: List[Transition], size: int) \
        -> Batch:
    """Sample batch of transitions from replay buffer."""
    batch = random.sample(replaybuf, size)
    batch = map(np.stack, zip(*batch))  # transpose
    batch = {k: torch.from_numpy(v)
             for k, v in zip(Transition._fields, batch)}
    for k in ['reward', 'terminal']:
        batch[k] = batch[k].squeeze(1)
    batch['terminal'] = batch['terminal'].byte()
    return batch


def initialize_replay_buffer(replaybuf: List[Transition],
                             env: Env,
                             num_initial_exploration_episodes: int,
                             num_initial_exploration_steps: int) -> None:
    """Fill replay buffer with episodes from random policy."""

    sampler = EnvSampler(env)
    num_episodes = 0
    while (num_episodes < num_initial_exploration_episodes
           or len(replaybuf) < num_initial_exploration_steps):
        replaybuf.extend(sampler.sample_episode())
        num_episodes += 1


class EnvSampler:
    """Run policy on env to get transition samples."""

    def __init__(self, env: Env):
        self._env = env
        self._observation = self._env.reset()

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
            action = policy.choose_action(self._observation)
        else:
            # sample random action
            action = self._env.action_space.sample()
        # sample transition from the environment
        next_observation, reward, done, _ = self._env.step(action)
        tr = Transition(self._observation.astype(np.float32),
                        action,
                        np.array([reward], dtype=np.float32),
                        next_observation.astype(np.float32),
                        np.array([done], dtype=bool))
        if done:
            self._observation = self._env.reset()
        else:
            self._observation = next_observation
        return tr, done
