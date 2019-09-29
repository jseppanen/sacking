
import logging
import random
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn

from .environment import Env
from .policy import GaussianPolicy
from .typing import Transition

Batch = Dict[str, Tensor]

@torch.no_grad()
def sample_batch(replaybuf: List[Transition], size: int) \
        -> Batch:
    """Sample batch of transitions from replay buffer."""
    batch = random.sample(replaybuf, size)
    batch = map(np.stack, zip(*batch))  # convert to numpy and transpose
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
        self._ongoing_episode_length = 0
        self._ongoing_episode_reward = 0.0
        self.total_episodes = 0

    @torch.no_grad()
    def sample_episode(self, policy: Optional[GaussianPolicy] = None) \
            -> Generator[Transition, None, None]:
        """Sample one episode/trajectory/rollout from environment.
        """
        done = False
        while not done:
            tr, done = self._sample(policy)
            yield tr

    @torch.no_grad()
    def sample_transition(self, policy: Optional[GaussianPolicy] = None,
                          body: Optional[nn.Module] = None) \
            -> Transition:
        """Sample one transition/example from environment.
        """
        tr, _ = self._sample(policy, body)
        return tr

    @torch.no_grad()
    def _sample(self, policy: Optional[GaussianPolicy],
                body: Optional[nn.Module] = None) \
            -> Tuple[Transition, bool]:
        # sample action from the policy
        if policy:
            obs = np.array(self._observation)
            if body:
                # FIXME back and forth
                obs = torch.from_numpy(obs).unsqueeze(0)
                obs = body(obs)
                obs = obs.squeeze(0).numpy()
            action = policy.choose_action(obs)
        else:
            # sample random action
            action = self._env.action_space.sample()
        # XXX pong specific: alias to LEFT/RIGHT only
        action = (action % 2) + 2
        # sample transition from the environment
        next_observation, reward, done, _ = self._env.step(action)
        # XXX pong specific: restart life
        if reward != 0 and not done:
            next_observation, reward2, done, _ = self._env.step(1)
            reward += reward2
        self._ongoing_episode_length += 1
        self._ongoing_episode_reward += reward
        tr = Transition(self._observation,
                        action,
                        np.array([reward], dtype=np.float32),
                        next_observation,
                        np.array([done], dtype=bool))
        if done:
            self._observation = self._env.reset()
            self.total_episodes += 1
            logging.info(f'episode {self.total_episodes} '
                         f'length {self._ongoing_episode_length} '
                         f'reward {self._ongoing_episode_reward}')
            self._ongoing_episode_length = 0
            self._ongoing_episode_reward = 0.0
        else:
            self._observation = next_observation
        return tr, done
