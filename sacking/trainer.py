
import copy
import logging
import os
from collections import deque
from typing import Dict, Optional, List

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .environment import Env
from .policy import GaussianPolicy
from .q_network import QNetwork
from .replay_buffer import Batch, EnvSampler, initialize_replay_buffer, sample_batch
from .typing import Checkpoint, Transition


def train(policy: GaussianPolicy,
          q_network: QNetwork,
          env: Env,
          *,
          batch_size: int = 256,
          replay_buffer_size: int = int(1e6),
          learning_rate: float = 3e-4,
          discount: float = 0.99,
          num_steps: int = int(100e3),
          num_initial_exploration_episodes: int = 10,
          update_interval: int = 1,
          progress_interval: int = 1000,
          checkpoint_interval: int = 10000,
          target_network_update_weight: float = 5e-3,
          target_entropy: Optional[float] = None,
          rundir: str = 'runs',
          validation_env: Optional[Env] = None) -> None:
    """Train policy and Q-network with soft actor-critic (SAC)

    Reference: Haarnoja et al, Soft Actor-Critic Algorithms and Applications
    https://arxiv.org/pdf/1812.05905.pdf

    :param update_interval: Environment steps between each optimization step
    """

    if target_entropy is None:
        target_entropy = -np.prod(env.action_space.shape)

    os.makedirs(rundir, exist_ok=True)
    writer = SummaryWriter(rundir)

    # create target network
    target_q_network = copy.deepcopy(q_network)

    log_alpha = torch.tensor([0.0], requires_grad=True)

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    q_network_optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

    replaybuf: List[Transition] = deque(maxlen=replay_buffer_size)
    initialize_replay_buffer(replaybuf, env, num_initial_exploration_episodes, batch_size)
    sampler = EnvSampler(env)

    def optimization_step(batch: Batch) -> None:

        alpha = log_alpha.exp().detach()

        # Update Q targets
        # NB. use old policy that hasn't been updated for current batch
        with torch.no_grad():
            next_action_log_probs = policy(batch['next_observation'])
            next_q_values = target_q_network(batch['next_observation'])
            next_min_q_value = next_q_values.min(1)[0]
            next_state_value_dist = next_min_q_value - alpha * next_action_log_probs
            next_state_value = (
                next_action_log_probs.exp() * next_state_value_dist
            ).sum(1)
            next_state_value *= (~batch['terminal']).float()
            target_q_value = batch['reward'] + discount * next_state_value
            target_q_values = target_q_value.unsqueeze(1).expand(next_q_values.shape)

        # Update policy weights (Eq. 10)
        action_log_probs = policy(batch['observation'])
        q_values = q_network(batch['observation'])
        min_q_value = q_values.min(1)[0]
        state_value_dist = min_q_value - alpha * action_log_probs
        state_value = (
            action_log_probs.exp() * state_value_dist
        ).sum(1)
        policy_loss = -state_value.mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update Q-function parameters (Eq. 6)
        pred_q_value_dist = q_network(batch['observation'])
        ids = range(len(pred_q_value_dist))
        pred_q_values = pred_q_value_dist[ids, :, batch['action']]
        assert pred_q_values.shape == target_q_values.shape
        q_network_loss = F.mse_loss(pred_q_values, target_q_values)
        q_network_optimizer.zero_grad()
        q_network_loss.backward()
        q_network_optimizer.step()

        # Adjust temperature (Eq. 18)
        # NB. paper uses alpha (not log) but we follow the softlearning impln
        # see also https://github.com/rail-berkeley/softlearning/issues/37
        action_entropy = (action_log_probs.exp() * action_log_probs).sum(1)
        alpha_loss = (
            -log_alpha * (action_entropy + target_entropy).detach()
        ).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # Update target Q-network weights
        soft_update(target_q_network, q_network,
                    target_network_update_weight)

    def save_checkpoint(step: int) -> None:
        os.makedirs(f'{rundir}/checkpoints', exist_ok=True)
        path = f'{rundir}/checkpoints/checkpoint.{step:06d}.pt'
        cp = Checkpoint(policy.state_dict(),
                        q_network.state_dict(),
                        log_alpha.detach().numpy(),
                        policy_optimizer.state_dict(),
                        q_network_optimizer.state_dict(),
                        alpha_optimizer.state_dict())
        cp.save(path)
        logging.info('saved model checkpoint to %s', path)

    # main loop
    for step in range(num_steps):
        # environment step
        transition = sampler.sample_transition(policy)
        replaybuf.append(transition)
        # optimization step
        if step % update_interval == 0:
            batch = sample_batch(replaybuf, batch_size)
            optimization_step(batch)
        if step > 0 and step % progress_interval == 0:
            if validation_env:
                metrics = validate(policy, validation_env)
                for name in metrics:
                    writer.add_scalar(f'eval/{name}', metrics[name], step)
                logging.info('step %d reward %f', step, metrics['episode_reward'])
        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(step)

    writer.close()


def validate(policy: GaussianPolicy, env: Env) \
        -> Dict[str, float]:
    """Validate policy on environment
    NB mutates env state!
    """
    episode_reward = 0.0
    observation = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            action = policy.choose_action(observation, mode='best')
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
    return {'episode_reward': episode_reward}


def simulate(policy: GaussianPolicy, env: Env) \
        -> None:
    """Simulate policy on environment
    NB mutates env state!
    """
    observation = env.reset()
    env_done = False
    ui_active = True
    with torch.no_grad():
        while ui_active and not env_done:
            action = policy.choose_action(observation, mode='best')
            observation, _, env_done, _ = env.step(action)
            ui_active = env.render('human')


@torch.no_grad()
def soft_update(target_network: nn.Module, network: nn.Module, tau: float) \
        -> None:
    """Update target network (Polyak averaging)."""
    for targ_param, param in zip(target_network.parameters(),
                                 network.parameters()):
        if targ_param is param:
            continue
        new_param = tau * param.data + (1.0 - tau) * targ_param.data
        targ_param.data.copy_(new_param)
