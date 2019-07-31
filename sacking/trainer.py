
import copy
import logging
import os
import random
from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .policy import GaussianPolicy
from .q_network import QNetwork
from .typing import Checkpoint, Env, Transition


def train(policy: GaussianPolicy,
          q_network: QNetwork,
          env: Env,
          *,
          batch_size: int = 256,
          replay_buffer_size: int = int(1e6),
          learning_rate: float = 3e-4,
          discount: float = 0.99,
          num_steps: int = int(100e3),
          num_initial_exploration_steps: int = int(1e3),
          update_interval: int = 1,
          progress_interval: int = 1000,
          checkpoint_interval: int = 10000,
          target_network_update_weight: float = 5e-3,
          target_entropy: Optional[float] = None,
          rundir: str = 'runs') -> None:
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

    replaybuf = deque(maxlen=replay_buffer_size)
    env = torch_env(env)
    observation: Tensor = env.reset()
    done: bool = False

    def environment_step(step: int) -> None:
        nonlocal observation, done

        if done:
            observation = env.reset()

        # sample action from the policy
        with torch.no_grad():
            if step < num_initial_exploration_steps:
                action = env.action_space.sample()
                #action = torch.from_numpy(action)
                action = torch.tensor(action)
            else:
                action, _ = policy.sample_action(observation.unsqueeze(0))
                action = action.squeeze(0)

        # sample transition from the environment
        next_observation, reward, done, _ = env.step(action)
        replaybuf.append(
            Transition(observation.numpy(),
                       action.numpy(),
                       np.array([reward], dtype=np.float32),
                       next_observation.numpy(),
                       np.array([done], dtype=bool)))
        observation = next_observation

    def optimization_step(step: int) -> None:
        if step < num_initial_exploration_steps:
            return

        batch = sample_batch(replaybuf, batch_size)
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
            next_state_value *= (~batch['done']).float()
            target_q_value = batch['reward'] + discount * next_state_value

        # Update policy weights (Eq. 10)
        # NB. use old Q network that hasn't been updated for current batch
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
        # NB. Q network must be updated *after* policy update, because
        # we don't want to backprop policy update through a Q network
        # that is overfitted to current batch
        pred_q_value_dist = q_network(batch['observation'])
        ids = range(len(pred_q_value_dist))
        pred_q_values = pred_q_value_dist[ids, :, batch['action']]
        assert pred_q_values.dim() == 2 and pred_q_values.shape[1] == 2
        q_network_loss = F.mse_loss(
            pred_q_values,
            target_q_value.unsqueeze(1).expand(pred_q_values.shape)
        )
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
        environment_step(step)
        if step % update_interval == 0:
            optimization_step(step)
        if step > 0 and step % progress_interval == 0:
            metrics = validate(policy, env)
            for name in metrics:
                writer.add_scalar(f'eval/{name}', metrics[name], step)
            logging.info('step %d reward %f', step, metrics['episode_reward'])
        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(step)

    writer.close()


def validate(policy: GaussianPolicy, env: Env) \
        -> Dict[str, float]:
    """Validate policy on environment"""
    XXX messes with training episodes because its the same env!
    episode_reward = 0.0
    observation = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            action, _ = policy.best_action(observation.unsqueeze(0))
            observation, reward, done, _ = env.step(action.squeeze(0))
            episode_reward += reward
    return {'episode_reward': episode_reward}


def simulate(policy: GaussianPolicy, env: Env) \
        -> None:
    """Simulate policy on environment"""
    env = torch_env(env)
    observation = env.reset()
    env_done = False
    ui_active = True
    with torch.no_grad():
        while ui_active and not env_done:
            action, _ = policy.best_action(observation.unsqueeze(0))
            observation, reward, env_done, _ = env.step(action.squeeze(0))
            ui_active = env.render('human')


def torch_env(env: Env) -> Env:
    class TorchEnv:
        def __getattr__(self, key: str):
            return getattr(env, key)
        def reset(self) -> Tensor:
            obs = env.reset()
            obs = torch.from_numpy(obs.astype(np.float32))
            return obs
        def step(self, action) -> Tuple[Tensor, float, bool, Dict]:
            next_obs, reward, done, info = env.step(action.numpy())
            next_obs = torch.from_numpy(next_obs.astype(np.float32))
            return next_obs, reward, done, info
    return TorchEnv()


@torch.no_grad()
def sample_batch(replaybuf: Sequence[Transition], size: int) \
        -> Dict[str, Tensor]:
    """Sample batch of transitions from replay buffer."""
    batch = random.sample(replaybuf, size)
    batch = map(np.stack, zip(*batch))  # transpose
    batch = {k: torch.from_numpy(v)
             for k, v in zip(Transition._fields, batch)}
    for k in ['reward', 'done']:
        batch[k] = batch[k].squeeze(1)
    batch['done'] = batch['done'].byte()
    return batch


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
