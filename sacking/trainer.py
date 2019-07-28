
import copy
import logging
import os
import random
from collections import deque
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .policy import GaussianPolicy, QNetwork, StackedModule
from .typing import Checkpoint, Env, Transition

from .repro import approx, repro_batch, compare_weights


def train(policy: GaussianPolicy,
          q_network: Union[QNetwork, Sequence[QNetwork]],
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

    # check weights
    import pickle
    slinit = pickle.load(open('repro/init.pkl', 'rb'))
    for i in range(2):
        compare_weights(q_network[i].net.net, slinit['Q_weights'][i])
    compare_weights(policy.net.net, slinit['policy_weights'])

    # stack twin Q functions and create target network
    q_networks = StackedModule(q_network if isinstance(q_network, Sequence)
                               else [q_network])
    target_q_networks = copy.deepcopy(q_networks)

    log_alpha = torch.tensor([0.0], requires_grad=True)

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    q_networks_optimizer = optim.Adam(q_networks.parameters(), lr=learning_rate)
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
            if 1: # XXX step < num_initial_exploration_steps:
                action = env.action_space.sample()
                action = torch.from_numpy(action)
            else:
                action, _ = policy(observation.unsqueeze(0))
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

        # XXX repro
        # batch = sample_batch(replaybuf, batch_size)
        batch, slvalue = repro_batch()
        alpha = log_alpha.exp().detach()

        # Update Q-function parameters (Eq. 6)
        with torch.no_grad():
            next_action, next_action_log_prob = policy(batch['next_observation'])
            assert approx(batch['next_observation']) == slvalue['Q_target_next_observations'][0]
            assert approx(next_action) == slvalue['Q_target_next_actions']
            assert approx(next_action_log_prob) == slvalue['Q_target_next_log_pis'].flatten()
            next_q_values = target_q_networks(batch['next_observation'], next_action)
            assert approx(next_q_values) == slvalue['Q_target_next_Q_values'][:,:,0]
            next_state_value = next_q_values.min(0)[0] - alpha * next_action_log_prob
            assert approx(next_state_value) == slvalue['Q_target_next_values'].flatten()
            next_state_value *= (~batch['done']).float()
            target_q_value = batch['reward'] + discount * next_state_value
            assert approx(target_q_value) == slvalue['Q_target'].flatten()

        pred_q_values = q_networks(batch['observation'], batch['action'])
        assert approx(pred_q_values[0]) == slvalue['Q_values'][0].flatten()
        assert approx(pred_q_values[1]) == slvalue['Q_values'][1].flatten()

        q_networks_loss = 0.5 * (
            F.mse_loss(pred_q_values[0], target_q_value)
            + F.mse_loss(pred_q_values[1], target_q_value)
        )
        q_networks_optimizer.zero_grad()
        q_networks_loss.backward()
        q_networks_optimizer.step()

        # Update policy weights (Eq. 10)
        action, action_log_prob = policy(batch['observation'])
        q_values = q_networks(batch['observation'], action)
        state_value = q_values.min(0)[0] - alpha * action_log_prob
        policy_loss = -state_value.mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Adjust temperature (Eq. 18)
        # NB. disagreement between paper and softlearning implementation
        alpha_loss = -log_alpha.exp() * (action_log_prob + target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # Update target Q-network weights
        soft_update(target_q_networks, q_networks,
                    target_network_update_weight)


    def save_checkpoint(step: int) -> None:
        os.makedirs(f'{rundir}/checkpoints', exist_ok=True)
        path = f'{rundir}/checkpoints/checkpoint.{step:06d}.pt'
        cp = Checkpoint(policy.state_dict(),
                        q_networks.state_dict(),
                        log_alpha.detach().numpy(),
                        policy_optimizer.state_dict(),
                        q_networks_optimizer.state_dict(),
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
    episode_reward = 0.0
    observation = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            action, _ = policy(observation.unsqueeze(0), mode='best')
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
            action, _ = policy(observation.unsqueeze(0), mode='best')
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
