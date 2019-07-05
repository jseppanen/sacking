
import copy
import random
from collections import deque
from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .policy import GaussianPolicy, QNetwork, StackedModule
from .typing import Env, Transition


def train(policy: GaussianPolicy,
          q_network: Union[QNetwork, Sequence[QNetwork]],
          env: Env,
          *,
          batch_size: int = 256,
          replay_buffer_size: int = int(1e6),
          learning_rate: float = 0.001,
          discount: float = 0.99,
          num_steps: int = int(100e3),
          num_initial_exploration_steps: int = int(1e3),
          update_interval: int = 1,
          target_network_update_weight: float = 0.1,
          target_entropy: Optional[float] = None) -> None:
    """Train policy and Q-network with soft actor-critic (SAC)
    :param update_interval: Environment steps between each optimization step
    """

    # stack twin Q functions and create target network
    q_networks = StackedModule(q_network if isinstance(q_network, Sequence)
                               else [q_network])
    target_q_networks = copy.deepcopy(q_networks)

    log_alpha = torch.tensor([0.0], requires_grad=True)

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    q_networks_optimizer = optim.Adam(q_networks.parameters(), lr=learning_rate)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

    replaybuf = deque(maxlen=replay_buffer_size)
    observation, _, done = env.reset()

    def environment_step(step: int) -> None:
        nonlocal observation, done

        while done:
            observation, _, done = env.reset()

        # sample action from the policy
        with torch.no_grad():
            if step < num_initial_exploration_steps:
                action = 2 * torch.rand(1) - 1
            else:
                action = policy(observation).action

        # sample transition from the environment
        next_observation, reward, done = env.step(action)
        replaybuf.append(
            Transition(observation, action, reward, next_observation, done))
    
    def optimization_step(step: int) -> None:
        if step < num_initial_exploration_steps:
            return

        batch = sample_batch(replaybuf, batch_size)
        alpha = log_alpha.exp().detach()

        # Update Q-function parameters (Eq. 6)
        with torch.no_grad():
            next_action, next_action_log_prob = policy(batch.next_observation)
            next_q_values = target_q_networks(batch.next_observation, next_action)
            next_state_value = next_q_values.min(0) - alpha * next_action_log_prob
            next_state_value *= ~batch.done
            target_q_value = batch.reward + discount * next_state_value

        pred_q_values = q_networks(batch.observation, batch.action)
        q_networks_loss = F.mse_loss(pred_q_values, target_q_value[None, :])
        q_networks_optimizer.zero_grad()
        q_networks_loss.backward()
        q_networks_optimizer.step()

        # Update policy weights (Eq. 10)
        action, action_log_prob = policy(batch.observation)
        q_values = q_networks(batch.observation, action)
        state_value = q_values.min(0) - alpha * action_log_prob
        policy_loss = -state_value.mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Adjust temperature (Eq. 18)
        alpha_loss = -log_alpha * (action_log_prob + target_entropy).detach()
        alpha_loss = alpha_loss.mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # Update target Q-network weights
        soft_update(target_q_networks, q_networks,
                    target_network_update_weight)

    # main loop
    for step in range(num_steps):
        environment_step(step)
        if step % update_interval == 0:
            optimization_step(step)


@torch.no_grad()
def sample_batch(replaybuf: Sequence[Transition], size: int) -> Transition:
    """Sample batch of transitions from replay buffer."""
    batch = random.sample(replaybuf, size)
    batch = map(torch.cat, zip(*batch))  # transpose
    return Transition(*batch)


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
