
import copy
import logging
import os
from typing import Dict, Optional, List

import numpy as np
import torch
from torch import nn
from torch import optim
from torch import FloatTensor, LongTensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

from .environment import Env, NormalizedActionEnv
from .measurements import Measurements
from .policy import GaussianPolicy, PolicyOutput
from .q_network import QNetwork
from .replay_buffer import Batch, Datagen, ReplayBuffer
from .typing import Checkpoint


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
          temperature: Optional[float] = None,
          target_entropy: Optional[float] = None,
          rundir: str = 'runs',
          validation_env: Optional[Env] = None) -> None:
    """Train policy and Q-network with soft actor-critic (SAC)

    Reference: Haarnoja et al, Soft Actor-Critic Algorithms and Applications
    https://arxiv.org/pdf/1812.05905.pdf

    :param update_interval: Environment steps between each optimization step
    """

    optimize_alpha = bool(temperature is None)
    if optimize_alpha and target_entropy is None:
        import gym
        if isinstance(env.action_space, gym.spaces.Box):
            target_entropy = -np.prod(env.action_space.shape)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            target_entropy = -np.log(env.action_space.n)
        else:
            raise TypeError(env.action_space)
        temperature = 1.0
    elif temperature is not None and target_entropy is not None:
        raise TypeError("use only one of temperature or target_entropy, not both")

    env = NormalizedActionEnv(env)
    if validation_env:
        validation_env = NormalizedActionEnv(validation_env)

    os.makedirs(rundir, exist_ok=True)
    writer = SummaryWriter(rundir)
    measurements = Measurements()
    if wandb and wandb.run:
        wandb.config.update(dict(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            learning_rate=learning_rate,
            discount=discount,
            num_steps=num_steps,
            num_initial_exploration_episodes=num_initial_exploration_episodes,
            target_network_update_weight=target_network_update_weight,
            target_entropy=target_entropy,
        ))

    # create target network
    target_q_network = copy.deepcopy(q_network)

    log_alpha = torch.tensor([np.log(temperature)], requires_grad=optimize_alpha)

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    q_network_optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    alpha_optimizer = optim.Adam([log_alpha], lr=learning_rate)

    replaybuf = ReplayBuffer(replay_buffer_size)
    sampler = Datagen(env)
    replaybuf.initialize(sampler, num_initial_exploration_episodes, batch_size)

    def calc_state_value(q_network: QNetwork,
                         observation: FloatTensor,
                         output: PolicyOutput,
                         alpha: FloatTensor):
        """Calculate state value from policy output."""
        q_values = (q_network(observation, output.action)
                    if q_network.is_parametric
                    else q_network(observation))
        min_q_value = q_values.min(1)[0]
        value = output.average(min_q_value - alpha * output.log_prob)
        return value

    def optimization_step(batch: Batch) -> None:

        alpha = log_alpha.exp().detach()

        # Update Q targets
        # NB. use old policy that hasn't been updated for current batch
        with torch.no_grad():
            next_action_dist = policy(batch['next_observation'])
            next_action_params = next_action_dist.reparameterize()
            next_v_value = calc_state_value(
                target_q_network, batch['next_observation'], next_action_params, alpha
            )
            next_v_value *= (~batch['terminal'].bool()).float()
            target_q_value = batch['reward'] + discount * next_v_value
            # replicate same target for all Q networks
            target_q_values = target_q_value.unsqueeze(1).expand(
                [len(target_q_value), target_q_network.num_nets])

        # Update policy weights (Eq. 10)
        action_dist = policy(batch['observation'])
        action_params = action_dist.reparameterize(measure=True)
        v_value = calc_state_value(
            q_network, batch['observation'], action_params, alpha
        )
        policy_losses = -v_value
        policy_loss = policy_losses.mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        action_entropy = action_params.entropy()
        action_logp = action_dist.log_prob(batch["action"])
        Measurements.update({
            "policy/entropy": action_entropy,
            "policy/temperature": alpha,
            "policy/ips": (action_logp - batch['log_propensity']).exp(),
            "losses/policy": policy_losses,
        })

        # Update Q-function parameters (Eq. 6)
        pred_q_values = q_network(batch['observation'], batch['action'])
        assert pred_q_values.shape == target_q_values.shape
        q_network_losses = F.mse_loss(pred_q_values, target_q_values, reduction="none")
        q_network_loss = q_network_losses.mean()
        q_network_optimizer.zero_grad()
        q_network_loss.backward()
        q_network_optimizer.step()
        Measurements.update({
            "q_network/reward": batch["reward"],
            "q_network/value": pred_q_values.flatten(),
            "q_network/target": target_q_value,
            "losses/q_network": q_network_losses.flatten(),
        })

        # Adjust temperature (Eq. 18)
        # NB. paper uses alpha (not log) but we follow the softlearning impln
        # see also https://github.com/rail-berkeley/softlearning/issues/37
        if optimize_alpha:
            alpha_losses = (
                log_alpha * (action_entropy - target_entropy).detach()
            )
            alpha_loss = alpha_losses.mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            Measurements.update({"losses/temperature": alpha_losses})

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
        if wandb and wandb.run:
            wandb.save(path)
        logging.info('saved model checkpoint to %s', path)

    # main loop
    for step in range(num_steps):
        # environment step
        transition = sampler.sample_transition(policy)
        replaybuf.append(transition)
        # optimization step
        if step % update_interval == 0:
            batch = replaybuf.sample_batch(batch_size)
            optimization_step(batch)
        if step > 0 and step % progress_interval == 0:
            if validation_env:
                metrics = validate(policy, q_network, validation_env, num_episodes=10)
                logging.info('step %d eval episode length %.0f return %f',
                             step,
                             metrics['episode_length'],
                             metrics['episode_return'])
            measurements.report(writer, step)
            writer.add_scalar("datagen/total_episodes", sampler.total_episodes, step)
            writer.add_scalar("datagen/total_steps", sampler.total_steps, step)
            if wandb and wandb.run:
                wandb.log({
                    "datagen/total_episodes": sampler.total_episodes,
                    "datagen/total_steps": sampler.total_steps,
                }, step=step)
        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(step)

    writer.close()


@torch.no_grad()
def validate(policy: GaussianPolicy, q_network: QNetwork, env: Env, *,
             num_episodes: int = 1) \
        -> Dict[str, List[float]]:
    """Validate policy on environment
    NB mutates env state!
    """
    total_return = 0.0
    total_steps = 0
    for _ in range(num_episodes):
        observations = []
        actions = []
        rewards = []
        observation = env.reset()
        done = False
        while not done:
            action, _ = policy.choose_action(observation, greedy=True)
            observations.append(observation)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_return += reward
            total_steps += 1
        observations = torch.tensor(np.stack(observations), dtype=torch.float32)
        actions = torch.tensor(np.stack(actions))
        rewards = torch.tensor(rewards)
        returns = torch.cumsum(rewards.flip(0), 0).flip(0)
        q_values = q_network(observations, actions)
        Measurements.update({
            "eval/episode_return": sum(rewards).item(),
            "eval/episode_length": len(rewards),
            "eval/q_value": q_values.flatten(),
            "eval/q_value_error": q_values[:, 0] - returns,
            "eval/q_value_r2_score": r2_score(returns, q_values[:, 0]),
        })
    return {
        "episode_return": total_return / num_episodes,
        "episode_length": total_steps / num_episodes,
    }


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
            action, _ = policy.choose_action(observation, greedy=True)
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


@torch.no_grad()
def r2_score(target: FloatTensor, input: FloatTensor) -> float:
    """Calculate R^2 score."""
    assert target.shape == input.shape
    return 1.0 - (
        (target - input).pow(2).sum()
        / (target - target.mean()).pow(2).sum()
    ).item()
