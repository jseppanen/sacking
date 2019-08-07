
import logging

import click
import gym
import yaml

from .environment import load_env
from .policy import GaussianPolicy, DiscretePolicy
from .q_network import QNetwork, DiscreteQNetwork
from .trainer import train, simulate
from .typing import Checkpoint
from .version import __version__


@click.group()
@click.version_option(__version__)
def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


@main.command('train')
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--rundir', default='runs', type=click.Path())
def train_cmd(config: str, rundir: str):
    """Train SAC policy
    """
    logging.info(f'sacking {__version__}')

    config = yaml.safe_load(open(config))

    env = load_env(config['env'])
    valid_env = load_env(config['env'])

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    obs_dim = env.observation_space.shape[0]

    #assert isinstance(env.action_space, gym.spaces.Box)
    #assert len(env.action_space.shape) == 1
    #action_dim = env.action_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert len(env.action_space.shape) == 0
    action_dim = env.action_space.n
    policy = DiscretePolicy(obs_dim, action_dim,
                            hidden_layers=config['policy']['hidden_layers'])
    q_network = DiscreteQNetwork(obs_dim, action_dim,
                         hidden_layers=config['q_network']['hidden_layers'],
                         num_nets=config['q_network']['num_heads'])

    train(policy, q_network, env,
          batch_size=config['batch_size'],
          learning_rate=config['learning_rate'],
          num_steps=config['num_steps'],
          num_initial_exploration_episodes=config['num_initial_exploration_episodes'],
          replay_buffer_size=config['replay_buffer_size'],
          target_network_update_weight=config['target_network_update_weight'],
          progress_interval=config['progress_interval'],
          checkpoint_interval=config['checkpoint_interval'],
          rundir=rundir,
          validation_env=valid_env,
          )


@main.command('show')
@click.argument('env')
@click.argument('checkpoint', type=click.Path(exists=True))
def show_cmd(env: str, checkpoint: str):
    """Show policy simulation on screen
    """
    env = load_env(env)
    checkpoint = Checkpoint.load(checkpoint)
    policy = GaussianPolicy.from_checkpoint(checkpoint)

    while True:
        simulate(policy, env)


if __name__ == '__main__':
    main()
