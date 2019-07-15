
import logging

import click
import gym
import yaml

from .policy import GaussianPolicy, QNetwork
from .trainer import train
from .typing import Env
from .version import __version__


@click.command()
@click.option('--config', required=True, type=click.Path(exists=True))
@click.option('--rundir', default='runs', type=click.Path())
def main(config: str, rundir: str):
    """Train SAC policy for a Gym environment.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'sacking {__version__}')

    config = yaml.load(open(config))

    env = load_env(config['env'])

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    assert len(env.action_space.shape) == 1
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, action_dim,
                            hidden_layers=config['policy']['hidden_layers'])
    q_networks = [
        QNetwork(obs_dim, action_dim,
                 hidden_layers=config['q_network']['hidden_layers'])
        for i in range(config['q_network']['num_heads'])
    ]

    train(policy, q_networks, env,
          batch_size=config['batch_size'],
          learning_rate=config['learning_rate'],
          num_steps=config['num_steps'],
          num_initial_exploration_steps=config['num_initial_exploration_steps'],
          replay_buffer_size=config['replay_buffer_size'],
          target_network_update_weight=config['target_network_update_weight'],
          progress_interval=config['progress_interval'],
          checkpoint_interval=config['checkpoint_interval'],
          rundir=rundir
          )


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


if __name__ == '__main__':
    main()
