
import logging

import click
import gym

from .policy import GaussianPolicy, QNetwork
from .trainer import train
from .version import __version__


@click.command()
@click.argument('env')
def main(env):
    """Train SAC policy for a Gym environment.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'sacking {__version__}')

    env = gym.make(env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    assert len(env.action_space.shape) == 1
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, action_dim)
    q_networks = [QNetwork(obs_dim, action_dim),
                  QNetwork(obs_dim, action_dim)]

    train(policy, q_networks, env)


if __name__ == '__main__':
    main()
