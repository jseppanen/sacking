
import os
import warnings

import gym

from sacking.environment import load_env
from sacking.policy import DiscretePolicy
from sacking.q_network import DiscreteQNetwork
from sacking.trainer import train


def main():
    if (os.environ.get('OMP_NUM_THREADS') != '1'
           or os.environ.get('MKL_NUM_THREADS') != '1'):
        warnings.warn('running without OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')

    env = load_env('gym/CartPole-v1')
    valid_env = load_env('gym/CartPole-v1')

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    obs_dim = env.observation_space.shape[0]

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert len(env.action_space.shape) == 0
    action_dim = env.action_space.n

    policy = DiscretePolicy(obs_dim, action_dim,
                            hidden_layers=[64, 64])
    q_network = DiscreteQNetwork(obs_dim, action_dim,
                                 hidden_layers=[64, 64],
                                 num_nets=2)

    train(
        policy, q_network, env,
        batch_size=128,
        learning_rate=0.003,
        num_steps=50000,
        replay_buffer_size=100000,
        rundir='output',
        validation_env=valid_env,
    )


if __name__ == '__main__':
    main()
