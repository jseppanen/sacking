
import os
import warnings

import gym
from sacking.environment import load_env
from sacking.policy import GaussianPolicy
from sacking.q_network import QNetwork
from sacking.trainer import train


def main():
    if (os.environ.get('OMP_NUM_THREADS') != '1'
           or os.environ.get('MKL_NUM_THREADS') != '1'):
        warnings.warn('running without OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')

    env = load_env('roboschool/RoboschoolWalker2d-v1')
    valid_env = load_env('roboschool/RoboschoolWalker2d-v1')

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    obs_dim = env.observation_space.shape[0]

    assert isinstance(env.action_space, gym.spaces.Box)
    assert len(env.action_space.shape) == 1
    action_dim = env.action_space.shape[0]

    q_network = QNetwork(obs_dim, action_dim,
                         hidden_layers=[256, 256],
                         num_nets=2)
    policy = GaussianPolicy(obs_dim, action_dim,
                            hidden_layers=[256, 256])

    train(
        policy, q_network, env,
        batch_size=256,
        learning_rate=3.0e-4,
        num_steps=3000000,
        replay_buffer_size=1000000,
        rundir='output',
        checkpoint_interval=50000,
        validation_env=valid_env,
    )


if __name__ == '__main__':
    main()
