
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import os
import warnings

import gym
import numpy as np

from sacking.environment import load_env
from sacking.policy import GaussianPolicy, DiscretePolicy
from sacking.q_network import QNetwork, DiscreteQNetwork
from sacking.resnet import Resnet
from sacking.trainer import train, simulate
from sacking.typing import Checkpoint
from sacking.openai.atari_wrappers import make_atari, wrap_deepmind
from torch import nn
from torch.nn import functional as F


class Flattener(nn.Module):
    def forward(self, obs):
        # input: batch x 84 x 84 x 4
        # crop
        x = obs[:, 14:77, :, :]
        # increase contrast
        x = (x.float() - 87) / (255 - 87)
        # flatten: batch x 21168
        x = x.view(x.shape[0], -1)
        return x


def main():
    if (os.environ.get('OMP_NUM_THREADS') != '1'
           or os.environ.get('MKL_NUM_THREADS') != '1'):
        warnings.warn('running without OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')

    env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)
    valid_env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)

    #obs_dim = np.prod(env.observation_space.shape)
    obs_dim = 21168
    action_dim = env.action_space.n

    body = Flattener()
    policy = DiscretePolicy(obs_dim, action_dim,
                            hidden_layers=[64, 64])
    q_network = DiscreteQNetwork(obs_dim, action_dim,
                                 hidden_layers=[64, 64],
                                 num_nets=2)

    train(
        policy, q_network, body, env,
        batch_size=256,
        learning_rate=3.0e-4,
        num_steps=1000000,
        num_initial_exploration_episodes=10,
        replay_buffer_size=100000,
        rundir='output',
        validation_env=valid_env,
        progress_interval=1000,
    )


if __name__ == '__main__':
    main()
