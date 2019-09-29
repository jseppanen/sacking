
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import gym

from sacking.environment import load_env
from sacking.policy import GaussianPolicy, DiscretePolicy
from sacking.q_network import QNetwork, DiscreteQNetwork
from sacking.resnet import Resnet
from sacking.trainer import train, simulate
from sacking.typing import Checkpoint
from sacking.openai.atari_wrappers import make_atari, wrap_deepmind
from torch import nn
from torch.nn import functional as F


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet(
                4,
                [{'channels': 4, 'blocks': 1, 'stride': 2},
                {'channels': 4, 'blocks': 1, 'stride': 2},
                {'channels': 8, 'blocks': 1, 'stride': 2},
                {'channels': 8, 'blocks': 1, 'stride': 2}],
                dropout=0.1, batchnorm=True)
        # input resolution 84 x 84
        # output resolution 6 x 6
        self.feature_dim = self.resnet.channels * 6 * 6

    def forward(self, x):
        # input: batch x 84 x 84 x 4
        x = x.permute(0, 3, 1, 2).float() / 255
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        return x


def main():
    env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)
    valid_env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)

    #observation_shape = env.observation_space.shape
    action_dim = env.action_space.n

    body = Vision()
    policy = DiscretePolicy(body.feature_dim, action_dim,
                            hidden_layers=[64, 64])
    q_network = DiscreteQNetwork(body.feature_dim, action_dim,
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
