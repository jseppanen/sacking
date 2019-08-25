
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
    def __init__(self, feature_dim):
        super().__init__()
        # downsample to 42 x 42
        self.pool = nn.MaxPool2d(2, stride=2)
        # output resolution 11 x 11
        self.resnet = Resnet(
                4,
                [{'channels': 16, 'blocks': 2, 'stride': 2},
                {'channels': 32, 'blocks': 2, 'stride': 2}],
                dropout=0.1, batchnorm=True)
        self.fc = nn.Linear(self.resnet.channels * 11 * 11, feature_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float() / 256
        x = self.pool(x)
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(x)
        x = self.fc(x)
        return x


def main():
    env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)
    valid_env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), frame_stack=True)

    observation_shape = env.observation_space.shape
    action_dim = env.action_space.n
    feature_dim = 256

    body = Vision(feature_dim)
    policy = DiscretePolicy(feature_dim, action_dim,
                            hidden_layers=[256, 256])
    q_network = DiscreteQNetwork(feature_dim, action_dim,
                                 hidden_layers=[256, 256],
                                 num_nets=2)

    train(
        policy, q_network, body, env,
        batch_size=256,
        learning_rate=0.001,
        num_steps=1000000,
        num_initial_exploration_episodes=10,
        replay_buffer_size=100000,
        rundir='output',
        validation_env=valid_env,
        progress_interval=10,
    )


if __name__ == '__main__':
    main()
