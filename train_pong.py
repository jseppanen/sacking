
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


def main():
    env = load_env('PongNoFrameskip-v4')
    valid_env = load_env('PongNoFrameskip-v4')

    observation_shape = env.observation_space.shape
    action_dim = env.action_space.n

    frame_dim = 48
    vision = Resnet(
            1,
            [{'channels': 16, 'blocks': 2, 'stride': 2},
             {'channels': 32, 'blocks': 2, 'stride': 2},
             {'channels': 32, 'blocks': 2, 'stride': 2}],
            dropout=0.1, batchnorm=True)
    feature_dim = 256
    xxx = nn.Linear(vision.channels, feature_dim)
            
    policy = DiscretePolicy(feature_dim, action_dim,
                            hidden_layers=256)
    q_network = DiscreteQNetwork(feature_dim, action_dim,
                         hidden_layers=256,
                         num_nets=2)

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


if __name__ == '__main__':
    main()
