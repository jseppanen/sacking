
import gym

from sacking.environment import load_env
from sacking.policy import DiscretePolicy
from sacking.q_network import DiscreteQNetwork
from sacking.trainer import train


def test_cartpole():
    """Test solving gym CartPole task (discrete actions)"""

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
        learning_rate=0.001,
        num_steps=100000,
        replay_buffer_size=100000,
        #target_entropy=-2.0,
        rundir='/tmp/test_cartpole',
        validation_env=valid_env,
    )
