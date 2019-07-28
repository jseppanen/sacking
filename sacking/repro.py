
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from _pytest.python_api import ApproxNumpy

SESSION = tf.compat.v1.Session()
tf.set_random_seed(1337)

_seed = 0


def next_seed():
    global _seed
    _seed += 1
    return _seed


def repro_glorot_uniform_(tensor: torch.Tensor) -> None:
    """Initialize tensor with TF glorot_uniform initializer."""
    init = tf.glorot_uniform_initializer(next_seed())
    op = init(tensor.shape[::-1])
    values = SESSION.run(op)
    tensor[:] = torch.from_numpy(values.T)


def repro_normal_sample(shape):
    """Sample batch from TFP MultivariateNormalDiag."""
    tf_dist = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(shape[1]),
        scale_diag=tf.ones(shape[1]))
    tf_latents = tf_dist.sample(shape[0], seed=1)
    latents = SESSION.run(tf_latents)
    return torch.from_numpy(latents)


ITERATION = 0


@torch.no_grad()
def repro_batch():
    """Load repro batch from disk."""
    global ITERATION
    slbatch = pickle.load(open(f'repro/batch{ITERATION}.pkl', 'rb'))
    slvalue = pickle.load(open(f'repro/values{ITERATION}.pkl', 'rb'))
    slweight = pickle.load(open(f'repro/weights{ITERATION}.pkl', 'rb'))
    ITERATION += 1

    batch = {
        'observation': torch.from_numpy(slbatch['observations']['observations']),
        'action': torch.from_numpy(slbatch['actions']),
        'reward': torch.from_numpy(slbatch['rewards']),
        'next_observation': torch.from_numpy(slbatch['next_observations']['observations']),
        'done': torch.from_numpy(slbatch['terminals']),
    }
    for k in ['reward', 'done']:
        batch[k] = batch[k].squeeze(1)
    batch['done'] = batch['done'].byte()
    return batch, slvalue, slweight


def compare_weights(net, weights):
    for j in range(len(weights) // 2):
        assert approx(net[2 * j].weight) == weights[2 * j].T
        assert approx(net[2 * j].bias) == weights[2 * j + 1].T


class approx(ApproxNumpy):
    """Approx comparison for 32-bit floats"""

    def __init__(self, expected, abs=1e-5, rel=1e-6):
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().numpy()
        assert isinstance(expected, np.ndarray)
        super().__init__(expected, abs=abs, rel=rel)

    def __eq__(self, actual):
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().numpy()
        assert isinstance(actual, (np.ndarray, np.float32))
        return super().__eq__(actual)
