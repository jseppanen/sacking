
import tensorflow as tf
import torch


tf.set_random_seed(1337)

_seed = 0


def next_seed():
    global _seed
    _seed += 1
    return _seed


def glorot_uniform_(tensor: torch.Tensor) -> None:
    """Initialize tensor with TF glorot_uniform initializer."""
    import tensorflow as tf
    init = tf.glorot_uniform_initializer(next_seed())
    op = init(tensor.shape[::-1])
    with tf.compat.v1.Session() as sess:
        values = sess.run(op)
    tensor[:] = torch.from_numpy(values.T)
