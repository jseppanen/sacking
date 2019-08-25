
from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class FCNetwork(nn.Module):
    """Fully-connected (MLP) network."""

    def __init__(self, input_dim: int, output_dim: int, *,
                 hidden_layers: Sequence[int] = (64,),
                 activation: str = 'relu'):
        super().__init__()

        try:
            activation_cls = activations[activation]
        except KeyError:
            raise ValueError(f'unknown activation: {activation}')

        sizes: List[int] = [input_dim] + list(hidden_layers)
        layers: List[nn.Module] = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            layers.append(linear(s1, s2))
            layers.append(activation_cls())
        layers.append(linear(sizes[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


def linear(dim_in: int, dim_out: int) -> nn.Linear:
    fc = nn.Linear(dim_in, dim_out)
    # softlearning initialization
    init.xavier_uniform_(fc.weight.data)
    fc.bias.data.fill_(0.0)
    return fc


class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


activations = {
    'relu': nn.ReLU,
    'swish': Swish,
}
