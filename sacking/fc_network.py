
from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import init


class FCNetwork(nn.Module):
    """Fully-connected (MLP) network."""

    def __init__(self, input_dim: int, output_dim: int, *,
                 hidden_layers: Sequence[int] = (64,)):
        super().__init__()

        sizes: List[int] = [input_dim] + list(hidden_layers) + [output_dim]
        layers: List[nn.Module] = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            fc = nn.Linear(s1, s2)
            # softlearning initialization
            init.xavier_uniform_(fc.weight.data)
            fc.bias.data.fill_(0.0)
            layers.append(fc)
            # softlearning activation
            layers.append(nn.ReLU(inplace=True))
        # remove final ReLU
        del layers[-1]
        self.net = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)

