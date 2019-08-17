
from typing import Dict, List, Optional

from torch import nn
from torch import FloatTensor
from torch.nn import functional as F


def conv3x3(in_chans: int, out_chans: int, stride: int = 1) -> nn.Module:
    m = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1,
                  stride=stride, bias=False)
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return m


def conv1x1(in_chans: int, out_chans: int, stride: int = 1) -> nn.Module:
    m = nn.Conv2d(in_chans, out_chans, kernel_size=1,
                  stride=stride, bias=False)
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return m


def batchnorm2d(chans: int) -> nn.Module:
    m = nn.BatchNorm2d(chans)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


def batchnorm1d(chans: int) -> nn.Module:
    m = nn.BatchNorm1d(chans)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


def relu(x: FloatTensor) -> FloatTensor:
    return F.relu(x, inplace=True)


class Resblock(nn.Module):
    def __init__(self, in_chans: int, chans: int, *,
                 stride: int = 1, dropout: float = 0.0,
                 batchnorm: bool = False):
        super().__init__()
        self.conv1 = conv3x3(in_chans, chans, stride=stride)
        self.norm1 = batchnorm2d(chans) if batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout2d(dropout)
        self.conv2 = conv3x3(chans, chans)
        self.norm2 = batchnorm2d(chans) if batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout2d(dropout)
        self.res_conv: Optional[nn.Module] = None
        self.res_norm: Optional[nn.Module] = None
        if chans != in_chans or stride != 1:
            self.res_conv = conv1x1(in_chans, chans, stride=stride)
            self.res_norm = batchnorm2d(chans) if batchnorm else nn.Identity()

    def forward(self, x: FloatTensor) -> FloatTensor:
        y = relu(self.dropout1(self.norm1(self.conv1(x))))
        y = self.dropout2(self.norm2(self.conv2(y)))
        # residual connection
        if self.res_conv:
            x = self.res_norm(self.res_conv(x))
        y += x
        y = relu(y)
        return y


class Resnet(nn.Module):
    def __init__(self, chans: int, layers: List[Dict[str, int]], *,
                 dropout: float = 0.0,
                 batchnorm: bool = False):
        """Residual network.
        """
        super().__init__()
        blocks = []
        for layer in layers:
            assert isinstance(layer, dict)
            for block_idx in range(layer['blocks']):
                if block_idx == 0:
                    # downsample in the first block
                    blk = Resblock(chans, layer['channels'],
                                   stride=layer['stride'],
                                   dropout=dropout,
                                   batchnorm=batchnorm)
                    chans = layer['channels']
                else:
                    blk = Resblock(chans, chans,
                                   dropout=dropout,
                                   batchnorm=batchnorm)
                blocks.append(blk)
        self.net = nn.Sequential(*blocks)
        self.channels = chans

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.net(x)
