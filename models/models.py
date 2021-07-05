"""
Define models.
"""

from numpy import prod

import torch.nn as nn


def cnn(data_size, num_channels, nout):
    """
    Basic fast EBM.
    """
    assert data_size[1] == data_size[2], "Image size should be square"
    img_ch, img_size, _ = data_size
    assert img_size in (32, 64, 128), f"Got {img_size}"

    penult_ch = 8 * num_channels

    channels = [num_channels]
    while True:
        if len(channels) == 3 and img_size == 32:
            break
        elif len(channels) == 4 and img_size == 64:
            break
        elif len(channels) == 4 and img_size == 128:
            channels.append(channels[-1])
            break
        else:
            channels.append(2 * channels[-1])

    channels.extend((penult_ch, nout))

    layers = []
    for layer_num, (in_ch, out_ch) in enumerate(zip([img_ch] + channels[:-1], channels)):
        if layer_num == len(channels) - 1:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=0))
        else:
            if layer_num == 0:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(.2))

    return nn.Sequential(*layers)


def small_mlp_ebm(data_size, nout=1, h_dim=100):
    """
    Small MLP EBM.
    """
    data_dim = prod(data_size)
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(data_dim, h_dim),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(h_dim, h_dim),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(h_dim, nout, bias=True)
    )


def large_mlp_ebm(data_size, nout=1):
    """
    Large MLP EBM.
    """
    data_dim = prod(data_size)
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(data_dim, 1000),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(1000, 500),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(500, 500),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(500, 250),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(250, 250),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(250, 250),
        nn.LeakyReLU(.2, inplace=True),
        nn.Linear(250, nout, bias=True)
    )


def smooth_mlp_ebm(data_size, nout=1):
    data_dim = prod(data_size)
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(data_dim, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, nout),
    )
