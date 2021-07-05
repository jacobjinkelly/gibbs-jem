"""
Utilities for models.
"""

from functools import partial

import torch

from utils import wrap_args, TOY_DSETS, CTS_TOY_DSETS
from .classifier import Classifier
from .ebm import EBM
from .jem import ProductOfJEMS
from .models import cnn, smooth_mlp_ebm


@wrap_args
def get_models(mode,
               num_steps,
               sigma,
               nums_attributes,
               eval_batch_size,
               data_size,
               dataset,
               device,
               init_dist,
               **_):
    if dataset in TOY_DSETS + CTS_TOY_DSETS:
        network = smooth_mlp_ebm  # large_mlp_ebm

    else:
        network = cnn

    network = partial(network, data_size=data_size)

    if mode == "ebm":
        logp_net = network(nout=1)
        return EBM(logp_net,
                   num_steps=num_steps,
                   sigma=sigma,
                   sample_batch_size=eval_batch_size,
                   data_size=data_size,
                   init_dist=init_dist,
                   device=device)
    elif mode == "poj":
        return ProductOfJEMS(logp_net=network,
                             nums_attributes=nums_attributes,
                             sample_batch_size=eval_batch_size,
                             num_steps=num_steps,
                             sigma=sigma,
                             data_size=data_size,
                             init_dist=init_dist,
                             device=device)
    elif mode == "sup":
        return Classifier(network, nums_attributes=nums_attributes, device=device)
    else:
        raise ValueError


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d


def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d


def approx_difference_function_multi_dim(x, model, axis_mask=None):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    if axis_mask is not None:
        x_embed = torch.zeros(axis_mask.shape, device=axis_mask.device)
        gx_embed = torch.zeros(axis_mask.shape, device=axis_mask.device)
        x_embed[axis_mask], gx_embed[axis_mask] = x, gx
        gx_cur_embed = (gx_embed * x_embed).sum(-1)[:, :, None]
        return (gx_embed - gx_cur_embed)[axis_mask]
    else:
        gx_cur = (gx * x).sum(-1)[:, :, None]
        return gx - gx_cur


def approx_difference_function_multi_dim_struct(x, shape, model, temp):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    diffs = []
    for prev_label_axis_dim, label_axis_dim in zip((0, ) + shape[:-1], shape):
        gx_ = gx[:, :, prev_label_axis_dim: (label_axis_dim + prev_label_axis_dim)]
        x_ = x[:, :, prev_label_axis_dim: (label_axis_dim + prev_label_axis_dim)]
        gx_cur_ = (gx_ * x_).sum(-1)[:, :, None]
        diffs.append((gx_ - gx_cur_) / temp)
    return diffs
