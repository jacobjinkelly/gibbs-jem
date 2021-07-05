"""
Utilities for training EBMs.
"""
from functools import reduce

import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn

from models import utils


def short_run_mcmc(f, x_k, num_steps, sigma, step_size=1.0):
    x_k = x_k.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k.detach()


def cond_short_run_mcmc(f, x_k, x_cond, obs, num_steps, sigma, step_size=1.0):
    x_k = x_k * (1. - obs) + x_cond * obs
    x_k = x_k.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        # compute updates to unobserved variables
        x_k_update = step_size * f_prime + sigma * torch.randn_like(x_k)
        # update only unobserved variables
        x_k.data += x_k_update * (1. - obs) + x_cond * obs

    return x_k


def partial_gibbs_short_run_mcmc(f, *args, **kwargs):
    def sample_energy(x):
        logits = f(x)
        return torch.distributions.Categorical(logits=logits).sample()

    def categorical_energy(x):
        logp_x_y = f.logp_x_y(x, ys=sample_energy(x))
        return logp_x_y

    x_k = short_run_mcmc(categorical_energy, *args, **kwargs)
    a_k = sample_energy(x_k)
    return x_k, a_k


def get_one_hot(labels, num_labels):
    return torch.scatter(torch.zeros(labels.shape[0], num_labels).to(labels.device), 1, labels[:, None], 1)


def one_hot_missing(y, num_labels):
    missing_label_mask = y == -1
    y[missing_label_mask] = 0  # set missing labels to valid value for one hot function
    y_onehot = nn.functional.one_hot(y, num_labels).float()
    y[missing_label_mask] = -1  # set missing labels back to before (inplace modification!)
    y_onehot[missing_label_mask] = 0  # set onehots for missing labels to all 0

    return y_onehot


def get_onehots(label_shape, unif_label_shape, y):
    if unif_label_shape is not None:
        return one_hot_missing(y, unif_label_shape)
    else:
        y = y.to(torch.int64)
        one_hots = torch.empty(y.shape[0], sum(label_shape), device=y.device)

        def add_one_hot(val, el):
            label_axis, label_axis_dim = el
            next_val = val + label_axis_dim
            one_hots[:, val:next_val] = one_hot_missing(y[:, label_axis], label_axis_dim)
            return next_val

        reduce(add_one_hot, enumerate(label_shape), 0)
        return one_hots


def convert_struct_onehot(label_shape, unif_label_shape, y):
    """
    Expect x.shape = (batch_size, label_dim)
    Return x.shape = (batch_size, len(label_shape))
    """
    if unif_label_shape is not None:
        y = y.view(y.shape[0], len(label_shape), unif_label_shape)
        return y.max(-1).indices
    else:
        labels = torch.empty(y.shape[0], len(label_shape), device=y.device)

        def add_label(val, el):
            label_axis, label_axis_dim = el
            next_val = val + label_axis_dim
            y_onehot = y[:, val:next_val]
            labels[:, label_axis] = y_onehot.max(-1).indices
            return next_val

        reduce(add_label, enumerate(label_shape), 0)
        return labels


def get_label_axes(label_shape, dim=0):
    return torch.cumsum(label_shape, dim=dim)


def get_diff_label_axes(label_axes, zeros_shape=(1,), dim=0):
    return torch.cat([torch.zeros(zeros_shape, device=label_axes.device, dtype=torch.int64), label_axes], dim=dim)


def get_struct_mask(label_axes):
    return torch.arange(label_axes[-1], device=label_axes.device)[None]


def get_onehot_struct_mask(diff_label_axes, struct_mask, struct_block):
    """
    Get full one-hot mask from block numbers.
    """
    # lower and upper bounds for indices
    l, u = diff_label_axes[struct_block], diff_label_axes[struct_block + 1]
    struct_mask = torch.logical_and(l <= struct_mask, struct_mask < u)
    return struct_mask[:, None]


def get_struct_blocks(label_axes, changes_r):
    inds = changes_r.max(-1).indices  # get inds to be changed
    return torch.searchsorted(label_axes, inds, right=True)  # get block for each ind


def per_example_mask(axis_mask):
    return len(axis_mask.squeeze().shape) > 1


class DiffSampler(nn.Module):
    def __init__(self, n_steps=10, approx=False, temp=1., step_size=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.approx = approx
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        # initialize these to a null value
        self._ar = -1
        self._lr = -1
        self._lf = -1
        self._la = -1
        self._mt = -1
        self._pt = -1
        self._hops = -1

    def step(self, x, model):

        assert len(x.shape) == 2

        x_cur = x

        a_s = []
        lr = []
        lf = []
        la_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)

            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes = cd_forward.sample()

            lp_forward = cd_forward.log_prob(changes)

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            a_s.append(a.mean().item())
            lr.append(lp_reverse.exp().mean().item())
            lf.append(lp_forward.exp().mean().item())
            la_s.append(la.exp().mean().item())
            m_terms.append(m_term.exp().mean().item())
            prop_terms.append((lp_reverse - lp_forward).exp().mean().item())

        self._ar = np.mean(a_s)
        self._lr = np.mean(lr)
        self._lf = np.mean(lf)
        self._la = np.mean(la_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).mean().item()

        return x_cur


class DiffSamplerMultiDim(nn.Module):
    def __init__(self, n_steps=10, approx=False, temp=1., struct=False, label_shape=None,
                 shift_logit=True, other_reverse_changes=False, device=torch.device('cpu')):
        super().__init__()
        self.shift_logit = shift_logit
        self.other_reverse_changes = other_reverse_changes
        self.n_steps = n_steps
        self.approx = approx
        self.temp = temp
        self.struct = struct
        self.label_shape = torch.tensor(label_shape, device=device)
        self.label_axes = get_label_axes(self.label_shape)
        self.diff_label_axes = get_diff_label_axes(self.label_axes)
        self.struct_mask = get_struct_mask(self.label_axes)
        if approx:
            self.diff_fn = lambda x, m, **kwargs: utils.approx_difference_function_multi_dim(x, m, **kwargs) / self.temp
        else:
            assert False, f"no, no, no"

        # initialize these to a null value
        self._ar = -1
        self._lr = -1
        self._lf = -1
        self._la = -1
        self._mt = -1
        self._pt = -1
        self._hops = -1

    def step(self, x, model, axis_mask=None):

        if self.struct and axis_mask is not None and per_example_mask(axis_mask):
            pass
        else:
            assert len(x.shape) == 3

        is_per_example_mask = False
        per_example_x_mask = torch.tensor(0.)  # just for the linter
        per_example_x_mask_shape = None

        struct = self.struct

        label_axes = self.label_axes
        diff_label_axes = self.diff_label_axes
        struct_mask = self.struct_mask

        if axis_mask is not None:
            is_per_example_mask = per_example_mask(axis_mask)
            if is_per_example_mask:
                # x is the result of applying axis_mask, so it's a view
                assert x.shape[0] != axis_mask.shape[0]
                if struct:
                    per_example_x_mask_shape = axis_mask.shape
                else:
                    per_example_x_mask_shape = (axis_mask.shape + x.shape[-1:])
                per_example_x_mask = -torch.ones(per_example_x_mask_shape, device=x.device) * float("inf")

                if struct:
                    block_axis_mask = axis_mask[..., self.diff_label_axes[:-1]]
                    label_axes = get_label_axes(self.label_shape.repeat(block_axis_mask.shape[0], 1)[:, None] *
                                                block_axis_mask, dim=-1)

            elif struct:
                axis_mask = axis_mask[:, self.diff_label_axes[:-1]].squeeze()
                if axis_mask.sum() == 1:
                    # conditoning all but one variable, so we're doing normal sampling without structured labels
                    struct = False
                else:
                    label_axes = get_label_axes(self.label_shape[axis_mask])
                    diff_label_axes = get_diff_label_axes(label_axes)
                    struct_mask = get_struct_mask(label_axes)

        def cd_changes(logits):
            per_example_x_mask[axis_mask] = logits.squeeze()
            logits = per_example_x_mask

            # sampling will break if all options are impossible
            # this case occurs when all the b are observed and nothing needs to be resampled
            logits[(~axis_mask).all(-1)] = -1e9

            cd = dists.OneHotCategorical(logits=logits.view(axis_mask.size(0), -1))

            changes_ = cd.sample()

            # reset the container
            per_example_x_mask[axis_mask] = -float("inf")
            per_example_x_mask[(~axis_mask).all(-1)] = -float("inf")

            return cd, changes_

        x_cur = x

        a_s = []
        lr = []
        lf = []
        la_s = []
        m_terms = []
        prop_terms = []

        x_mask = None

        for i in range(self.n_steps):
            if is_per_example_mask and struct:
                forward_delta = self.diff_fn(x_cur, model, axis_mask=axis_mask)
            else:
                forward_delta = self.diff_fn(x_cur, model)
            if self.shift_logit:
                # make sure we dont choose to stay where we are!
                forward_logits = forward_delta - 1e9 * x_cur
            else:
                forward_logits = forward_delta

            if is_per_example_mask:
                cd_forward, changes = cd_changes(forward_logits)
            else:
                cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
                changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            if is_per_example_mask:
                changes_r = changes.view(per_example_x_mask_shape)[axis_mask][:, None]
            else:
                changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out changed dim and add in the change
            if struct:
                if is_per_example_mask:
                    x_mask = torch.ones(axis_mask.shape, device=axis_mask.device)
                    struct_blocks = get_struct_blocks(self.label_axes, changes)[:, None]
                else:
                    x_mask = torch.ones_like(x_cur)
                    struct_blocks = get_struct_blocks(label_axes, changes_r)
                onehot_struct_mask = get_onehot_struct_mask(diff_label_axes, struct_mask, struct_blocks)
                # mask out the one-hot block
                x_mask[onehot_struct_mask] = 0
                if is_per_example_mask:
                    x_mask = x_mask[axis_mask]
                    changes_r = changes_r.squeeze()
                x_delta = x_cur.clone() * x_mask + changes_r
            else:
                if is_per_example_mask:
                    x_delta = x_cur.clone() * (1. - changed_ind[:, None]) + changes_r
                else:
                    x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            if is_per_example_mask and struct:
                reverse_delta = self.diff_fn(x_delta, model, axis_mask=axis_mask)
            else:
                reverse_delta = self.diff_fn(x_delta, model)
            if self.shift_logit:
                reverse_logits = reverse_delta - 1e9 * x_delta
            else:
                reverse_logits = reverse_delta

            if is_per_example_mask:
                cd_reverse, _ = cd_changes(reverse_logits)
            else:
                cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_cur.size(0), -1))

            if self.other_reverse_changes:
                assert not struct, "Can't use struct here!!"
                reverse_changes = changes_r
            else:
                if struct:
                    reverse_changes = x_cur * (1 - x_mask)
                    if is_per_example_mask:
                        # set container to 0
                        per_example_x_mask[per_example_x_mask == -float("inf")] = 0
                        # fill contain with reverse changes
                        per_example_x_mask[axis_mask] = reverse_changes.squeeze()
                        # point reverse changes to container
                        reverse_changes = per_example_x_mask
                else:
                    if is_per_example_mask:
                        reverse_changes = x_cur * changed_ind[:, None]
                        # set container to 0
                        per_example_x_mask[per_example_x_mask == -float("inf")] = 0
                        # fill contain with reverse changes
                        per_example_x_mask[axis_mask] = reverse_changes.squeeze()
                        # point reverse changes to container
                        reverse_changes = per_example_x_mask
                    else:
                        reverse_changes = x_cur * changed_ind[:, :, None]

            if is_per_example_mask:
                lp_reverse = cd_reverse.log_prob(reverse_changes.view(axis_mask.size(0), -1))
            else:
                lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            # computer acceptance probabilities
            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()

            # accept changes
            if is_per_example_mask:
                per_example_x_mask[per_example_x_mask != 0] = 0
                x_delta_mask, x_cur_mask = per_example_x_mask, per_example_x_mask.clone()
                x_delta_mask[axis_mask], x_cur_mask[axis_mask] = x_delta.squeeze(), x_cur.squeeze()
                x_cur = (x_delta_mask * a[:, None, None] + x_cur_mask * (1. - a[:, None, None]))[axis_mask]
            else:
                x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])

            a_s.append(a.mean().item())
            lr.append(lp_reverse.exp().mean().item())
            lf.append(lp_forward.exp().mean().item())
            la_s.append(la.exp().mean().item())
            m_terms.append(m_term.exp().mean().item())
            prop_terms.append((lp_reverse - lp_forward).exp().mean().item())

        self._ar = np.mean(a_s)
        self._lr = np.mean(lr)
        self._lf = np.mean(lf)
        self._la = np.mean(la_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()

        return x_cur
