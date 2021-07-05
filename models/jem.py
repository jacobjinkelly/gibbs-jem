"""
Defines JEM models.
"""
from functools import partial

import torch
import torch.nn as nn

from .mcmc import short_run_mcmc, partial_gibbs_short_run_mcmc


class ProductOfJEMS(nn.Module):
    """
    Product of JEM models for attributes.
    """
    def __init__(self, logp_net, nums_attributes, sample_batch_size, data_size, init_dist, device, **kwargs):
        """
        logp_net : Constructor for logp_net requiring nout.
        nums_attributes : list which contains the number of choices for each attribute, ex: [2, 2, 2, 2, 4, 2, ... ]
        """
        super().__init__()

        assert len(set(nums_attributes)) == 1, \
            f"Num. attributes must be uniform across attributes, got {set(nums_attributes)}"

        self.logp_net = logp_net(nout=sum(nums_attributes))
        self.n_att = len(nums_attributes)
        self.init_dist = init_dist

        self.sample_batch_size = sample_batch_size
        self.data_size = data_size
        self.device = device
        self.sample_kwargs = kwargs

        self.fixed_noise = self.sample_noise(sample_batch_size, device)

    def sample_noise(self, batch_size, device):
        return self.init_dist.sample((batch_size,)).to(device)

    def _split_logits(self, logits):
        atts = logits.view(logits.size(0), self.n_att, -1)
        return atts

    def forward(self, x):
        return self._split_logits(self.logp_net(x).squeeze())

    def _process_inputs(self, inputs, from_logits):
        if from_logits:
            return inputs
        else:
            return self.forward(inputs)

    def _sample(self, sampler, batch_size=None, fixed_noise=False, detach=True):
        if fixed_noise:
            x_k = self.fixed_noise
        else:
            if batch_size is None:
                batch_size = self.sample_batch_size
            x_k = self.sample_noise(batch_size, self.device)
        samples = sampler(x_k)
        if detach:
            samples = samples.detach()
        return samples

    def logp_x(self, inputs, from_logits=False):
        split_logits = self._process_inputs(inputs, from_logits)

        logsumexp_logits = split_logits.logsumexp(-1)
        return logsumexp_logits.sum(1)

    def logp_x_y(self, inputs, ys, obs=None, from_logits=False):
        split_logits = self._process_inputs(inputs, from_logits)

        gathered_logits = split_logits.gather(2, ys[:, :, None])[:, :, 0]

        if obs is None:
            return gathered_logits.sum(1)
        else:
            logsumexp_logits = split_logits.logsumexp(-1)
            return (obs * gathered_logits + (1 - obs) * logsumexp_logits).sum(1)

    def logp_y_given_x(self, inputs, ys, obs=None, from_logits=False):
        split_logits = self._process_inputs(inputs, from_logits)

        if obs is None:
            obs = torch.ones_like(ys).float()

        ce = nn.CrossEntropyLoss(reduction='none')
        lls = -ce(split_logits.view(split_logits.size(0) * split_logits.size(1), -1), ys.view(-1))
        lls = lls.view(split_logits.size(0), split_logits.size(1))
        return (obs * lls).sum(1)

    def sample_logp_x(self, *args, **kwargs):
        sampler = partial(short_run_mcmc, self.logp_x, **self.sample_kwargs)
        return self._sample(sampler, *args, **kwargs)

    def sample_logp_xy(self, *args, **kwargs):
        sampler = partial(partial_gibbs_short_run_mcmc, self.forward, **self.sample_kwargs)
        return self._sample(sampler, *args, **kwargs, detach=False)

    def sample_logp_x_given_y(self, a_train, obs_train, *args, **kwargs):
        sampler = partial(short_run_mcmc, partial(self.logp_x_y, ys=a_train, obs=obs_train), **self.sample_kwargs)
        return self._sample(sampler, *args, **kwargs)

    def accuracy(self, x, ys=None, obs=None):
        split_logits = self.forward(x)
        preds = split_logits.max(2).indices
        rights = (preds == ys).float()
        if obs is None:
            return rights.mean()
        else:
            n_right = (obs * rights).sum()
            return n_right.float() / obs.sum()

    def get_metrics(self, x_train, _, a_train, c_train):
        logits_real = self.forward(x_train)

        logp_x_real = self.logp_x(logits_real, from_logits=True).mean().item()

        x_fake = self.sample_logp_x(fixed_noise=True)
        logp_x_fake = self.logp_x(x_fake).mean().item()

        x_cond_fake = self.sample_logp_x_given_y(a_train, c_train)
        logp_x_given_y_real = self.logp_x_y(logits_real, ys=a_train, obs=c_train, from_logits=True).mean().item()
        logp_x_given_y_fake = self.logp_x_y(x_cond_fake, ys=a_train, obs=c_train).mean().item()

        acc = self.accuracy(x_train, a_train, c_train).item()
        obs_acc = self.accuracy(x_train, a_train).item()

        return {
            "logp_x_real": logp_x_real,
            "logp_x_fake": logp_x_fake,
            "logp_x_diff": logp_x_real - logp_x_fake,
            "logp_x_given_y_real": logp_x_given_y_real,
            "logp_x_given_y_fake": logp_x_given_y_fake,
            "logp_x_given_y_diff": logp_x_given_y_real - logp_x_given_y_fake,
            "acc": acc,
            "obs_acc": obs_acc
        }
