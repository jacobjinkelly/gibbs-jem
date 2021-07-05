"""
Energy-based model class.
"""

from functools import partial

from torch import nn as nn

from models.mcmc import short_run_mcmc


class EBM(nn.Module):
    """
    Energy-based model.
    """
    def __init__(self, logp_net, sample_batch_size, data_size, init_dist, device, **kwargs):
        super().__init__()

        self.logp_net = logp_net
        self.sample_batch_size = sample_batch_size
        self.data_size = data_size
        self.device = device
        self.sample_kwargs = kwargs
        self.init_dist = init_dist

        self.fixed_noise = self.sample_noise(sample_batch_size, device)

    def sample_noise(self, batch_size, device):
        return self.init_dist.sample((batch_size,)).to(device)

    def forward(self, x):
        return self.logp_net(x)

    def _sample(self, sampler, batch_size, fixed_noise):
        if fixed_noise:
            x_k = self.fixed_noise
        else:
            if batch_size is None:
                batch_size = self.sample_batch_size
            x_k = self.sample_noise(batch_size, self.device)
        samples = sampler(x_k)
        return samples.detach()

    def sample(self, batch_size=None, fixed_noise=False):
        sampler = partial(short_run_mcmc, self.logp_net, **self.sample_kwargs)
        return self._sample(sampler, batch_size, fixed_noise)

    def get_metrics(self, x_train, *_):
        """
        Get metrics.
        """
        logp_real = self.forward(x_train).mean().item()

        x_fake = self.sample(fixed_noise=True)
        logp_fake = self.forward(x_fake).mean().item()

        return {
            "logp_real": logp_real,
            "logp_fake": logp_fake,
            "logp_diff": logp_real - logp_fake
        }
