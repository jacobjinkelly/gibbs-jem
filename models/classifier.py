"""
Classifier.
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Supervised classifier.
    """

    def __init__(self, logp_net, nums_attributes, device):
        super().__init__()

        self.logp_net = logp_net(nout=sum(nums_attributes))
        self.n_att = len(nums_attributes)
        self.device = device

    def _split_logits(self, logits):
        atts = logits.view(logits.size(0), self.n_att, -1)
        return atts

    def forward(self, x):
        return self._split_logits(self.logp_net(x).squeeze())

    def loss(self, x, attributes, obs=None):

        split_logits = self.forward(x)
        if obs is None:
            obs = torch.ones_like(attributes).float()

        ce = nn.CrossEntropyLoss(reduction='none')
        lls = -ce(split_logits.view(split_logits.size(0) * split_logits.size(1), -1), attributes.view(-1))
        lls = lls.view(split_logits.size(0), split_logits.size(1))
        attribute_loss = (obs * lls).sum(1)

        return attribute_loss

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
        """
        Get metrics. Take in *args for compatibility with other models.
        """
        self.logp_net.eval()

        acc = self.accuracy(x_train, a_train, c_train).item()
        obs_acc = self.accuracy(x_train, a_train).item()

        self.logp_net.train()

        return {
            "acc": acc,
            "obs_acc": obs_acc
        }
