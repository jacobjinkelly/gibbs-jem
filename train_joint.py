import argparse
import json
import os
import time
from bisect import bisect
from functools import partial, reduce
from itertools import product
from operator import itemgetter

import kornia
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import pickle
import numpy as np
import sklearn
import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils import spectral_norm, clip_grad_norm_
from torchvision.utils import save_image

from models.mcmc import short_run_mcmc, \
    DiffSampler, DiffSamplerMultiDim, get_one_hot, get_onehots, convert_struct_onehot, \
    get_onehot_struct_mask, get_label_axes, get_diff_label_axes, get_struct_mask, per_example_mask
from utils import get_logger, makedirs, save_ckpt
from utils.data import MNISTWithAttributes
from utils.downsample import Downsample
from utils.dsprites import get_dsprites_dset
from utils.process import utzappos_tensor_dset, utzappos_zero_shot_tensor_dset, split_utzappos, \
    celeba_tensor_dset, split_celeba, cub_tensor_dset, split_cub, get_data_gen, log_dset_label_info, \
    CELEBA_ZERO_SHOT_COMBOS, CELEBA_ZERO_SHOT_ATTRIBUTES, find_combos_in_tst_set, lbl_in_combos, index_by_combo_name
from utils.visualize_flow import plt_flow_density, plt_samples

# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.enabled = True

matplotlib.use("Agg")

IMG_DSETS = ["mnist", "fashionmnist", "celeba", "mwa", "dsprites", "utzappos", "cub"]
UTZAPPOS_TEST_LEN = CELEBA_TEST_LEN = CUB_TEST_LEN = 5000


def xor(x, y):
    return (x or y) and not (x and y)


def implies(x, y):
    return (not x) or y


def clamp_x(x, min_val=0, max_val=1):
    return torch.clamp(x, min=min_val, max=max_val)


def deq_x(x):
    return (255 * x + torch.rand_like(x)) / 256.


def cond_attributes_from_labels(labels_b, cond_cls):
    cond_cls_ind, cond_cls_val = cond_cls
    return labels_b[:, cond_cls_ind] == cond_cls_val


def f1_score_missing(y_true, y_pred, individual=False, micro=False):
    """
    Compute the f1 score, accounting for missing labels in `y_true`.
    """
    assert not (individual and micro)

    assert y_true.shape == y_pred.shape

    assert len(y_true.shape) == 2

    if micro:
        # average over attributes using "micro" method (treat each attribute as a separate sample)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

    missing_mask = y_true != -1

    tp = (y_true == y_pred) * (y_true == 1)
    err = (y_true != y_pred)

    # any missing values don't contribute to the sums
    tp = (tp * missing_mask).sum(0)
    err = (err * missing_mask).sum(0)

    f1 = (tp / (tp + .5 * err))

    if not individual:
        # if no individual scores desired, average over attributes ("macro" average)
        f1 = f1.mean()

    return f1


def get_calibration(y, p_mean, num_bins=20, axis=-1, multi_label=True, individual=False, micro=False,
                    debug=False):
    """Compute the calibration.
    Modified from: https://github.com/xwinxu/bayesian-sde/blob/main/brax/utils/utils.py
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
      y: class labels (binarized)
      p_mean: numpy array, size (batch_size, num_classes)
            containing the mean output predicted probabilities
      num_bins: number of bins
      axis: Axis of the labels
      multi_label: Multiple labels
      individual: Return ECE for individual labels
      micro: Return micro average of ECE scores across attributes
      debug: Return debug information
    Returns:
      cal: a dictionary
      {
        reliability_diag: realibility diagram
        ece: Expected Calibration Error
        nb_items: nb_items_bin/np.sum(nb_items_bin)
    }
    """
    assert implies(individual or micro, multi_label)
    assert not (individual and micro)

    if micro:
        y = y.view(-1)
        p_mean = p_mean.view(-1, p_mean.shape[2])

    # compute predicted class and its associated confidence (probability)
    conf, class_pred = p_mean.max(axis)

    assert y.shape[0] == p_mean.shape[0]
    assert len(p_mean.shape) == len(y.shape) + 1
    assert p_mean.shape[1] > 1
    if multi_label and not micro:
        assert len(y.shape) == 2
        assert y.shape[1] > 1
        assert p_mean.shape[2] > 1
    else:
        assert len(y.shape) == 1

    tau_tab = torch.linspace(0, 1, num_bins + 1, device=p_mean.device)

    conf = conf[None]
    for _ in range(len(y.shape)):
        tau_tab = tau_tab.unsqueeze(-1)

    sec = (conf < tau_tab[1:]) & (conf >= tau_tab[:-1])

    nb_items_bin = sec.sum(1)

    mean_conf = (conf * sec).sum(1) / nb_items_bin
    acc_tab = ((class_pred == y)[None] * sec).sum(1) / nb_items_bin

    _weights = nb_items_bin.float() / nb_items_bin.sum(0)
    ece = ((mean_conf - acc_tab).abs() * _weights).nansum(0)
    if not individual:
        # pytorch doesn't have a built in nanmean
        ece[ece.isnan()] = 0
        ece = ece.mean(0)

    cal = {
        'reliability_diag': (mean_conf, acc_tab),
        'ece': ece,
        '_weights': _weights,
    }
    if debug:
        cal.update({
            'conf': conf,
            'sec': sec,
            'tau_tab': tau_tab,
            'acc_tab': acc_tab,
            'p_mean': p_mean
        })
    return cal


def ap_score(y_true, y_pred, individual=False, micro=False):
    assert not (individual and micro)

    assert y_true.shape == y_pred.shape

    assert len(y_true.shape) == 2

    if (y_true == -1).any():
        raise NotImplementedError(f"Missing AP score not implemented.")

    if micro:
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        return average_precision_score(y_true=y_true.cpu().numpy(),
                                       y_score=y_pred.cpu().numpy())
    else:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        individual_ap = [average_precision_score(y_true=y_true[:, label_dim_],
                                                 y_score=y_pred[:, label_dim_])
                         for label_dim_ in range(y_true.shape[1])]
        individual_ap = torch.tensor(individual_ap)
        if not individual:
            return individual_ap.mean()
        return individual_ap


def auroc_score(y_true, y_pred, individual=False, micro=False):
    assert not (individual and micro)

    assert y_true.shape == y_pred.shape

    assert len(y_true.shape) == 2

    if (y_true == -1).any():
        raise NotImplementedError(f"Missing AP score not implemented.")

    if micro:
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        return roc_auc_score(y_true=y_true.cpu().numpy(), y_score=y_pred.cpu().numpy())
    else:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        individual_ap = [roc_auc_score(y_true=y_true[:, label_dim_], y_score=y_pred[:, label_dim_])
                         for label_dim_ in range(y_true.shape[1])]
        individual_ap = torch.tensor(individual_ap)
        if not individual:
            return individual_ap.mean()
        return individual_ap


class ReplayBuffer:
    def __init__(self, max_size, example_sample):
        """
        Parameters
        ----------
        max_size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        example_sample:
            Example samples (shape and device)
        """
        self._storage = example_sample
        self.buffer_len = 0
        self.max_size = max_size
        self._next_idx = 0

    def __len__(self):
        return self.buffer_len

    def _add_full_buffer(self, x, which_added=None):
        batch_size = x.shape[0]
        if which_added is None:
            if batch_size + self._next_idx < self.max_size:
                self._storage[self._next_idx:self._next_idx + batch_size] = x
                which_added = torch.arange(self._next_idx, self._next_idx + batch_size)
            else:
                split_idx = self.max_size - self._next_idx
                self._storage[self._next_idx:] = x[:split_idx]
                self._storage[:batch_size - split_idx] = x[split_idx:]
                which_added = torch.cat([torch.arange(batch_size - split_idx),
                                         torch.arange(self._next_idx, len(self._storage))], dim=0)
        else:
            self._storage[which_added] = x
        assert len(which_added) == batch_size
        return batch_size, which_added

    def add(self, x, which_added=None):
        num_added, which_added = self._add_full_buffer(x, which_added)

        batch_size = x.shape[0]

        self._next_idx = (self._next_idx + batch_size) % self.max_size

        self.buffer_len = min(self.max_size, self.buffer_len + batch_size)

        return num_added, which_added

    def sample(self, batch_size, inds=None):
        if inds is None:
            inds = torch.randint(0, len(self._storage), (batch_size,))
        return self._storage[inds], inds


class ReservoirBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n = 0

    def _add_full_buffer(self, x, which_added_add_inds_mask=None):
        batch_size = x.shape[0]

        if which_added_add_inds_mask is None:
            if self._next_idx >= len(self):
                num_added, which_added = super()._add_full_buffer(x, which_added_add_inds_mask)
                add_inds_mask = torch.ones(which_added.shape, device=which_added.device, dtype=torch.bool)
                which_added_add_inds_mask = (which_added, add_inds_mask)
            else:
                randint_bounds = self._n + 1 + torch.arange(batch_size)
                inds = (torch.rand((batch_size,)) * randint_bounds).floor().long()
                add_inds_mask = inds < len(self)
                which_added = inds[add_inds_mask]
                self._storage[which_added] = x[add_inds_mask]

                num_added = add_inds_mask.sum()
                which_added_add_inds_mask = (which_added, add_inds_mask)

            self._n += batch_size
        else:
            which_added, add_inds_mask = which_added_add_inds_mask
            self._storage[which_added] = x[add_inds_mask]
            num_added = len(which_added)

        return num_added, which_added_add_inds_mask


class GaussianBlur:

    def __init__(self, min_val=0.1, max_val=2.0, kernel_size=9):
        self.min_val = min_val
        self.max_val = max_val
        self.kernel_size = kernel_size

    def __call__(self, sample):
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max_val - self.min_val) * np.random.random_sample() + self.min_val
            sample = kornia.filters.GaussianBlur2d((self.kernel_size, self.kernel_size), (sigma, sigma))(sample)

        return sample


def get_color_distortion(s=1.0):
    color_jitter = kornia.augmentation.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.4 * s)
    rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = kornia.augmentation.RandomGrayscale(p=0.2)
    color_distort = torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.model = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)


def repeat_x(x):
    repeat_shape = (x.shape[0],) + tuple(1 for _ in range(len(x.shape) - 1))
    return x[0][None].repeat(*repeat_shape)


def get_data(args, dataset, batch_size, device, return_dset=False, zero_shot=False, return_trn_len=False,
             return_tst_dset=False):

    if return_dset and dataset not in ("utzappos", "celeba", "cub"):
        raise NotImplementedError(f"Returning dataset is not implemented for {dataset}.")

    if zero_shot and dataset != "utzappos":
        raise NotImplementedError(f"Zero-shot learning is not implemented for {dataset}.")

    if return_trn_len and dataset not in ["celeba", "utzappos"]:
        raise NotImplementedError(f"Returning dataset length not configured for {dataset}.")

    if not implies(return_tst_dset, dataset == "celeba" and args.dset_split_type == "zero_shot"):
        raise NotImplementedError(f"Retuning tst dataset not configured for {dataset} and {args.dset_split_type}.")

    assert implies(zero_shot, return_dset), f"Must be returning dataset labels when doing zero-shot."
    assert implies(return_dset, args.full_test), f"Must be using test set when returning dataset labels."

    if dataset == "mnist":
        if args.small_cnn:
            assert args.img_size == 32
            transforms = [
                torchvision.transforms.Resize(args.img_size),
                torchvision.transforms.ToTensor(),
                lambda x: (255 * x + torch.rand_like(x)) / 256.,
            ]
            if args.logit:
                logger("================= DOING LOGIT TRANSFORM BOIII =================")
                transforms += [
                    lambda x: x * (1 - 2 * 1e-6) + 1e-6,
                    lambda x: x.log() - (1. - x).log()
                ]
        else:
            transforms = [
                torchvision.transforms.ToTensor(),
                lambda x: x.view(-1),
                lambda x: (255 * x + torch.rand_like(x)) / 256.,
            ]
            if not args.cnn and args.logit:
                logger("================= DOING LOGIT TRANSFORM BOIII =================")
                transforms += [
                    lambda x: x * (1 - 2 * 1e-6) + 1e-6,
                    lambda x: x.log() - (1. - x).log()
                ]
        dset_train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                                transform=torchvision.transforms.Compose(transforms))

        dset_test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                               transform=torchvision.transforms.Compose(transforms))

        trn_batch = get_data_gen(dataset=dset_train, batch_size=batch_size, split="trn", device=device)
        tst_batch = get_data_gen(dataset=dset_test, batch_size=batch_size, split="tst", device=device)

        return trn_batch, tst_batch

    elif dataset == "dsprites":
        print("================= DOING LOGIT TRANSFORM BOIII =================")
        if args.cnn:
            transforms = []
        else:
            transforms = [lambda x: x.view(-1)]
        transforms += [
            lambda x: (7. * x + torch.rand_like(x)) / 8.,
            lambda x: x * (1 - 2 * 1e-6) + 1e-6,
            lambda x: x.log() - (1. - x).log()
        ]
        dset = get_dsprites_dset(args.root, torchvision.transforms.Compose(transforms), test=args.dsprites_test)
        if args.dsprites_test:
            dset_train, dset_test = dset
            trn_batch = get_data_gen(dataset=dset_train, batch_size=batch_size, split="trn", device=device)
            tst_batch = get_data_gen(dataset=dset_test, batch_size=batch_size, split="tst", device=device)
            return trn_batch, tst_batch
        else:
            batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)
            return batch, None

    elif dataset == "fashionmnist":
        dset_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           lambda x: x.view(-1),
                                                           lambda x: (255 * x + torch.rand_like(x)) / 256.,
                                                           lambda x: x * (1 - 2 * 1e-6) + 1e-6,
                                                           lambda x: x.log() - (1. - x).log()
                                                       ]))

        batch = get_data_gen(dataset=dset_train, batch_size=batch_size, split="trn", device=device)

        return batch

    elif dataset == "celeba":

        transforms = [
            lambda x: (255 * x + torch.rand_like(x)) / 256.,
            lambda x: x + args.img_sigma * torch.randn_like(x)
        ]

        if args.clamp_data:
            transforms.append(clamp_x)

        root = os.path.join(args.root, "celeba")
        dset, dset_label_info, cache_fn = celeba_tensor_dset(root=root,
                                                             img_size=args.img_size,
                                                             transform=torchvision.transforms.Compose(transforms),
                                                             attr_fn=os.path.join(root, "list_attr_celeba.txt"),
                                                             pkl_fn=os.path.join(root, "cache.pickle"),
                                                             cache_fn=os.path.join(root, "dset.pickle"),
                                                             drop_infreq=args.celeba_drop_infreq)

        log_dset_label_info(logger, dset_label_info)

        if args.full_test:
            # overwrite dset instead of naming it trn_dset, since we return a copy not a pointer?
            dset, tst_dset = split_celeba(dset, root=root, cache_fn=cache_fn,
                                          split_len=CELEBA_TEST_LEN,
                                          split_type=args.dset_split_type,
                                          balanced_split_ratio=args.dset_balanced_split_ratio,
                                          dset_label_info=dset_label_info)
            trn_batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)
            tst_batch = get_data_gen(dataset=tst_dset, batch_size=batch_size, split="tst", device=device)
            return_tpl = (trn_batch, tst_batch, dset_label_info, len(tst_dset))

            if return_tst_dset:
                return_tpl += (tst_dset,)

            if return_dset:
                # get all labels in the train set
                return_tpl += (dset[:][1].float().mean(0)[:, None].to(args.device),)

            if args.dset_split_type == "zero_shot":
                logger("=========== ZERO SHOT ATTRIBUTE COMBINATION SAMPLING ===========")
                tst_combos = find_combos_in_tst_set(dset[:][1], tst_dset[:][1], dset_label_info,
                                                    CELEBA_ZERO_SHOT_COMBOS)
                trn_combos = [combo for combo in CELEBA_ZERO_SHOT_COMBOS if combo not in tst_combos]
                assert len(trn_combos) + len(tst_combos) == len(CELEBA_ZERO_SHOT_COMBOS)
                logger(f"{len(tst_combos)} COMBOS HELD OUT IN TEST SET")
                for tst_combo in tst_combos:
                    logger(tst_combo)

                # find mutually exclusive labels in the test set
                tst_labels_combos = lbl_in_combos(tst_dset[:][1], dset_label_info, tst_combos)
                tst_labels_combos_filter = tst_labels_combos.sum(1) == 1
                tst_labels_combos_nums = tst_labels_combos.sum(0) > 0
                logger(f"Found {tst_labels_combos_filter.sum()} of {len(tst_dset)} "
                       f"in test set with mutually exclusive labels.")
                logger(f"Found {tst_labels_combos_nums.sum()} of {len(tst_combos)} "
                       f"mutually exclusive combos in the test set.")
                return_tpl += (tst_combos,)

                trn_labels_combos = lbl_in_combos(tst_dset[:][1], dset_label_info, trn_combos)
                trn_labels_combos_filter = trn_labels_combos.sum(1) == 1
                trn_labels_combos_nums = trn_labels_combos.sum(0) > 0
                logger(f"Found trn {trn_labels_combos_filter.sum()} of {len(tst_dset)} "
                       f"in test set with mutually exclusive labels.")
                logger(f"Found trn {trn_labels_combos_nums.sum()} of {len(trn_combos)} "
                       f"mutually exclusive combos in the test set.")
        else:
            trn_batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)
            return_tpl = (trn_batch, dset_label_info)

        if return_trn_len:
            return_tpl += (len(dset),)

        return return_tpl

    elif "utzappos" in dataset:

        transforms = [
            lambda x: (255 * x + torch.rand_like(x)) / 256.,
            lambda x: x + args.img_sigma * torch.randn_like(x)
        ]

        if args.clamp_data:
            transforms.append(clamp_x)

        if zero_shot:
            root = os.path.join(args.root, "utzappos", "zero-shot")
            _dset_kwargs = {
                "root": os.path.join(root, "images"),
                "attr_fn": root,
                "pkl_fn": os.path.join(root, "cache.pickle"),
                "cache_fn": os.path.join(root, "dset.pickle"),
                "transform": torchvision.transforms.Compose(transforms)
            }
        else:
            root = os.path.join(args.root, "utzappos", "ut-zap50k-images-square")
            _dset_kwargs = {
                "root": root,
                "attr_fn": os.path.join(root, "meta-data.csv"),
                "pkl_fn": os.path.join(root, "cache.pickle"),
                "cache_fn": os.path.join(root, "dset.pickle"),
                "transform": torchvision.transforms.Compose(transforms)
            }
        if zero_shot:
            _dset_kwargs.update({
                "img_size": args.img_size
            })
        elif dataset == "utzappos":
            _dset_kwargs.update({
                "observed": True,
                "binarized": True,
                "drop_infreq": args.utzappos_drop_infreq,
                "img_size": args.img_size
            })
        else:
            assert dataset == "utzappos_old", f"Unrecognized dataset {dataset}"
            assert False

        if zero_shot:
            trn_dset, dset_label_info, cache_fn = utzappos_zero_shot_tensor_dset(split="trn", **_dset_kwargs)
            val_dset, *_ = utzappos_zero_shot_tensor_dset(split="val", **_dset_kwargs)
            tst_dset, *_ = utzappos_zero_shot_tensor_dset(split="tst", **_dset_kwargs)

            dset_lengths = {'trn': len(trn_dset), 'val': len(val_dset), 'tst': len(tst_dset)}

            log_dset_label_info(logger, dset_label_info)

            trn_batch = get_data_gen(dataset=trn_dset, batch_size=batch_size, split="trn", device=device, zs_mode=True)
            val_batch = get_data_gen(dataset=val_dset, batch_size=batch_size, split="tst", device=device, zs_mode=True)
            tst_batch = get_data_gen(dataset=tst_dset, batch_size=batch_size, split="tst", device=device, zs_mode=True)

            return_tpl = (trn_batch, val_batch, tst_batch, dset_label_info, dset_lengths)

            if return_dset:
                # get all labels in the train set
                trn_labels = trn_dset[:][1]  # trn_dset is a CustomTensorDataset, we need special indexing
                return_tpl += (trn_labels.float().mean(0)[:, None].to(args.device),)

            if return_trn_len:
                return_tpl += (len(trn_dset),)
        else:
            dset, dset_label_info, cache_fn = utzappos_tensor_dset(**_dset_kwargs)

            log_dset_label_info(logger, dset_label_info)

            # find_duplicates_in_dsets(dset, dset, itself=True)

            if args.full_test:
                trn_dset, tst_dset = split_utzappos(dset, root=root, cache_fn=cache_fn,
                                                    split_len=UTZAPPOS_TEST_LEN,
                                                    split_type=args.dset_split_type,
                                                    balanced_split_ratio=args.dset_balanced_split_ratio)
                trn_batch = get_data_gen(dataset=trn_dset, batch_size=batch_size, split="trn", device=device)
                tst_batch = get_data_gen(dataset=tst_dset, batch_size=batch_size, split="tst", device=device)
                return_tpl = (trn_batch, tst_batch, dset_label_info, len(tst_dset))
                if return_dset:
                    # get all labels in the train set
                    trn_labels = trn_dset[:][1]  # trn_dset is a CustomTensorDataset, we need special indexing
                    return_tpl += (trn_labels.float().mean(0)[:, None].to(args.device),)

                # find_duplicates_in_dsets(trn_dset, tst_dset)

                if return_trn_len:
                    return_tpl += (len(trn_dset),)

            else:
                trn_batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)
                return_tpl = (trn_batch, dset_label_info)

                if return_trn_len:
                    return_tpl += (len(dset),)

        return return_tpl

    elif dataset == "cub":

        transforms = [
            lambda x: (255 * x + torch.rand_like(x)) / 256.,
            lambda x: x + args.img_sigma * torch.randn_like(x)
        ]

        if args.clamp_data:
            transforms.append(clamp_x)

        root = os.path.join(args.root, "CUB")

        dset, dset_label_info, cache_fn = cub_tensor_dset(root=root,
                                                          img_size=args.img_size,
                                                          transform=torchvision.transforms.Compose(transforms),
                                                          drop_infreq=args.cub_drop_infreq,
                                                          attr_fn=os.path.join(root, "CUB_200_2011", "attributes",
                                                                               "image_attribute_labels.txt"),
                                                          attr_name_fn=os.path.join(root, "attributes.txt"),
                                                          img_id_fn=os.path.join(root, "CUB_200_2011",
                                                                                 "images.txt"),
                                                          bb_fn=os.path.join(root, "CUB_200_2011",
                                                                             "bounding_boxes.txt"),
                                                          pkl_fn=os.path.join(root, "cache.pickle"),
                                                          cache_fn=os.path.join(root, "dset.pickle"))

        log_dset_label_info(logger, dset_label_info)

        if args.full_test:
            trn_dset, tst_dset = split_cub(dset, root=root, cache_fn=cache_fn,
                                           split_len=CUB_TEST_LEN,
                                           split_type=args.dset_split_type,
                                           balanced_split_ratio=args.dset_balanced_split_ratio)
            trn_batch = get_data_gen(dataset=trn_dset, batch_size=batch_size, split="trn", device=device)
            tst_batch = get_data_gen(dataset=tst_dset, batch_size=batch_size, split="tst", device=device)
            return_tpl = (trn_batch, tst_batch, dset_label_info)

            if return_dset:
                # get all labels in the train set
                trn_labels = trn_dset[:][1]  # trn_dset is a CustomTensorDataset, we need special indexing
                return_tpl += (trn_labels.float().mean(0)[:, None].to(args.device),)
        else:
            trn_batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)
            return_tpl = (trn_batch, dset_label_info)

        return return_tpl

    elif dataset == "mwa":
        transforms = [
            torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
            lambda x: x + args.img_sigma * torch.randn_like(x)
        ]

        dset = MNISTWithAttributes(root="data", img_size=args.img_size,
                                   transform=torchvision.transforms.Compose(transforms))

        batch = get_data_gen(dataset=dset, batch_size=batch_size, split="trn", device=device)

        return batch

    elif dataset == "moons":
        data, labels = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
    elif dataset == "swissroll":
        data, labels = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
    elif dataset in ["rings", "rings_struct"]:
        rng = np.random.RandomState()
        obs = batch_size
        batch_size *= 20
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2
        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        Y = np.array([0] * n_samples1 + [1] * n_samples2 + [2] * n_samples3 + [3] * n_samples4)

        # Add noise
        X += rng.normal(scale=0.08, size=X.shape)
        inds = np.random.choice(list(range(batch_size)), obs)

        data = X[inds].astype("float32")
        labels = Y[inds]

        if dataset == "rings_struct":
            labels_1 = (labels < 2).astype(int)
            labels_2 = (labels % 2)
            labels = np.hstack((labels_1[:, None], labels_2[:, None]))
    elif dataset == "circles":
        data, labels = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)
        data = data.astype("float32")
        data *= 3
    elif dataset == "checkerboard":
        # uniform on (-2, 2)
        x1 = np.random.rand(batch_size) * 4 - 2

        # uniform on (0, 1) U (-2, -1)
        row = np.random.randint(0, 2, batch_size)
        x2_ = np.random.rand(batch_size) - row * 2
        # add {-2, -1, 0, 1} -> {0, 1} -> {(-2, -1), (-1, 0), (0, 1), (1, 2)}
        x2 = x2_ + (np.floor(x1) % 2)

        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        col = (np.floor(x1) + 2).astype(int)
        labels = col * 2 + row
    elif "8gaussians" in dataset:
        centers = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = 4. * np.array(centers)
        labels = np.random.randint(8, size=batch_size)
        points = np.random.randn(batch_size, 2) * 0.5
        data = centers[labels] + points
        data /= 1.414
        if "struct" in dataset:
            labels_1 = (labels < 4).astype(int)
            labels_2 = labels % 4
            labels = np.hstack((labels_1[:, None], labels_2[:, None]))
        elif "multi" in dataset:
            labels_1 = (labels < 4).astype(int)
            labels_2 = labels % 2
            labels_3 = (labels // 2) % 2
            labels = np.hstack((labels_1[:, None], labels_2[:, None], labels_3[:, None]))
        elif "hierarch" in dataset:
            labels_1 = (labels < 4).astype(int)
            labels_2 = np.isin(labels, [0, 1, 4, 7]).astype(int)
            labels_3 = np.isin(labels, [0, 2, 4, 5]).astype(int)
            labels = np.hstack((labels_1[:, None], labels_2[:, None], labels_3[:, None]))
        else:
            raise ValueError(f"Unrecognized dataset {dataset}")
    else:
        assert False, f"Unknown dataset {dataset}"

    if "missing" in dataset:
        mask_ind = np.array(list(product(*(range(2) for _ in range(labels.shape[1]))))).astype(bool)
        assert mask_ind[-1].all()  # corresponds to removing all labels
        num_label_masks = 2 ** labels.shape[1]
        if "unsup" not in dataset:
            num_label_masks -= 1  # don't choose the option to remove all labels
        labels_to_mask = np.random.choice(num_label_masks, labels.shape[0])
        labels_mask = mask_ind[labels_to_mask]
        labels[labels_mask] = -1

    return torch.tensor(data).to(device).float(), torch.tensor(labels).to(device)


def get_data_batch(args, dataset, batch_size, device, return_dset=False, zero_shot=False, return_trn_len=False,
                   return_tst_dset=False):
    if dataset in IMG_DSETS:
        return get_data(args, dataset, batch_size, device, return_dset, zero_shot, return_trn_len, return_tst_dset)
    else:
        return lambda: get_data(args, dataset, batch_size, device, zero_shot)


class SpectralLinear(nn.Module):

    def __init__(self, nin, nout, init_scale=1):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(nin, nout))
        self.log_scale = nn.Parameter(torch.zeros(1,) + np.log(init_scale), requires_grad=True)

    @property
    def scale(self):
        return self.log_scale.exp()

    def forward(self, x):
        return self.scale * self.linear(x)


def smooth_mlp_ebm_big(nin, nout=1):
    return nn.Sequential(
        nn.Linear(nin, 1000),
        nn.ELU(),
        nn.Linear(1000, 1000),
        nn.ELU(),
        nn.Linear(1000, 500),
        nn.ELU(),
        nn.Linear(500, nout),
    )


def smooth_mlp_ebm_bigger(act, nin, nout=1, spectral=False):
    """
    Large MLP EBM.
    """
    if act == "elu":
        assert not spectral
        return nn.Sequential(
            nn.Linear(nin, 1000),
            nn.ELU(),
            nn.Linear(1000, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 250),
            nn.ELU(),
            nn.Linear(250, 250),
            nn.ELU(),
            nn.Linear(250, 250),
            nn.ELU(),
            nn.Linear(250, nout)
        )
    elif act == "lrelu":
        assert not spectral
        return nn.Sequential(
            nn.Linear(nin, 1000),
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
            nn.Linear(250, nout)
        )
    elif act == "swish":
        if spectral:
            return nn.Sequential(
                spectral_norm(nn.Linear(nin, 1000)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(1000, 500)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(500, 500)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(500, 250)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(250, 250)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(250, 250)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(250, nout))
            )
        else:
            return nn.Sequential(
                nn.Linear(nin, 1000),
                nn.SiLU(inplace=True),
                nn.Linear(1000, 500),
                nn.SiLU(inplace=True),
                nn.Linear(500, 500),
                nn.SiLU(inplace=True),
                nn.Linear(500, 250),
                nn.SiLU(inplace=True),
                nn.Linear(250, 250),
                nn.SiLU(inplace=True),
                nn.Linear(250, 250),
                nn.SiLU(inplace=True),
                nn.Linear(250, nout)
            )
    else:
        assert f"act {act} not known"


def smooth_mlp_ebm(spectral, nin, nout=1, init_scale=1):
    if spectral:
        net = nn.Sequential(
            SpectralLinear(nin, 256, init_scale),
            nn.ELU(),
            SpectralLinear(256, 256, init_scale),
            nn.ELU(),
            SpectralLinear(256, 256, init_scale),
            nn.ELU(),
            SpectralLinear(256, nout, init_scale),
        )
    else:
        net = nn.Sequential(
            nn.Linear(nin, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, nout),
        )
    return net


def small_mlp_ebm(nin, nout=1, nhidden=256, spectral=False):
    if spectral:
        net = nn.Sequential(
            spectral_norm(nn.Linear(nin, nhidden)),
            nn.SiLU(inplace=True),
            spectral_norm(nn.Linear(nhidden, nhidden)),
            nn.SiLU(inplace=True),
            spectral_norm(nn.Linear(nhidden, nout)),
        )
    else:
        net = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.SiLU(inplace=True),
            nn.Linear(nhidden, nhidden),
            nn.SiLU(inplace=True),
            nn.Linear(nhidden, nout),
        )
    return net


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class CondResBlock(nn.Module):
    def __init__(self,
                 args,
                 downsample=True,
                 rescale=True,
                 filters=64,
                 latent_dim=64,
                 im_size=64,
                 classes=512,
                 norm=True,
                 spec_norm=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters)

        if not norm:
            self.bn1 = None

        self.args = args

        if spec_norm:
            self.conv1 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv1 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        if not norm:
            self.bn2 = None

        if spec_norm:
            self.conv2 = spectral_norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        else:
            self.conv2 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)

        # Upscale to an mask of image
        self.latent_map = nn.Linear(classes, 2*filters)
        self.latent_map_2 = nn.Linear(classes, 2*filters)

        self.relu = torch.nn.ReLU(inplace=True)
        self.act = nn.SiLU(inplace=True)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=2*filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, y):

        if y is not None:
            latent_map = self.latent_map(y).view(-1, 2*self.filters, 1, 1)

            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]
        else:
            gain = bias = None  # appeasing the linter

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)

        if y is not None:
            latent_map = self.latent_map_2(y).view(-1, 2*self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)

        x_out = x

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (W * H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class MNISTModel(nn.Module):
    def __init__(self, args):
        super(MNISTModel, self).__init__()
        self.act = nn.SiLU(inplace=True)

        self.args = args
        self.n_f = args.n_f
        self.init_main_model()
        self.init_label_map()

        self.cond = False

    def init_main_model(self):
        args = self.args
        filter_dim = self.n_f
        im_size = 28
        self.conv1 = nn.Conv2d(1, filter_dim, kernel_size=3, stride=1, padding=1)
        self.res1 = CondResBlock(args, filters=filter_dim, latent_dim=1, im_size=im_size)
        self.res2 = CondResBlock(args, filters=2*filter_dim, latent_dim=1, im_size=im_size)

        self.res3 = CondResBlock(args, filters=4*filter_dim, latent_dim=1, im_size=im_size)
        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_label_map(self):
        self.map_fc1 = nn.Linear(10, 256)
        self.map_fc2 = nn.Linear(256, 256)

    def main_model(self, x, latent):
        x = x.view(-1, 1, 28, 28)
        x = self.act(self.conv1(x))
        x = self.res1(x, latent)
        x = self.res2(x, latent)
        x = self.res3(x, latent)
        x = self.act(x)
        x = x.mean(dim=2).mean(dim=2)
        energy = self.energy_map(x)

        return energy

    def label_map(self, latent):
        x = self.act(self.map_fc1(latent))
        x = self.map_fc2(x)

        return x

    def forward(self, x, latent=None):
        x = x.view(x.size(0), -1)

        if self.cond:
            latent = self.label_map(latent)

        energy = self.main_model(x, latent)

        return energy


class CNNCond(nn.Module):
    def __init__(self, n_c=3, n_f=32, cnn_out_dim=512, label_dim=0, uncond=False):
        super(CNNCond, self).__init__()

        self.label_dim = label_dim
        self.cnn_out_dim = cnn_out_dim
        self.uncond = uncond

        self.cnn = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(.2),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(.2),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(.2),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(.2),
            nn.Conv2d(n_f * 8, n_f, 1, 1, 0),
            nn.Flatten()
        )

        self.mlp = smooth_mlp_ebm(spectral=False, nin=label_dim + cnn_out_dim, nout=1)

    def forward(self, x, label=None):
        x = self.cnn(x)
        if label is None:
            assert self.uncond
            return self.mlp(x)
        else:
            label = label.flatten(start_dim=1)
            return self.mlp(torch.cat([x, label], dim=1))


class CelebAModel(nn.Module):
    def __init__(self, args, debug=False):
        super(CelebAModel, self).__init__()
        self.act = nn.SiLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cond = False

        self.args = args
        self.init_main_model()

        if args.multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)
        self.heir_weight = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))
        self.debug = debug

    def init_main_model(self):
        args = self.args
        filter_dim = args.n_f
        latent_dim = args.n_f
        im_size = args.img_size

        self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1)

        self.res_1a = CondResBlock(args,
                                   filters=filter_dim // 2,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   downsample=True,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)
        self.res_1b = CondResBlock(args,
                                   filters=filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   rescale=False,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)

        self.res_2a = CondResBlock(args,
                                   filters=filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   downsample=True,
                                   rescale=False,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)
        self.res_2b = CondResBlock(args,
                                   filters=filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   rescale=True,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)

        self.res_3a = CondResBlock(args,
                                   filters=2 * filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   downsample=False,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)
        self.res_3b = CondResBlock(args,
                                   filters=2 * filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   rescale=True,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)

        self.res_4a = CondResBlock(args,
                                   filters=4 * filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   downsample=False,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)
        self.res_4b = CondResBlock(args,
                                   filters=4 * filter_dim,
                                   latent_dim=latent_dim,
                                   im_size=im_size,
                                   rescale=True,
                                   classes=2,
                                   norm=args.norm,
                                   spec_norm=args.spec_norm)

        self.self_attn = Self_Attn(4 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_mid_model(self):
        args = self.args
        filter_dim = args.n_f
        latent_dim = args.n_f
        im_size = args.img_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = CondResBlock(args,
                                       filters=filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       downsample=True,
                                       rescale=False,
                                       classes=2)
        self.mid_res_1b = CondResBlock(args,
                                       filters=filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       rescale=False,
                                       classes=2)

        self.mid_res_2a = CondResBlock(args,
                                       filters=filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       downsample=True,
                                       rescale=False,
                                       classes=2)
        self.mid_res_2b = CondResBlock(args,
                                       filters=filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       rescale=True,
                                       classes=2)

        self.mid_res_3a = CondResBlock(args,
                                       filters=2 * filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       downsample=False,
                                       classes=2)
        self.mid_res_3b = CondResBlock(args,
                                       filters=2 * filter_dim,
                                       latent_dim=latent_dim,
                                       im_size=im_size,
                                       rescale=True,
                                       classes=2)

        self.mid_energy_map = nn.Linear(filter_dim * 4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        args = self.args
        filter_dim = args.n_f
        latent_dim = args.n_f
        im_size = args.img_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = CondResBlock(args,
                                         filters=filter_dim,
                                         latent_dim=latent_dim,
                                         im_size=im_size,
                                         downsample=True,
                                         rescale=False,
                                         classes=2)
        self.small_res_1b = CondResBlock(args,
                                         filters=filter_dim,
                                         latent_dim=latent_dim,
                                         im_size=im_size,
                                         rescale=False,
                                         classes=2)

        self.small_res_2a = CondResBlock(args,
                                         filters=filter_dim,
                                         latent_dim=latent_dim,
                                         im_size=im_size,
                                         downsample=True,
                                         rescale=False,
                                         classes=2)
        self.small_res_2b = CondResBlock(args,
                                         filters=filter_dim,
                                         latent_dim=latent_dim,
                                         im_size=im_size,
                                         rescale=True,
                                         classes=2)

        self.small_energy_map = nn.Linear(filter_dim * 2, 1)

    def main_model(self, x, latent):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)

        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        if self.args.self_attn:
            x, _ = self.self_attn(x)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.energy_map(x)

        return energy

    def mid_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.mid_energy_map(x)

        return energy

    def small_model(self, x, latent):
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        energy = self.small_energy_map(x)

        return energy

    def label_map(self, latent):
        x = self.act(self.map_fc1(latent))
        x = self.act(self.map_fc2(x))
        x = self.act(self.map_fc3(x))
        x = self.act(self.map_fc4(x))

        return x

    def forward(self, x, latent=None):
        assert (latent is None) == (not self.cond)

        args = self.args

        if not self.cond:
            latent = None

        energy = self.main_model(x, latent)

        if args.multiscale:
            large_energy = energy
            mid_energy = self.mid_model(x, latent)
            small_energy = self.small_model(x, latent)
            energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

        return energy


class CNNCondBigger(nn.Module):
    def __init__(self, n_c=1, n_f=8, label_dim=0, uncond=False, cond_mode=None, small_mlp=False, spectral=False):
        super(CNNCondBigger, self).__init__()

        self.label_dim = label_dim
        self.uncond = uncond
        self.cond_mode = cond_mode

        if uncond:
            if spectral:
                self.cnn = nn.Sequential(
                    spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                    nn.SiLU(inplace=True),
                    spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                    nn.SiLU(inplace=True),
                    spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                    nn.SiLU(inplace=True),
                    spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                    nn.SiLU(inplace=True),
                    spectral_norm(nn.Conv2d(n_f * 8, 1, 4, 1, 0))
                )
            else:
                self.cnn = nn.Sequential(
                    nn.Conv2d(n_c, n_f, 3, 1, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(n_f * 8, 1, 4, 1, 0)
                )
        else:
            assert cond_mode is not None
            if cond_mode == "dot":
                cnn_out_dim = 1024
                if n_f == 8:
                    pass
                elif n_f == 16:
                    cnn_out_dim *= 2
                elif n_f == 32:
                    cnn_out_dim *= 4
                elif n_f == 64:
                    cnn_out_dim *= 8
                else:
                    raise ValueError

                if spectral:
                    self.cnn = nn.Sequential(
                        spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                        nn.Flatten(),
                    )
                else:
                    self.cnn = nn.Sequential(
                        nn.Conv2d(n_c, n_f, 3, 1, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                        nn.Flatten(),
                    )

                if small_mlp:
                    self.mlp = small_mlp_ebm(nin=label_dim, nout=cnn_out_dim, spectral=spectral)
                else:
                    self.mlp = smooth_mlp_ebm_bigger('swish', nin=label_dim, nout=cnn_out_dim, spectral=spectral)
            elif cond_mode == "cnn-mlp":
                cnn_out_dim = 128
                if n_f == 8:
                    pass
                elif n_f == 16:
                    cnn_out_dim *= 2
                elif n_f == 32:
                    cnn_out_dim *= 4
                elif n_f == 64:
                    cnn_out_dim *= 8
                else:
                    raise ValueError

                if spectral:
                    self.cnn = nn.Sequential(
                        spectral_norm(nn.Conv2d(n_c, n_f, 3, 1, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f, n_f * 2, 4, 2, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)),
                        nn.SiLU(inplace=True),
                        spectral_norm(nn.Conv2d(n_f * 8, n_f, 1, 1, 0)),
                        nn.Flatten(),
                    )
                else:
                    self.cnn = nn.Sequential(
                        nn.Conv2d(n_c, n_f, 3, 1, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(n_f * 8, n_f, 1, 1, 0),
                        nn.Flatten(),
                    )

                if small_mlp:
                    self.mlp = small_mlp_ebm(nin=label_dim + cnn_out_dim, nout=1, spectral=spectral)
                else:
                    self.mlp = smooth_mlp_ebm_bigger('swish', nin=label_dim + cnn_out_dim, nout=1, spectral=spectral)
            else:
                raise ValueError

    def forward(self, x, label=None):
        x = self.cnn(x)
        if label is None:
            assert self.uncond
            return x.squeeze()
        else:
            label = label.flatten(start_dim=1)
            if self.cond_mode == "dot":
                label = self.mlp(label)
                return (x * label).sum(-1)
            elif self.cond_mode == "cnn-mlp":
                return self.mlp(torch.cat([x, label], dim=1)).squeeze()
            else:
                raise ValueError


class MNISTCNNCond(nn.Module):
    def __init__(self, n_c=1, n_f=8, label_dim=0, cond_mode=None, small_mlp=False):
        super(MNISTCNNCond, self).__init__()

        self.label_dim = label_dim
        self.cond_mode = cond_mode
        assert cond_mode is not None

        if cond_mode in ("dot", "cos"):
            cnn_out_dim = 1024
            assert n_f in (8, 16, 32, 64), f"Unrecognized n_f {n_f}"
            cnn_out_dim *= n_f // 8

            self.cnn = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                nn.Flatten(),
            )

            if small_mlp:
                self.mlp = small_mlp_ebm(nin=label_dim, nout=cnn_out_dim, spectral=False)
            else:
                self.mlp = smooth_mlp_ebm_bigger('swish', nin=label_dim, nout=cnn_out_dim, spectral=False)
        elif cond_mode == "cnn-mlp":
            cnn_out_dim = 128
            assert n_f in (8, 16, 32, 64), f"Unrecognized n_f {n_f}"
            cnn_out_dim *= n_f // 8

            self.cnn = nn.Sequential(
                nn.Conv2d(n_c, n_f, 3, 1, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(n_f * 8, n_f, 1, 1, 0),
                nn.Flatten(),
            )

            if small_mlp:
                self.mlp = small_mlp_ebm(nin=label_dim + cnn_out_dim, nout=1, spectral=False)
            else:
                self.mlp = smooth_mlp_ebm_bigger('swish', nin=label_dim + cnn_out_dim, nout=1, spectral=False)
        else:
            raise ValueError(f"Unrecognized cond_mode {cond_mode}")

    def forward(self, x, label):
        x = self.cnn(x)
        label = label.flatten(start_dim=1)
        if self.cond_mode in ("dot", "cos"):
            label = self.mlp(label)
            if self.cond_mode == "dot":
                return (x * label).sum(-1)
            else:
                return (x * label).sum(-1) / (x.norm(p=2, dim=-1) * label.norm(p=2, dim=-1))
        elif self.cond_mode == "cnn-mlp":
            return self.mlp(torch.cat([x, label], dim=1)).squeeze()
        else:
            raise ValueError


class ZapposCNNCond(nn.Module):
    def __init__(self, img_size=64, n_c=1, n_f=8, label_dim=0, cond_mode=None,
                 small_mlp=False, small_mlp_nhidden=256, all_binary=False):
        super(ZapposCNNCond, self).__init__()

        self.label_dim = label_dim
        self.cond_mode = cond_mode
        assert cond_mode is not None

        assert n_f in (8, 16, 32, 64), f"Unrecognized n_f {n_f}"

        cnn_layers = [nn.Conv2d(n_c, n_f, 3, 1, 1),
                      nn.SiLU(inplace=True),
                      nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
                      nn.SiLU(inplace=True),
                      nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
                      nn.SiLU(inplace=True),
                      nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1)]

        if img_size == 64:
            cnn_layers.extend([nn.SiLU(inplace=True),
                               nn.Conv2d(n_f * 8, n_f * 8, 4, 2, 1)])
        else:
            if img_size != 32:
                raise ValueError(f"Unrecognized img size {img_size}")

        if cond_mode in ("dot", "cos"):
            cnn_out_dim = 1024
            cnn_out_dim *= n_f // 8

            nin, nout = label_dim, cnn_out_dim

            # get input features to last layer
            cnn_layers = cnn_layers + [nn.Flatten()]

        elif cond_mode in ("cnn-mlp", "poj"):
            cnn_out_dim = 128
            cnn_out_dim *= n_f // 8

            if cond_mode == "cnn-mlp":
                nin, nout = label_dim + cnn_out_dim, 1
            elif cond_mode == "poj":
                nin, nout = cnn_out_dim, label_dim
                if all_binary:
                    nout *= 2
            else:
                raise ValueError(f"Unrecognized cond mode {cond_mode}")

            # add 1x1 conv layer
            cnn_layers = cnn_layers + [nn.SiLU(inplace=True),
                                       nn.Conv2d(n_f * 8, n_f, 1, 1, 0),
                                       nn.Flatten()]
        else:
            raise ValueError(f"Unrecognized cond_mode {cond_mode}")

        self.cnn = nn.Sequential(*cnn_layers)

        if small_mlp:
            mlp_ = partial(small_mlp_ebm, spectral=False, nhidden=small_mlp_nhidden)
        else:
            mlp_ = partial(smooth_mlp_ebm_bigger, 'swish', spectral=False)

        self.mlp = mlp_(nin=nin, nout=nout)

    def forward(self, x, label=None):
        assert implies(label is None, self.cond_mode == "poj")
        x = self.cnn(x)
        if self.cond_mode == "poj":
            return self.mlp(x).squeeze()
        else:
            label = label.flatten(start_dim=1)
            if self.cond_mode in ("dot", "cos"):
                label = self.mlp(label)
                if self.cond_mode == "dot":
                    return (x * label).sum(-1)
                else:
                    return (x * label).sum(-1) / (x.norm(p=2, dim=-1) * label.norm(p=2, dim=-1))
            elif self.cond_mode == "cnn-mlp":
                return self.mlp(torch.cat([x, label], dim=1)).squeeze()
            else:
                raise ValueError


def small_cnn(n_c=3, n_f=32):
    return nn.Sequential(
        nn.Conv2d(n_c, n_f, 3, 1, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 8, 1, 4, 1, 0)
    )


def medium_cnn(n_c=3, n_f=64):
    return nn.Sequential(
        nn.Conv2d(n_c, n_f, 3, 1, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 8, n_f * 8, 4, 2, 1),
        nn.SiLU(inplace=True),
        nn.Conv2d(n_f * 8, 1, 4, 1, 0)
    )


def get_scales(net):
    scales = []
    for k, v in net.named_children():
        if int(k) % 2 == 0:
            scales.append(v.scale.item())

    return scales


def ema_model(model, model_ema, mu=0.99):
    assert 0 <= mu <= 1
    if mu != 1:  # if mu is 1, the ema parameters stay the same
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def ema_params(model, model_ema):
    """
    Check if params are the same.
    """
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        if not torch.eq(param_ema.data, param.data).all():
            return False
    return True


def get_sample_q(init_x_random, init_b_random, reinit_freq, step_size, sigma, device, transform=None, one_hot_b=None,
                 only_transform_buffer=False):

    def sample_p_0(replay_buffer, bs, y=None, y_replay_buffer=None, y_cond=None):

        if len(replay_buffer) == 0 and not isinstance(replay_buffer, ReplayBuffer):
            return init_x_random(bs), []
        assert implies(y_cond is not None, y_replay_buffer is not None), "Buffer conditional sampling requires buffer"
        assert implies(y is not None, y_replay_buffer is not None), "Joint initialization requires buffer"
        assert y is None or y_cond is None, "Buffer conditional sampling or joint initialization, but not both"

        # conditional buffer sampling
        if y_cond is not None:
            if not isinstance(y_cond, int):
                assert len(y_cond) == 1, "Init conditionally with 1 label only!"
                assert len(y_cond.shape) == 1, "Wrong shape"
            if isinstance(replay_buffer, ReplayBuffer):
                raise NotImplementedError("Condtional yd buffer is not available")
            else:
                replay_buffer = replay_buffer[y_replay_buffer == y_cond]
        buffer_size = len(replay_buffer)
        if buffer_size == 0 and not isinstance(replay_buffer, ReplayBuffer):
            logger(f"====== WARNING ====== Encountered buffer size 0 for class {y_cond}")
            return init_x_random(bs), []

        choose_random = (torch.rand(bs) < reinit_freq).float().to(device)

        if y is None:
            random_samples = init_x_random(bs)

            buffer_samples, inds = get_buffer_samples(replay_buffer, buffer_size, bs)

            if only_transform_buffer:
                buffer_samples = transform(buffer_samples)
                samples = get_random_or_buffer_samples(choose_random, random_samples, buffer_samples)
            else:
                samples = get_random_or_buffer_samples(choose_random, random_samples, buffer_samples)
                samples = samples.to(device)  # TODO: i'm pretty sure this is redundant, everything's already on device
                samples = transform(samples)
        else:
            random_x_samples, random_y_samples = init_x_random(bs), init_b_random(bs, label_samples=True)

            # pass inds when sampling y, so we get the corresponding y buffer samples
            buffer_x_samples, inds = get_buffer_samples(replay_buffer, buffer_size, bs)
            buffer_y_samples, inds = get_buffer_samples(y_replay_buffer, buffer_size, bs, inds)

            if only_transform_buffer:
                buffer_x_samples = transform(buffer_x_samples)
                x_samples = get_random_or_buffer_samples(choose_random, random_x_samples, buffer_x_samples)
                y_samples = get_random_or_buffer_samples(choose_random, random_y_samples, buffer_y_samples)

                x_samples, y_samples = x_samples.to(device), y_samples.to(device)
                y_samples = one_hot_b(y_samples.type(random_y_samples.dtype))
            else:
                x_samples = get_random_or_buffer_samples(choose_random, random_x_samples, buffer_x_samples)
                y_samples = get_random_or_buffer_samples(choose_random, random_y_samples, buffer_y_samples)
                x_samples, y_samples = x_samples.to(device), y_samples.to(device)
                x_samples, y_samples = transform(x_samples), one_hot_b(y_samples.type(random_y_samples.dtype))

            samples = (x_samples, y_samples)

        return samples, inds

    def get_buffer_samples(replay_buffer, buffer_size, bs, inds=None):
        if inds is None and not isinstance(replay_buffer, ReplayBuffer):
            # if yd buffer, let it generate its own inds
            inds = torch.randint(0, buffer_size, (bs,))

        if isinstance(replay_buffer, ReplayBuffer):
            buffer_samples, inds = replay_buffer.sample(bs, inds)
        else:
            buffer_samples = replay_buffer[inds]

        return buffer_samples.to(device), inds

    def get_random_or_buffer_samples(choose_random, random_samples, buffer_samples):
        assert random_samples.shape == buffer_samples.shape
        assert len(choose_random.shape) == 1
        assert choose_random.shape[0] == random_samples.shape[0]

        choose_random = choose_random[:, None]
        structured_shape = random_samples.shape

        if len(random_samples.shape) == 2:
            pass
        elif len(random_samples.shape) == 3:
            random_samples = random_samples.view(random_samples.shape[0], -1)
            buffer_samples = random_samples.view(buffer_samples.shape[0], -1)
        elif len(random_samples.shape) == 4:
            choose_random = choose_random[:, None, None]
        else:
            raise ValueError(f"Unrecognized samples shape {random_samples.shape}")

        final_samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

        return final_samples.view(structured_shape)

    def init_sampling(replay_buffer, x, y=None, y_replay_buffer=None, y_cond=None):
        """
        Generate initial samples and buffer inds of those samples (if buffer is used)
        """

        if y_replay_buffer is not None:
            assert len(replay_buffer) == len(y_replay_buffer)
            assert isinstance(replay_buffer, ReplayBuffer) == isinstance(y_replay_buffer, ReplayBuffer)

        if len(replay_buffer) == 0 and not isinstance(replay_buffer, ReplayBuffer):
            init_x_sample = x
            init_y_sample = y
            buffer_inds = []
        else:
            bs = x.size(0)
            init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs,
                                                  y=y, y_replay_buffer=y_replay_buffer, y_cond=y_cond)
            if y is None:
                init_x_sample, init_y_sample = init_sample, None
            else:
                init_x_sample, init_y_sample = init_sample

        init_x_sample = init_x_sample.clone().detach().requires_grad_(True)

        if init_y_sample is not None:
            init_y_sample = init_y_sample.clone().detach().requires_grad_(True)
            init_sample = (init_x_sample, init_y_sample)
        else:
            init_sample = init_x_sample

        return init_sample, buffer_inds

    def step_sampling(f, x_k, sigma_=sigma, bp=False):
        if bp:
            # track changes for autograd so we can do truncated langevin backprop
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True, create_graph=True)[0]
            x_k = x_k + step_size * f_prime + sigma_ * torch.randn_like(x_k)  # += is in-place!!
        else:
            f_prime = torch.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += step_size * f_prime + sigma_ * torch.randn_like(x_k)
        return x_k

    def set_sampling(x_k, replay_buffer, y_replay_buffer, buffer_inds, update_buffer=True, y=None, y_k=None):
        if y_replay_buffer is not None:
            assert len(replay_buffer) == len(y_replay_buffer)
            assert isinstance(replay_buffer, ReplayBuffer) == isinstance(y_replay_buffer, ReplayBuffer)
            assert implies(update_buffer, y_k is not None)

        final_samples = x_k.detach()
        if y_k is not None:
            final_y_samples = y_k.detach()
        else:
            final_y_samples = None
        num_added = None

        # update replay buffer
        if isinstance(replay_buffer, ReplayBuffer):
            if update_buffer:
                num_added, which_added = replay_buffer.add(final_samples)
                if y_replay_buffer is not None:
                    y_replay_buffer.add(final_y_samples, which_added)
        else:
            if len(replay_buffer) > 0 and update_buffer:
                replay_buffer[buffer_inds] = final_samples
                if y is not None:
                    # noinspection PyUnresolvedReferences
                    y_replay_buffer[buffer_inds] = y.squeeze()

        if y_k is not None:
            final_samples = (final_samples, one_hot_b(final_y_samples).detach())

        return final_samples, num_added

    return init_sampling, step_sampling, set_sampling


def _plot_pr_curve(save_dir, fn, precision, recall, ap, freq, dset_label_info):
    plt.clf()

    with open(f"{save_dir}/cache_pr_{fn}.pickle", "wb") as f:
        d = {"precision": precision, "recall": recall, "ap": ap, "freq": freq, "dset_label_info": dset_label_info}
        pickle.dump(d, f, protocol=4)

    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Precision-Recall Curve. Average Precision: {ap:.4f} (Freq. {freq:.4f})")
    plt.savefig(f"{save_dir}/pr_{fn}.png")


def _plot_roc_curve(save_dir, fn, fpr, tpr, auroc, freq, dset_label_info):
    plt.clf()

    with open(f"{save_dir}/cache_auroc_{fn}.pickle", "wb") as f:
        d = {"fpr": fpr, "tpr": tpr, "auroc": auroc, "freq": freq, "dset_label_info": dset_label_info}
        pickle.dump(d, f, protocol=4)

    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title(f"AUROC: {auroc:.4f} (Freq. {freq:.4f})")
    plt.savefig(f"{save_dir}/auroc_{fn}.png")


def _plot_ece_hist_no_cache(save_dir, fn, reliability_diag, ece, freq, dset_label_info):
    plt.clf()

    with open(f"{save_dir}/cache_ece_{fn}.pickle", "wb") as f:
        d = {"reliability_diag": reliability_diag, "ece": ece, "freq": freq, "dset_label_info": dset_label_info}
        pickle.dump(d, f, protocol=4)

    conf, acc = reliability_diag
    conf[conf.isnan()] = 0
    acc[acc.isnan()] = 0

    num_bins = 20
    tau_tab = torch.linspace(0, 1, num_bins + 1)
    binned_confs = tau_tab[torch.searchsorted(tau_tab, conf)]

    font = {'family': 'normal',
            'size': 20}
    plt.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.bar(binned_confs, acc, color="lightcoral", linewidth=.1, edgecolor="black", align="edge", width=1 / num_bins)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--")
    plt.annotate(f"ECE: {ece * 100:.2f}%", xy=(140, 240), xycoords='axes points',  # TODO: xy=(130, 240) for ebm (140 s)
                 size=20, ha='right', va='top',
                 bbox=dict(boxstyle='round', fc="#f6b2b2", color="#f6b2b2"))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([.0, .5, 1.])
    plt.yticks([.0, .5, 1.])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.gcf().subplots_adjust(right=.95, left=.14, bottom=.15)
    plt.savefig(f"{save_dir}/ece_{fn}.png")
    plt.close(fig)


def _plot_ece_hist(save_dir, fn, reliability_diag=None, ece=None, freq=None, dset_label_info=None,
                   from_cache=False):

    if from_cache:
        with open(f"{save_dir}/cache_ece_{fn}.pickle", "rb") as f:
            d = pickle.load(f)
    else:
        d = {"reliability_diag": reliability_diag, "ece": ece, "freq": freq, "dset_label_info": dset_label_info}

    _plot_ece_hist_no_cache(save_dir, fn, **d)


def main(args):
    zero_shot_tst_dset = zero_shot_combos = tst_dset_len = None
    if args.zero_shot:
        *_rest, trn_dset_freqs = get_data_batch(args, args.data, args.batch_size, args.device, return_dset=True,
                                                zero_shot=True)
        data_batch, data_val_batch, data_test_batch, dset_label_info, dset_lengths = _rest
        trn_dset_len = dset_lengths['trn']
    else:
        if args.data == "mnist" or args.dsprites_test or args.full_test:

            if args.data == "celeba" and args.dset_split_type == "zero_shot":
                *_rest, trn_dset_freqs, zero_shot_combos, trn_dset_len \
                    = get_data_batch(args, args.data, args.batch_size, args.device, return_dset=True,
                                     return_trn_len=True, return_tst_dset=True)
            else:
                *_rest, trn_dset_freqs, trn_dset_len = get_data_batch(args, args.data, args.batch_size, args.device,
                                                                      return_dset=True, return_trn_len=True)
            
            if args.data in ("utzappos", "celeba", "cub"):
                if args.data == "celeba" and args.dset_split_type == "zero_shot":
                    data_batch, data_test_batch, dset_label_info, tst_dset_len, zero_shot_tst_dset = _rest
                else:
                    data_batch, data_test_batch, dset_label_info, tst_dset_len = _rest
                    tst_dset_len, zero_shot_tst_dset = None, None
            else:
                data_batch, data_test_batch = _rest
                dset_label_info, tst_dset_len, zero_shot_tst_dset = None, None, None
        else:
            data_batch = get_data_batch(args, args.data, args.batch_size, args.device)
            data_test_batch, trn_dset_freqs, trn_dset_len, tst_dset_len, dset_label_info = None, None, None, None, None
            zero_shot_tst_dset = None
        data_val_batch = dset_lengths = None

    label_shape = (-1,)
    all_labels = 0
    all_binary = False
    unif_label_shape = None
    diff_label_axes = None
    struct_mask = None
    fix_label_axis_mask = None

    if args.data == "celeba":
        if args.dset_split_type == "zero_shot":
            custom_cls_combos = CELEBA_ZERO_SHOT_COMBOS
            if args.eval_filter_to_tst_combos:
                custom_cls_combos = [combo for combo in CELEBA_ZERO_SHOT_COMBOS if combo in zero_shot_combos]
        else:
            custom_cls_combos = [{"Male": 1, "No_Beard": 0},
                                 {"Male": 0, "Smiling": 1},
                                 {"Male": 0, "Wearing_Lipstick": 1},
                                 {"Male": 0, "Arched_Eyebrows": 1},
                                 {"Male": 0, "High_Cheekbones": 1},
                                 {"Male": 0, "Bangs": 1},
                                 {"Male": 0, "Young": 1},
                                 {"Male": 0, "Smiling": 1, "Young": 1},
                                 {"Male": 0, "Smiling": 1, "Young": 1, "Wavy_Hair": 1},
                                 {"Male": 1, "No_Beard": 0, "Black_Hair": 1},
                                 {"Male": 1, "No_Beard": 0, "Black_Hair": 1, "Bushy_Eyebrows": 1}]
    else:
        custom_cls_combos = [{"Gender_Men": 1, "Category_Boots": 1, "Material_Leather": 1},
                             {"Gender_Men": 1, "Category_Boots": 1, "Closure_Lace up": 1},
                             {"Gender_Men": 1, "SubCategory_Sneakers and Athletic Shoes": 1, "Closure_Lace up": 1},
                             {"Gender_Women": 1, "Category_Boots": 1, "Closure_Lace up": 1},
                             {"Gender_Women": 1, "Category_Boots": 1, "Material_Leather": 1},
                             {"Gender_Women": 1, "Closure_Slip-On": 1, "SubCategory_Heels": 1},
                             {"Gender_Women": 1, "Closure_Slip-On": 1, "Material_Leather": 1},
                             {"Gender_Women": 1, "Closure_Slip-On": 1, "SubCategory_Heels": 1, "Material_Leather": 1}]

    if args.data in IMG_DSETS:
        if args.data == "dsprites":
            data_dim = 32 ** 2  # need to actually set this because using MLP
        else:
            data_dim = 784
        if args.mode == "uncond" and not args.uncond_poj:
            label_dim = 0
        else:
            if args.data == "mwa":
                label_shape = (4, 4, 4, 4, 4)
                unif_label_shape, = set(label_shape)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            elif args.data == "celeba":
                all_binary = True
                if not args.uncond_poj:
                    label_dim = all_labels = len(dset_label_info)
                    label_shape = (label_dim,)  # set this to reuse the plotting code
                else:
                    label_dim = 0
            elif args.data == "cub":
                all_binary = True
                label_dim = all_labels = len(dset_label_info)
                label_shape = (label_dim,)  # set this to reuse the plotting code
            elif args.data == "utzappos_old":
                label_shape = (4, 21, 7, 14, 18, 8, 62, 19)
                label_axes = get_label_axes(torch.tensor(label_shape, device=args.device))
                diff_label_axes = get_diff_label_axes(label_axes)
                struct_mask = get_struct_mask(label_axes)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            elif args.data == "utzappos":
                all_binary = True
                if not args.uncond_poj:
                    label_dim = all_labels = len(dset_label_info)
                    label_shape = (label_dim,)  # set this to reuse the plotting code
                else:
                    label_dim = 0
            elif args.data == "dsprites":
                label_shape = (16, 16)
                unif_label_shape, = set(label_shape)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            else:
                label_dim = 10
                all_labels = label_dim

        def plot_samples(x, fn):
            if x.size(0) > 0:
                pad_value = 0
                if args.data == "celeba":
                    img_size = args.img_size
                    num_channels = 3
                    # x = (x + 1) / 2  # colours and shit
                elif args.data == "utzappos":
                    img_size = args.img_size
                    num_channels = 3
                elif args.data == "cub":
                    img_size = args.img_size
                    num_channels = 3
                elif args.data == "mwa":
                    img_size = args.img_size
                    num_channels = 3
                    x = (x + 1) / 2
                elif args.data == "dsprites":
                    img_size = 32
                    num_channels = 1
                    x = torch.sigmoid(x)
                    x = (x - 1e-6) / (1 - 2 * 1e-6)
                    pad_value = 1
                elif args.data == "mnist":
                    if args.small_cnn:
                        img_size = args.img_size
                    else:
                        img_size = 28
                    num_channels = 1
                    if args.logit:
                        x = torch.sigmoid(x)
                        x = (x - 1e-6) / (1 - 2 * 1e-6)
                else:
                    img_size = 28
                    num_channels = 1
                save_image(x.view(x.size(0), num_channels, img_size, img_size), fn,
                           normalize=False, nrow=int(x.size(0) ** .5), pad_value=pad_value)

        if args.data == "celeba":
            if label_dim == 0 and not args.uncond_poj:
                # net_fn = lambda *_: CelebAModel(args)
                if args.img_size == 32:
                    net_fn = lambda *_: small_cnn(n_f=args.n_f)
                elif args.img_size == 64:
                    net_fn = lambda *_: medium_cnn(n_f=args.n_f)
                else:
                    raise ValueError(f"Got img size {args.img_size} for CelebA")
            else:
                net_fn = lambda *_: ZapposCNNCond(img_size=args.img_size, n_c=3, n_f=args.n_f,
                                                  label_dim=len(dset_label_info) if args.uncond_poj else label_dim,
                                                  cond_mode="poj" if args.model == "poj" else args.cond_mode,
                                                  small_mlp=args.small_mlp,
                                                  small_mlp_nhidden=args.small_mlp_nhidden,
                                                  all_binary=all_binary)

        elif args.data == "cub":
            if label_dim == 0:
                # net_fn = lambda *_: CelebAModel(args)
                if args.img_size == 32:
                    net_fn = lambda *_: small_cnn(n_f=args.n_f)
                elif args.img_size == 64:
                    net_fn = lambda *_: medium_cnn(n_f=args.n_f)
                else:
                    raise ValueError(f"Got img size {args.img_size} for CelebA")
            else:
                net_fn = lambda *_: ZapposCNNCond(img_size=args.img_size, n_c=3, n_f=args.n_f,
                                                  label_dim=label_dim,
                                                  cond_mode="poj" if args.model == "poj" else args.cond_mode,
                                                  small_mlp=args.small_mlp,
                                                  small_mlp_nhidden=args.small_mlp_nhidden,
                                                  all_binary=all_binary)

        elif args.data == "utzappos":
            if label_dim == 0 and not args.uncond_poj:
                # net_fn = lambda *_: CelebAModel(args)
                if args.img_size == 32:
                    net_fn = lambda *_: small_cnn(n_f=args.n_f)
                elif args.img_size == 64:
                    net_fn = lambda *_: medium_cnn(n_f=args.n_f)
                else:
                    raise ValueError(f"Got img size {args.img_size} for UTZappos")
            else:
                net_fn = lambda *_: ZapposCNNCond(img_size=args.img_size, n_c=3, n_f=args.n_f,
                                                  label_dim=len(dset_label_info) if args.uncond_poj else label_dim,
                                                  cond_mode="poj" if args.model == "poj" else args.cond_mode,
                                                  small_mlp=args.small_mlp,
                                                  small_mlp_nhidden=args.small_mlp_nhidden,
                                                  all_binary=all_binary)
                # net_fn = lambda *_: CNNCond(label_dim=label_dim, uncond=label_dim == 0)
        elif args.data == "mwa":
            if args.model != "joint":
                raise NotImplementedError
            if label_dim == 0 and not args.cond_arch:
                net_fn = lambda *_: small_cnn()
            else:
                net_fn = lambda *_: CNNCond(label_dim=label_dim, uncond=label_dim == 0)
        elif args.data == "dsprites":
            if args.model != "joint":
                raise NotImplementedError
            if args.cnn:
                net_fn = lambda *_: CNNCondBigger(n_c=1, n_f=args.n_f,
                                                  label_dim=label_dim,
                                                  uncond=label_dim == 0,
                                                  cond_mode=args.cond_mode,
                                                  small_mlp=args.small_mlp,
                                                  spectral=args.spectral)
            else:
                net_fn = partial(smooth_mlp_ebm_bigger, 'elu')
        else:
            if args.model != "joint":
                raise NotImplementedError
            if args.cnn:
                net_fn = lambda *_: MNISTModel(args)
            elif args.small_cnn:
                if label_dim == 0:
                    net_fn = lambda *_: small_cnn(n_c=1)
                else:
                    net_fn = lambda *_: MNISTCNNCond(n_c=1, n_f=args.n_f,
                                                     label_dim=label_dim,
                                                     cond_mode=args.cond_mode,
                                                     small_mlp=args.small_mlp)
            else:
                net_fn = partial(smooth_mlp_ebm_bigger, args.mnist_act)

    else:
        data_dim = 2
        if args.mode == "uncond":
            label_dim = 0
        else:
            if args.data == "rings":
                label_dim = 4
                all_labels = label_dim
            elif args.data == "rings_struct":
                label_shape = (2, 2)
                unif_label_shape, = set(label_shape)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            elif args.data in ["checkerboard", "8gaussians"]:
                label_dim = 8
                all_labels = label_dim
            elif args.data == "circles":
                all_binary = True
                label_dim = 1
                all_labels = label_dim
            elif "8gaussians_struct" in args.data:
                label_shape = (2, 4)
                label_axes = get_label_axes(torch.tensor(label_shape, device=args.device))
                diff_label_axes = get_diff_label_axes(label_axes)
                struct_mask = get_struct_mask(label_axes)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            elif args.data in ["8gaussians_multi", "8gaussians_hierarch", "8gaussians_hierarch_missing"]:
                label_shape = (2, 2, 2)
                unif_label_shape, = set(label_shape)
                label_dim = sum(label_shape)
                all_labels = int(np.prod(label_shape))
            elif "8gaussians_hierarch_binarized" in args.data:
                all_binary = True
                label_dim = all_labels = 3
                label_shape = (label_dim,)  # set this to reuse the plotting code
            else:
                raise ValueError(f"Unrecognized data {args.data}")

        def plot_samples(x, fn):
            plt.clf()
            plt_samples(x, plt.gca())
            plt.savefig(fn)

        net_fn = partial(smooth_mlp_ebm, args.spectral)

    if args.model == "joint":
        net_fn_args = (data_dim + label_dim, 1)
    elif args.model == "poj":
        if all_binary:
            net_fn_args = (data_dim, 2 * label_dim)
        else:
            net_fn_args = (data_dim, label_dim)
    else:
        raise ValueError(f"Unrecognized model {args.model}")

    logp_net = net_fn(*net_fn_args)
    if args.mode == "sup":
        ema_logp_net = logp_net
    else:
        ema_logp_net = net_fn(*net_fn_args)
        ema_logp_net.load_state_dict(logp_net.state_dict())  # copy the weights of logp_net

    if args.multi_gpu:
        logp_net = nn.DataParallel(logp_net).cuda()
        ema_logp_net = nn.DataParallel(ema_logp_net).cuda()
    elif args.old_multi_gpu:
        logp_net = nn.DataParallel(logp_net)
        ema_logp_net = nn.DataParallel(ema_logp_net)
        logp_net.to(args.device)
        ema_logp_net.to(args.device)
    else:
        # idk if this does anything
        logp_net.to(args.device)
        ema_logp_net.to(args.device)

    logger(logp_net)

    if args.log_ema:
        logger(f"Params the same: {ema_params(logp_net, ema_logp_net)}")

    if label_dim > 1 and not all_binary:
        if label_shape != (-1,):
            if unif_label_shape is None:
                sampler = DiffSamplerMultiDim(n_steps=1, approx=True, temp=args.temp, struct=True,
                                              label_shape=label_shape,
                                              shift_logit=not args.no_shift_logit,
                                              other_reverse_changes=args.other_reverse_changes,
                                              device=args.device)
                cond_sampler = sampler
            else:
                sampler = DiffSamplerMultiDim(n_steps=1, approx=True, temp=args.temp, struct=False,
                                              label_shape=label_shape,
                                              shift_logit=not args.no_shift_logit,
                                              other_reverse_changes=args.other_reverse_changes,
                                              device=args.device)
                cond_sampler = sampler
        else:
            sampler = DiffSamplerMultiDim(n_steps=1, approx=True, temp=args.temp,
                                          shift_logit=not args.no_shift_logit,
                                          other_reverse_changes=args.other_reverse_changes, device=args.device)
            cond_sampler = sampler
    else:
        sampler = DiffSampler(n_steps=1, approx=True, temp=args.temp)
        cond_sampler = sampler

    def init_x_random(bs=args.batch_size, to_device=True):
        init_x_random_samples_ = x_init_dist.sample((bs,))
        if to_device:
            init_x_random_samples_ = init_x_random_samples_.to(args.device)
        return init_x_random_samples_

    def init_b_random(bs=args.batch_size, label_samples=False, to_device=True):
        samples_ = b_init_dist.sample((bs,))
        if to_device:
            samples_ = samples_.to(args.device)
        if label_samples:
            samples_ = label(samples_)
        return samples_

    def energy(net, x, b=None):
        """
        Define tuple interface for energy so we can hold one of them fixed while sampling.
        """
        if args.model == "poj":
            return logit_logsumexp(net(x)).sum(-1)

        if label_dim == 0:
            return net(x).squeeze()
        else:
            assert b is not None

            if args.data in ["mwa", "utzappos"]:
                return net(x, b.float()).squeeze()
            else:
                if args.cnn or args.small_cnn:
                    return net(x, b).squeeze()
                else:
                    if args.data in ["dsprites", "8gaussians_multi",
                                     "8gaussians_hierarch", "8gaussians_hierarch_missing", "rings_struct"]:
                        return net(torch.cat([x, b.view(-1, label_dim).float()], 1)).squeeze()
                    else:
                        return net(torch.cat([x, b.float()], 1)).squeeze()

    def format_fix_label_axis(fix_label_axis):
        if fix_label_axis is None:
            return fix_label_axis

        if isinstance(fix_label_axis, int):
            fix_label_axis = torch.tensor([fix_label_axis])
        elif isinstance(fix_label_axis, list):
            assert len(fix_label_axis) > 0
            assert len(set(map(type, fix_label_axis))) == 1
            assert isinstance(fix_label_axis[0], int)
            fix_label_axis = torch.tensor(fix_label_axis)
        else:
            assert isinstance(fix_label_axis, torch.Tensor)
            if unif_label_shape:
                assert len(fix_label_axis.shape) == 1
        return fix_label_axis.to(args.device)

    def init_sample_b(b_init, fix_label_axes=None):
        """
        Process b_init for sampling.

        Reshape b. If labels need to be fixed, create mask and split labels.

        Returns:
            b:          The labels that need to be resampled.
            b_all:      The entire label, both those that are fixed and those are resampled.
            b_fixed:    The labels that are fixed.
            axis_mask:  The mask from which b and b_fixed can be obtained from b_all.
        """

        # reshape b
        if label_shape != (-1,) and unif_label_shape is not None:
            # in the case of structured labels, we'll want to pass in structured b to the sampler
            b = b_init
            b = b.view(b.shape[0], len(label_shape), unif_label_shape)
        elif all_binary:
            b = b_init  # don't expand dims
        else:
            b = b_init[:, None]

        if fix_label_axes is not None:
            if not all_binary:
                assert (fix_label_axes < len(label_shape)).all()
            b_all = b

            # per example mask (from masking out missing data), in the case of structured labels
            if fix_label_axes.shape[0] == b_init.shape[0]:
                fix_label_axes_nonzero = fix_label_axes.nonzero()[:, -1][:, None]
                set_axis_mask = get_onehot_struct_mask(diff_label_axes, struct_mask, fix_label_axes_nonzero)
                axis_mask = torch.zeros_like(b_all).type(torch.bool)
                axis_mask[fix_label_axes.any(-1)] = set_axis_mask
                b = init_b_random(axis_mask.shape[0])[:, None][axis_mask]  # init missing b
                b_fixed = b_all[~axis_mask]

            # per example mask (from masking out missing data), in the case of simple (unstructured) labels
            elif (fix_label_axes == -1).all():
                axis_mask = (b_all == 0).all(-1)

                b = init_b_random(axis_mask.shape[0])[axis_mask][:, None]  # init missing b
                b_fixed = b_all[~axis_mask][:, None]

            # fixing one label axis everywhere, for conditional sampling
            else:
                axis_mask = (fix_label_axis_mask[..., None] != fix_label_axes[None]).all(-1)

                if unif_label_shape is None and not all_binary:
                    axis_mask = ~get_onehot_struct_mask(diff_label_axes, struct_mask,
                                                        fix_label_axes[:, None]).squeeze()
                    if len(fix_label_axes) > 1:
                        axis_mask = axis_mask.all(0)
                    axis_mask = axis_mask[None]

                b = b_all[:, axis_mask]
                b_fixed = b_all[:, ~axis_mask]
        else:
            b_all = None
            axis_mask = None
            b_fixed = None

        return b, b_all, b_fixed, axis_mask

    def sample_b(net, x_init, b_init, steps=args.n_steps, verbose=False, fix_label_axis=None):
        """
        Sample b from net conditioned on x_init (and b_init).
        """

        a_s = np.zeros(steps)
        lrs = np.zeros(steps)
        lfs = np.zeros(steps)
        l_s = np.zeros(steps)
        m_s = np.zeros(steps)
        p_s = np.zeros(steps)
        h_s = np.zeros(steps)

        fix_label_axis = format_fix_label_axis(fix_label_axis)

        b, b_all, b_fixed, axis_mask = init_sample_b(b_init, fix_label_axis)

        for i in range(steps):
            b, *_info = step_sample_b(net, x_init, b,
                                      axis_mask=axis_mask, b_fixed=b_fixed, b_all=b_all)
            a_s[i], lrs[i], lfs[i], l_s[i], m_s[i], p_s[i], h_s[i] = _info

        if fix_label_axis is not None:
            if per_example_mask(axis_mask):
                b_all[axis_mask] = b.squeeze()
                b_all[~axis_mask] = b_fixed.squeeze()
            else:
                b_all[:, axis_mask] = b
                b_all[:, ~axis_mask] = b_fixed
            b = b_all

        if label_shape != (-1,) and unif_label_shape is not None:
            # we'll want to flatten b again in case it wasn't, so that
            # concatenation operations work outside the function
            b = b.view(b.shape[0], label_dim)

        if label_dim > 1:
            return_tpl = (b.squeeze(), )
        else:
            return_tpl = (b, )

        if verbose:
            log_tpl = (np.mean(a_s), np.mean(lrs), np.mean(lfs), np.mean(l_s), np.mean(m_s), np.mean(p_s), np.mean(h_s))
            return return_tpl + log_tpl
        else:
            return return_tpl[0]

    def step_sample_b(net, x_init, b, axis_mask=None, b_fixed=None, b_all=None):
        assert (axis_mask is None) == (b_fixed is None) == (b_all is None)

        if all_binary:
            # remove any added dims if they exist
            b = b.squeeze()

        if label_shape != (-1,) and unif_label_shape is not None:
            # if we have structured b we'll want to shape it here
            # if b is already structured then this is a no-op, but this isn't the case when doing interleaved sampling
            # if an axis is fixed, the number of labels is reduced, hence -1 instead of len(label_shape)
            b = b.view(b.shape[0], -1, unif_label_shape)

        if axis_mask is not None:
            b_all = torch.zeros_like(b_all)
            if per_example_mask(axis_mask):
                # the axis mask must be per batch example
                assert axis_mask.shape[0] == b_all.shape[0]
                if unif_label_shape:
                    # noinspection PyUnresolvedReferences
                    b_all[~axis_mask] = b_fixed.squeeze(1)
                else:
                    b_all[~axis_mask] = b_fixed
            else:
                b_all[:, ~axis_mask] = b_fixed

            def energy_(b_):
                if unif_label_shape is None:
                    b_ = b_.squeeze()
                if per_example_mask(axis_mask):
                    if unif_label_shape:
                        b_all[axis_mask] = b_.squeeze(1)
                    else:
                        b_all[axis_mask] = b_
                else:
                    b_all[:, axis_mask] = b_
                return energy(net, x_init, b_all.squeeze())
        else:
            if label_dim > 1:
                energy_ = lambda b_: energy(net, x_init, b_.squeeze())
            else:
                energy_ = lambda b_: energy(net, x_init, b_)

        if args.model == "joint":
            if axis_mask is not None:
                if all_binary:
                    b = cond_sampler.step(b.detach(), energy_).detach()
                else:
                    b = cond_sampler.step(b.detach(), energy_, axis_mask).detach()
            else:
                b = sampler.step(b.detach(), energy_).detach()
        else:
            b_logits = shape_logits(net(x_init))
            if axis_mask is not None:
                if all_binary:
                    # just resample the ones which aren't fixed
                    # need a categorical since logits parameterize a softmax over two categories
                    b = torch.distributions.Categorical(logits=b_logits[:, axis_mask]).sample().float()
                else:
                    raise NotImplementedError
            else:
                if all_binary:
                    b = torch.distributions.Categorical(logits=b_logits).sample().float()
                else:
                    raise NotImplementedError

        return b, sampler._ar, sampler._lr, sampler._lf, sampler._la, sampler._mt, sampler._pt, sampler._hops

    def sample_x_b(net, x_init, b_init, steps=args.k,
                   gibbs_steps=args.gibbs_steps, gibbs_k_steps=args.gibbs_k_steps, gibbs_n_steps=args.gibbs_n_steps,
                   verbose=False, fix_label_axis=None, update_buffer=True,
                   new_replay_buffer=None, new_y_replay_buffer=None, return_steps=0, steps_batch_ind=0,
                   temp=False, anneal=False, truncated_bp=False, full_bp=False, return_num_added=False,
                   transform_every=None, marginalize_free_b=False):
        assert (new_replay_buffer is None) == (new_y_replay_buffer is None)
        assert not (anneal and temp)
        if new_replay_buffer is None:
            new_replay_buffer = replay_buffer
        if new_y_replay_buffer is None:
            new_y_replay_buffer = y_replay_buffer
        assert (new_replay_buffer is None) == (new_y_replay_buffer is None)
        assert implies(update_buffer and (args.yd_buffer is None), len(new_replay_buffer) > 0)
        x, b = x_init, b_init
        assert implies(fix_label_axis is not None, label_shape != (-1,))

        fix_label_axis = format_fix_label_axis(fix_label_axis)

        if label_dim == 1:
            # need to expand dims if we do conditional sampling first, one-hot already has 2 dims
            b = b[:, None]

        if anneal:
            betas = torch.linspace(0., 1., gibbs_steps * gibbs_k_steps)
        else:
            betas = None

        if temp:
            start_, end_ = args.plot_temp_sigma_start, args.sigma
            assert start_ > end_, "Untempered should have larger noise!"
            sigmas = torch.linspace(start_, end_, gibbs_steps * gibbs_k_steps)
        else:
            sigmas = None

        a_s = np.zeros(gibbs_steps)
        lrs = np.zeros(gibbs_steps)
        lfs = np.zeros(gibbs_steps)
        l_s = np.zeros(gibbs_steps)
        m_s = np.zeros(gibbs_steps)
        p_s = np.zeros(gibbs_steps)
        h_s = np.zeros(gibbs_steps)

        if args.interleave:
            if args.sampling == "pcd":
                (x, b), buffer_inds = init_sampling(new_replay_buffer, x,
                                                    y=b, y_replay_buffer=new_y_replay_buffer)
                if args.clamp_samples:
                    x = clamp_x(x)
            else:
                buffer_inds = None
            b, b_all, b_fixed, axis_mask = init_sample_b(b.squeeze(), fix_label_axis)
        else:
            buffer_inds = None
            b_all, b_fixed, axis_mask = None, None, None
        x_steps = []
        x_kl = None
        num_added = None

        for i in range(gibbs_steps):
            if args.interleave:
                if args.first_gibbs == "dis":
                    for _ in range(gibbs_n_steps):
                        b, *_info = step_sample_b(net, x, b.squeeze()[:, None],
                                                  b_all=b_all, b_fixed=b_fixed, axis_mask=axis_mask)
                        a_s[i], lrs[i], lfs[i], l_s[i], m_s[i], p_s[i], h_s[i] = _info

                    for k in range(gibbs_k_steps):
                        if return_steps > 0 and (i * gibbs_k_steps + k) % return_steps == 0:
                            if steps_batch_ind is None:
                                x_steps.append(x.clone().detach()[None])
                            else:
                                x_steps.append(x.clone().detach()[steps_batch_ind][None])

                        if transform_every is not None and (i * gibbs_k_steps + k) % transform_every == 0 \
                                and (i * gibbs_k_steps + k) != 0:
                            # don't transform at first iteration since buffer sampling does that already
                            x = transform(x)

                        if fix_label_axis is not None:
                            if unif_label_shape is None and not all_binary:
                                b_all[:, axis_mask] = b.squeeze()
                                b_all[:, ~axis_mask] = b_fixed.squeeze()
                            else:
                                b_all[:, axis_mask] = b
                                b_all[:, ~axis_mask] = b_fixed
                            b = b_all

                        if args.model == "joint":
                            if label_dim > 1:
                                if temp:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x,
                                                      sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    net_ = AISModel(lambda x_: energy(net, x_, b.squeeze()), normal_x_init_dist)
                                    x = step_sampling(lambda x_: net_(x_, betas[i * gibbs_k_steps + k]), x)
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x, bp=True)
                                else:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x)
                            else:
                                if temp:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x,
                                                      sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    net_ = AISModel(lambda x_: energy(net, x_, b), normal_x_init_dist)
                                    x = step_sampling(lambda x_: net_(x_, betas[i * gibbs_k_steps + k]), x)
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(lambda x_: energy(net, x_, b), x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x, bp=True)
                                else:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x)
                        else:
                            if all_binary:
                                if marginalize_free_b:
                                    def cond_energy_x_(x_):
                                        logits_neg_ = net(x_)
                                        fixed_log_probs = logit_log_prob_ind_subset(shape_logits(logits_neg_),
                                                                                    b.long(),
                                                                                    list(fix_label_axis.cpu().numpy()))
                                        marginalized_log_probs = logit_logsumexp(logits_neg_)[:, axis_mask]
                                        return fixed_log_probs.sum(-1) + marginalized_log_probs.sum(-1)
                                else:
                                    def cond_energy_x_(x_):
                                        logits_neg_ = net(x_)
                                        # index the logits according to the sampled classes
                                        return logit_log_prob_ind(shape_logits(logits_neg_), b.long()).sum(-1)

                                if temp:
                                    x = step_sampling(cond_energy_x_, x, sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    raise NotImplementedError
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(cond_energy_x_, x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    raise NotImplementedError
                                else:
                                    x = step_sampling(cond_energy_x_, x)
                            else:
                                raise NotImplementedError

                        if args.clamp_samples:
                            x = clamp_x(x)

                        if fix_label_axis is not None:
                            b = b_all[:, axis_mask]
                else:
                    for k in range(gibbs_k_steps):
                        if return_steps > 0 and (i * gibbs_k_steps + k) % return_steps == 0:
                            if steps_batch_ind is None:
                                x_steps.append(x.clone().detach()[None])
                            else:
                                x_steps.append(x.clone().detach()[steps_batch_ind][None])

                        if transform_every is not None and (i * gibbs_k_steps + k) % transform_every == 0 \
                                and (i * gibbs_k_steps + k) != 0:
                            # don't transform at first iteration since buffer sampling does that already
                            x = transform(x)

                        if fix_label_axis is not None:
                            if unif_label_shape is None and not all_binary:
                                b_all[:, axis_mask] = b.squeeze()
                                b_all[:, ~axis_mask] = b_fixed.squeeze()
                            else:
                                b_all[:, axis_mask] = b
                                b_all[:, ~axis_mask] = b_fixed
                            b = b_all

                        if args.model == "joint":
                            if label_dim > 1:
                                if temp:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x,
                                                      sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    net_ = AISModel(lambda x_: energy(net, x_, b.squeeze()), normal_x_init_dist)
                                    x = step_sampling(lambda x_: net_(x_, betas[i * gibbs_k_steps + k]), x)
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x, bp=True)
                                else:
                                    x = step_sampling(lambda x_: energy(net, x_, b.squeeze()), x)
                            else:
                                if temp:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x,
                                                      sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    net_ = AISModel(lambda x_: energy(net, x_, b), normal_x_init_dist)
                                    x = step_sampling(lambda x_: net_(x_, betas[i * gibbs_k_steps + k]), x)
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(lambda x_: energy(net, x_, b), x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x, bp=True)
                                else:
                                    x = step_sampling(lambda x_: energy(net, x_, b), x)
                        else:
                            if all_binary:
                                if marginalize_free_b:
                                    def cond_energy_x_(x_):
                                        logits_neg_ = net(x_)
                                        fixed_log_probs = logit_log_prob_ind_subset(shape_logits(logits_neg_),
                                                                                    b.long(),
                                                                                    list(fix_label_axis.cpu().numpy()))
                                        marginalized_log_probs = logit_logsumexp(logits_neg_)[:, axis_mask]
                                        return fixed_log_probs.sum(-1) + marginalized_log_probs.sum(-1)
                                else:
                                    def cond_energy_x_(x_):
                                        logits_neg_ = net(x_)
                                        # index the logits according to the sampled classes
                                        return logit_log_prob_ind(shape_logits(logits_neg_), b.long()).sum(-1)

                                if temp:
                                    x = step_sampling(cond_energy_x_, x, sigmas[i * gibbs_k_steps + k])
                                elif anneal:
                                    raise NotImplementedError
                                elif truncated_bp and i == gibbs_steps - 1 and k == gibbs_k_steps - 1:
                                    x_kl = step_sampling(cond_energy_x_, x, bp=True)
                                    if args.clamp_samples:
                                        x_kl = clamp_x(x_kl)
                                    x = x_kl.detach()
                                elif full_bp:
                                    raise NotImplementedError
                                else:
                                    x = step_sampling(cond_energy_x_, x)
                            else:
                                raise NotImplementedError

                        if args.clamp_samples:
                            x = clamp_x(x)

                        if fix_label_axis is not None:
                            b = b_all[:, axis_mask]

                    for _ in range(gibbs_n_steps):
                        b, *_info = step_sample_b(net, x, b.squeeze()[:, None],
                                                  b_all=b_all, b_fixed=b_fixed, axis_mask=axis_mask)
                        a_s[i], lrs[i], lfs[i], l_s[i], m_s[i], p_s[i], h_s[i] = _info
            else:
                if args.first_gibbs == "dis":
                    # sample discrete
                    if verbose:
                        b, *_info = sample_b(net, x, b.squeeze(), verbose=verbose, fix_label_axis=fix_label_axis,
                                             steps=gibbs_n_steps)
                        a_s[i], lrs[i], lfs[i], l_s[i], m_s[i], p_s[i], h_s[i] = _info
                    else:
                        b = sample_b(net, x, b.squeeze(), verbose=verbose, fix_label_axis=fix_label_axis,
                                     steps=gibbs_n_steps)

                    # sample cts
                    x = uncond_sample_x(lambda x_: energy(net, x_, b), x, steps=steps,
                                        update_buffer=update_buffer,
                                        new_replay_buffer=new_replay_buffer, new_y_replay_buffer=new_y_replay_buffer,
                                        return_steps=return_steps, steps_batch_ind=steps_batch_ind, temp=temp,
                                        anneal=anneal, truncated_bp=truncated_bp, full_bp=full_bp,
                                        return_num_added=return_num_added, transform_every=transform_every)

                    if return_num_added:
                        *x, num_added = x

                    if return_steps > 0:
                        x, x_steps = x

                else:
                    # sample cts
                    x = uncond_sample_x(lambda x_: energy(net, x_, b), x, steps=steps,
                                        update_buffer=update_buffer,
                                        new_replay_buffer=new_replay_buffer, new_y_replay_buffer=new_y_replay_buffer,
                                        return_steps=return_steps, steps_batch_ind=steps_batch_ind, temp=temp,
                                        anneal=anneal, truncated_bp=truncated_bp, full_bp=full_bp,
                                        return_num_added=return_num_added, transform_every=transform_every)

                    if return_num_added:
                        *x, num_added = x

                    if return_steps > 0:
                        x, x_steps = x

                    # sample discrete
                    if verbose:
                        b, *_info = sample_b(net, x, b.squeeze(), verbose=verbose, fix_label_axis=fix_label_axis,
                                             steps=gibbs_n_steps)
                        a_s[i], lrs[i], lfs[i], l_s[i], m_s[i], p_s[i], h_s[i] = _info
                    else:
                        b = sample_b(net, x, b.squeeze(), verbose=verbose, fix_label_axis=fix_label_axis,
                                     steps=gibbs_n_steps)

        if args.interleave:
            if fix_label_axis is not None:
                b_all[:, axis_mask] = b
                b_all[:, ~axis_mask] = b_fixed
                b = b_all

            if label_dim > 1:
                b = b.squeeze()

            if label_shape != (-1,) and unif_label_shape is not None:
                b = b.view(b.shape[0], label_dim)

            if args.sampling == "pcd":
                (x, b), num_added = set_sampling(x, new_replay_buffer, new_y_replay_buffer, buffer_inds,
                                                 update_buffer=update_buffer, y_k=label(b))

            if return_steps > 0:
                x_steps = torch.cat(x_steps, dim=0)

        if return_steps > 0 or truncated_bp or full_bp or return_num_added:
            x = (x,)
            if return_steps > 0:
                x += (x_steps,)
            if truncated_bp or full_bp:
                x += (x_kl,)
            if return_num_added:
                x += (num_added,)

        if verbose:
            return x, b, \
                   np.mean(a_s), np.mean(lrs), np.mean(lfs), np.mean(l_s), np.mean(m_s), np.mean(p_s), np.mean(h_s)
        else:
            return x, b

    def sample_x(net, x_init, b_init, steps=args.k, anneal=False, temp=False, update_buffer=True,
                 new_replay_buffer=None, new_y_replay_buffer=None, y=None,
                 return_steps=0, steps_batch_ind=0, return_num_added=False):

        if args.model == "joint":
            net_ = lambda x: energy(net, x, b_init)
        else:
            if all_binary:
                def cond_energy_x_(x_):
                    logits_neg_ = net(x_)
                    return logit_log_prob_ind(shape_logits(logits_neg_), b_init.long()).sum(-1)
                net_ = cond_energy_x_
            else:
                raise NotImplementedError

        return uncond_sample_x(net=net_,
                               x_init=x_init,
                               steps=steps,
                               anneal=anneal,
                               temp=temp,
                               update_buffer=update_buffer,
                               new_replay_buffer=new_replay_buffer,
                               new_y_replay_buffer=new_y_replay_buffer,
                               y=y,
                               return_steps=return_steps,
                               steps_batch_ind=steps_batch_ind,
                               return_num_added=return_num_added)

    def uncond_sample_x(net, x_init, steps=args.k, anneal=False, temp=False, truncated_bp=False, full_bp=False,
                        update_buffer=True, new_replay_buffer=None, new_y_replay_buffer=None, y=None,
                        return_steps=0, steps_batch_ind=0, return_num_added=False, transform_every=None):
        assert (new_replay_buffer is None) == (new_y_replay_buffer is None)
        assert not (anneal and temp)
        assert implies(truncated_bp or full_bp, not (anneal or temp))
        if new_replay_buffer is None:
            new_replay_buffer = replay_buffer
        if new_y_replay_buffer is None and args.model != "poj" and args.mode != "uncond":
            new_y_replay_buffer = y_replay_buffer
        assert implies(update_buffer and (args.yd_buffer is None), len(new_replay_buffer) > 0)

        betas = torch.linspace(0., 1., steps)
        x_kl = None
        if anneal:
            net = AISModel(net, normal_x_init_dist)

        if temp:
            start_, end_ = args.plot_temp_sigma_start, args.sigma
            assert start_ > end_, "Untempered should have larger noise!"
            sigmas = torch.linspace(start_, end_, steps)
        else:
            sigmas = None

        if args.sampling == "pcd":
            x_k, buffer_inds = init_sampling(new_replay_buffer, x_init, y=y, y_replay_buffer=new_y_replay_buffer)

            if args.clamp_samples:
                x_k = clamp_x(x_k)

            save_steps = []

            for k in range(steps):
                if return_steps > 0 and k % return_steps == 0:
                    if steps_batch_ind is None:
                        save_steps.append(x_k.clone().detach()[None])
                    else:
                        save_steps.append(x_k.clone().detach()[steps_batch_ind][None])
                if anneal:
                    x_k = step_sampling(lambda x: net(x, betas[k]), x_k)
                elif temp:
                    x_k = step_sampling(net, x_k, sigmas[k])
                else:
                    if truncated_bp and k == steps - 1:
                        x_kl = step_sampling(net, x_k, bp=True)
                        if args.clamp_samples:
                            x_kl = clamp_x(x_kl)
                        x_k = x_kl.detach()
                    elif full_bp:
                        x_k = step_sampling(net, x_k, bp=True)
                    else:
                        x_k = step_sampling(net, x_k)

                if transform is not None and transform_every is not None \
                        and k != 0 and (k + 1) % transform_every == 0:
                    # don't transform at k = 0 since buffer sampling does that already
                    x_k = transform(x_k)

            if full_bp:
                x_kl = x_k

            final_samples, num_added = set_sampling(x_k, new_replay_buffer, new_y_replay_buffer, buffer_inds,
                                                    update_buffer=update_buffer, y=y)

            if return_steps > 0 or truncated_bp or return_num_added:
                return_tpl = (final_samples, )

                if return_steps > 0:
                    save_steps = torch.cat(save_steps, dim=0)
                    return_tpl += (save_steps,)

                if truncated_bp:
                    return_tpl += (x_kl,)

                if return_num_added:
                    return_tpl += (num_added,)

                return return_tpl
            else:
                return final_samples
        else:
            if return_steps > 0 or truncated_bp or transform_every is not None:
                raise NotImplementedError
            return short_run_mcmc(lambda x: net(x).squeeze(), x_init, steps, args.sigma, step_size=args.step_size)

    def sample_eval_b(net, x, num_samples=1, return_individual=False):
        x_r = x.repeat_interleave(num_samples, 0)
        b_r = sample_b(net, x_r, init_b_random(x_r.size(0)), steps=args.test_n_steps).squeeze()
        if label_dim > 1:
            b_r = b_r.view(-1, num_samples, label_dim).permute(0, 2, 1)
        else:
            b_r = b_r.view(-1, num_samples)

        if return_individual:
            return b_r.mean(-1), b_r
        else:
            return b_r.mean(-1)

    def format_lbl(bs=None, b=None, cls=None, device=None):
        """
        Returns the formatted label.
        """
        assert xor(bs is None, b is None)
        assert implies(bs is not None, cls is not None)

        if b is None:
            b = torch.zeros(bs)
            if device is not None:
                b = b.to(device)

        if cls is not None:
            b += cls

        if label_dim > 1:
            assert label_shape == (-1,)
            return one_hot(b.long())
        else:
            return b[:, None]

    def format_lbl_cond(fix_label_axis, b, cls):
        """
        Given b.shape = (batch_size, sum(label_shape))
        Return b.shape = (batch_size, sum(label_shape)) where b is set to cls at label_axis.
        """
        if all_binary:
            new_b = b.clone()
            new_b[:, fix_label_axis] = cls
            return new_b
        else:
            assert fix_label_axis < len(label_shape)
            b_struct = convert_struct_onehot(label_shape, unif_label_shape, b)
            b_struct[:, fix_label_axis] = cls

            return one_hot(b_struct)

    def one_hot(y):
        """
        Convert vector of labels into one-hot.
        Expects y.shape = (batch_size, ) or y.shape = (batch_size, len(label_shape)).
        Returns y.shape = (batch_size, label_dim) a one-hot vecotr (multi one-hot if structured)
        """
        if all_binary:
            return y  # nothing to be done!
        elif label_shape != (-1, ):
            return get_onehots(label_shape, unif_label_shape, y)
        else:
            # TODO: replace with torch functional
            return get_one_hot(y, label_dim)

    def label(y, check_shape=True):
        """
        Convert vector of probabilities over labels into individual label numbers.
        Expects y.shape = (batch_size, num_labels)
        Returns y.shape = (batch_size, )
        """
        if all_binary:
            return (y > .5).float()
        elif label_shape != (-1,):
            return convert_struct_onehot(label_shape, unif_label_shape, y)
        else:
            if label_dim == 1 and len(y.shape) == 1:
                return (y > .5).float()
            if check_shape:
                assert len(y.shape) == 2
            if y.shape[1] == 1:
                return y
            else:
                return y.max(1).indices

    def energy_eval_b(net, x, y, return_pred=False, return_score=False, batched=False, metric="acc", individual=False,
                      detach_and_cpu=False, check_shapes=False):

        if args.model == "poj":
            pred_logits = net(x)
            if check_shapes:
                assert pred_logits.shape[0] == x.shape[0]
                assert pred_logits.shape[1] == label_dim * 2
            lp = onehot_poj_logits(pred_logits)
        else:
            pred_logits = None
            lp = []
            if label_shape != (-1,):
                assert not batched
                cls_lbls = []
                if all_binary:
                    _iterator_ = product(*(range(2) for _ in range(label_dim)))
                else:
                    _iterator_ = product(*(range(i) for i in label_shape))
                for label_combo in _iterator_:
                    cls_lbl = one_hot(torch.tensor(label_combo).to(args.device)[None])
                    cls_lbls.append(cls_lbl[:, None].view(-1, label_dim))
                    lbl = cls_lbl.repeat_interleave(x.shape[0], 0)
                    lp.append(energy(net, x, lbl)[:, None])

                lp = torch.cat(lp, 1)
                cls_lbls = torch.cat(cls_lbls, 0)
                lp = cls_lbls[lp.max(-1).indices]

            else:
                if batched:
                    for cls in range(all_labels + (label_dim == 1)):
                        lp.append(format_lbl(bs=y.shape[0], cls=cls, device=args.device))

                    lp = torch.cat(lp, dim=0)

                    x_r = x.repeat((all_labels, 1))
                    lp = energy(net, x_r, lp)
                    lp = lp.view(all_labels, x.shape[0]).permute(1, 0)

                else:
                    for cls in range(all_labels + (label_dim == 1)):
                        lp.append(energy(net, x, format_lbl(bs=y.shape[0], cls=cls, device=args.device))[:, None])

                    lp = torch.cat(lp, 1)

        if metric == "acc":
            result = accuracy_b(lp, y)
        elif metric == "f1":
            result = f1_b(lp, y, individual=individual)
        elif metric == "ece":
            result = ece_b(pred_logits, y, individual=individual)
        elif metric == "ap":
            result = ap_b(pred_logits, y, individual=individual)
        elif metric == "auroc":
            result = auroc_b(pred_logits, y, individual=individual)
        else:
            raise ValueError(f"Unrecognized metric {metric}")

        if individual:
            result = torch.tensor(result)

        if detach_and_cpu:
            result = result.detach().cpu()

        if return_pred or return_score:
            return_tpl = (result,)
            if return_pred:
                if detach_and_cpu:
                    lp = lp.detach().cpu()
                return_tpl += (lp,)
            if return_score:
                if detach_and_cpu:
                    pred_logits = pred_logits.detach().cpu()
                return_tpl += (pred_logits,)
            return return_tpl
        else:
            return result

    def cond_from_joint(logit_slice):
        """
        logit_slice.shape = (batch_size, num_labels, 2)
        return: (batch_size, num_labels)
        """
        # TODO: just index all (1, 0, 0), (0, 1, 0) etc.
        # TODO: marginalize out subcat
        log_prob = logit_log_prob(logit_slice, logit_slice.logsumexp(-1))
        return log_prob[..., 1] + torch.matmul(log_prob.logsumexp(-1), 1 - torch.eye(log_prob.shape[1]))

    def logits_ind_combos(shaped_logits, combos, include_marginals=False):
        """
        Index the logits for each of the combos.
        """
        combo_preds = torch.zeros(shaped_logits.shape[0], len(combos), device=shaped_logits.device)
        for combo_ind, combo in enumerate(combos):
            combo_preds[:, combo_ind] = index_by_combo_name(shaped_logits, dset_label_info, combo, include_marginals)
        return combo_preds

    def shape_logits(raw_logits):
        """
        Shape logits for logsumexp.
        """
        if unif_label_shape:
            raw_logits = raw_logits.view(raw_logits.shape[0], len(label_shape), unif_label_shape)
        elif all_binary:
            if args.uncond_poj:
                raw_logits = raw_logits.view(raw_logits.shape[0], len(dset_label_info), 2)
            else:
                raw_logits = raw_logits.view(raw_logits.shape[0], label_dim, 2)
        else:
            # clone logits since we need to write to them
            raw_logits = raw_logits[:, None].repeat(1, 2, 1)
            broadcast_poj_logit_mask = poj_logit_mask.expand(raw_logits.shape[0], -1, -1)
            # set logits to -inf (0 out probability) using broadc_poj_logit_mask or its negation is arbitrary
            raw_logits[~broadcast_poj_logit_mask] = -float('inf')
        return raw_logits

    def logit_logsumexp(raw_logits):
        """
        Logsumexp of raw logits.
        """
        return shape_logits(raw_logits).logsumexp(-1)

    def logit_log_prob(raw_logits, shaped_logits_logsumexp):
        """
        Converts raw logits to log probabilities.
        """
        return shape_logits(raw_logits) - shaped_logits_logsumexp[..., None]

    def logit_log_prob_ind(log_prob, y):
        return torch.gather(log_prob, index=y[..., None], dim=2).squeeze()

    def logit_log_prob_ind_subset(log_prob, y, subset_axes):
        """
        Index log probabilities/logits at a subset of the labels. 
        The subset of labels is the same along the batch axis, and is given by subset_axes.
        """
        if isinstance(subset_axes, int):
            subset_axes = [subset_axes]
        else:
            assert isinstance(subset_axes, list)
        log_prob = log_prob[:, subset_axes]
        y = y[:, subset_axes]
        subset_log_prob = logit_log_prob_ind(log_prob, y.long())
        if len(subset_log_prob.shape) == 1:
            # undo squeeze by logit_log_prob_ind
            subset_log_prob = subset_log_prob[:, None]
        return subset_log_prob

    def logit_log_prob_ind_missing(log_prob, y, missing_mask, in_place=True):
        """
        logit_log_prob_ind with support for missing values, where it will return 0.
        Sharp edges: Modifies log_prob in place!
        """
        if not in_place:
            log_prob = log_prob.clone()

        if missing_mask.any():
            # set all logits for missing labels to 0
            log_prob = log_prob * (~missing_mask[..., None])
            # set all missing labels to 0 (setting to 0 is abitrary, all the missing logits have been zeroed)
            y = y * ~missing_mask

        return logit_log_prob_ind(log_prob, y)

    def logit_log_p_y_x(raw_logits, y, missing_mask, shaped_logits_logsumexp):
        """
        Compute log p(y|x) from raw unshaped logits and labels.
        """
        # subtract logsumexp from logits to get log prob
        log_prob = logit_log_prob(raw_logits, shaped_logits_logsumexp)

        if unif_label_shape is None and not all_binary:
            # convert y index to flattened version
            y = y + diff_label_axes[:-1][None]

        ce_per_example = logit_log_prob_ind_missing(log_prob, y, missing_mask)
        if args.weighted_ce:
            weights_per_example = \
                torch.gather(cls_weights.expand(y.shape[0], *cls_weights.shape), index=y[..., None], dim=2).squeeze()
            ce_per_example = ce_per_example * weights_per_example
        ce_per_example = ce_per_example.sum(-1)

        # only average over examples with at least one observed variable
        ce_per_example = ce_per_example[(~missing_mask).any(-1)]

        return ce_per_example.mean()

    def onehot_poj_logits(raw_logits):
        """
        Given raw logits, get onehot predictions for the labels.
        """
        # take argmax over logit dim
        argmax_logits = shape_logits(raw_logits).max(-1).indices
        if unif_label_shape is None and not all_binary:
            # subtract cumulative sum
            argmax_logits = argmax_logits - diff_label_axes[:-1][None]
        return one_hot(argmax_logits)

    def label_onehot_poj(onehot_poj):
        """
        Label one-hot vector for poj model.
        """
        onehot_poj = label(onehot_poj)
        if unif_label_shape is None and not all_binary:
            # shift labels to index one-hot vector
            onehot_poj = onehot_poj + diff_label_axes[:-1]
        return onehot_poj

    def filter_post_hoc(net, x, y, axes):
        if args.plot_cond_filter > 1:
            filter_logits = shape_logits(net(x))
            filter_scores = logit_log_prob_ind_subset(filter_logits, y, axes).sum(-1)
            return x[filter_scores.argsort(descending=True)[:args.batch_size]]
        else:
            return x

    init_x, init_y = data_batch()

    def check_cond_cls_utzappos(label_axis_, cond_cls_):
        if args.utzappos_drop_infreq is None:
            # TODO: reimplement by directly indexing dset_label_info
            # men, women, boots, sandals, shoes, heels, over the knee, oxfords
            # return (label_axis_ in (87, 90, 100, 101, 103, 105, 110)) and cond_cls_ == 1
            raise NotImplementedError
        elif args.utzappos_drop_infreq == 0.1:
            raise NotImplementedError
        elif args.utzappos_drop_infreq == 10:
            if args.eval is None:
                _filter_classes = ["Gender_Women", "Material_Leather"]
            else:
                _filter_classes = ["Gender_Women", "Gender_Men", "Gender_Girls",
                                   "Material_Leather", "Material_Synthetic", "Material_Rubber", "Material_Suede",
                                   "Closure_Lace up", "Closure_Slip-On",
                                   "Category_Boots", "Category_Sandals",
                                   "SubCategory_Flat", "SubCategory_Heels", "SubCategory_Sneakers and Athletic Shoes"]
        elif args.utzappos_drop_infreq == 20:
            raise NotImplementedError
        else:
            raise ValueError(f"Unconfigured drop infreq {args.utzappos_drop_infreq}")

        return cond_cls_ == 1 and label_axis_ in map(lambda _cls_name: dset_label_info[_cls_name][0], _filter_classes)

    def check_cond_cls_mul_utzappos(cond_cls_0, cond_cls_1):
        if args.utzappos_drop_infreq is None:
            # (men, boots), (women, boots), (leather, heels), (canvas, lace up), (canvas, sneakers)
            # TODO: reimplement by directly indexing dset_label_info
            # return (cond_cls_0, cond_cls_1) in ((0, 86), (1, 86), (4, 100), (9, 67,), (9, 110))
            raise NotImplementedError
        elif args.utzappos_drop_infreq == 0.1:
            raise NotImplementedError
        elif args.utzappos_drop_infreq == 10:
            if args.eval is None:
                # only plot a few classes during training
                _filter_classes = [("Gender_Men", "Category_Boots"),
                                   ("Gender_Women", "Category_Boots")]
            else:
                # plot more classes if we're evaluating a model
                _filter_classes = [("Gender_Men", "Category_Boots"),
                                   ("Gender_Women", "Category_Boots"),
                                   ("Gender_Women", "SubCategory_Heels"),
                                   ("Gender_Women", "SubCategory_Flat"),
                                   ("Gender_Men", "Category_Sandals"),
                                   ("Gender_Women", "Category_Sandals"),
                                   ("Gender_Women", "Closure_Slip-On"),
                                   ("Gender_Men", "Closure_Lace up"),
                                   ("Material_Leather", "SubCategory_Heels"),
                                   ("Material_Leather", "Category_Boots"),
                                   # ("Material_Rubber", "Category_Boots"),
                                   # ("Closure_Hook and Loop", "Category_Boots"),
                                   ("Closure_Lace up", "Category_Boots"),
                                   ("SubCategory_Sneakers and Athletic Shoes", "Closure_Lace up"),
                                   ("Gender_Men", "SubCategory_Sneakers and Athletic Shoes")]
        elif args.utzappos_drop_infreq == 20:
            raise NotImplementedError
        else:
            raise ValueError(f"Unconfigured drop infreq {args.utzappos_drop_infreq}")

        return (cond_cls_0, cond_cls_1) in map(lambda _cls_names:
                                               tuple(map(lambda _cls_name: dset_label_info[_cls_name][0], _cls_names)),
                                               _filter_classes)

    def check_cond_cls_celeba(label_axis_, cond_cls_):
        if args.dset_split_type == "zero_shot":
            _pos_filter_classes = _neg_filter_classes = CELEBA_ZERO_SHOT_ATTRIBUTES
        else:
            if args.eval_cond_acc:
                _pos_filter_classes = _neg_filter_classes = []
            elif args.eval is None:
                _pos_filter_classes = ["Male", "Smiling", "Black_Hair"]
                _neg_filter_classes = ["Male", "No_Beard"]
            else:
                _pos_filter_classes = ["Male", "Smiling", "Black_Hair", "Young", "Wearing_Lipstick", "Bags_Under_Eyes",
                                       "Bangs", "Mouth_Slightly_Open", "High_Cheekbones", "Attractive",
                                       "Arched_Eyebrows"]
                _neg_filter_classes = ["Male", "Smiling", "No_Beard", "Young", "Wearing_Lipstick",
                                       "Mouth_Slightly_Open"]

        _in_pos_class = cond_cls_ == 1 and label_axis_ in map(lambda _cls_name: dset_label_info[_cls_name][0],
                                                              _pos_filter_classes)
        _in_neg_class = cond_cls_ == 0 and label_axis_ in map(lambda _cls_name: dset_label_info[_cls_name][0],
                                                              _neg_filter_classes)
        return _in_pos_class or _in_neg_class

    def check_cond_cls_mul_celeba(cond_cls_0, cond_cls_1):
        if args.eval is None:
            if args.dset_split_type == "zero_shot":
                _filter_classes = []  # just use the "custom option" to do all plotting
            else:
                _filter_classes = [("Male", "No_Beard"),
                                   ("Smiling", "No_Beard")]
        else:
            _filter_classes = [("Male", "No_Beard"),
                               ("Male", "Black_Hair"),
                               ("Smiling", "No_Beard"),
                               ("Male", "Smiling"),
                               ("Smiling", "Wearing_Lipstick"),
                               ("Smiling", "Mouth_Slightly_Open"),
                               ("Smiling", "Attractive"),
                               ("Smiling", "Young"),
                               ("Young", "Attractive")]

        return (cond_cls_0, cond_cls_1) in map(lambda _cls_names:
                                               tuple(map(lambda _cls_name: dset_label_info[_cls_name][0], _cls_names)),
                                               _filter_classes)

    def check_cond_cls_cub(label_axis_, cond_cls_):
        _filter_classes = ["has_belly_color::yellow", "has_belly_color::black",
                           "has_bill_shape::dagger", "has_bill_shape::cone",
                           "has_size::medium_(9_-_16_in)", "has_size::very_small_(3_-_5_in)"]

        return cond_cls_ == 1 and label_axis_ in map(lambda _cls_name: dset_label_info[_cls_name][0], _filter_classes)

    def check_cond_cls_mul_cub(cond_cls_0, cond_cls_1):
        _filter_classes = [("has_wing_color::black", "has_throat_color::white"),
                           ("has_wing_color::white", "has_throat_color::white"),
                           ("has_size::medium_(9_-_16_in)", "has_belly_color::black"),
                           ("has_size::very_small_(3_-_5_in)", "has_belly_color::yellow")]

        return (cond_cls_0, cond_cls_1) in map(lambda _cls_names:
                                               tuple(map(lambda _cls_name: dset_label_info[_cls_name][0], _cls_names)),
                                               _filter_classes)

    plot_samples(init_x, f"{args.save_dir}/init_samples.png")
    if label_dim != 0 and not args.zero_shot:
        if label_shape != (-1,):
            if all_binary:
                _iterator = zip(range(label_dim), (2 for _ in range(label_dim)))
            else:
                _iterator = enumerate(label_shape)
            for label_axis, label_axis_dim in _iterator:
                for cond_cls in range(label_axis_dim):
                    if args.data == "dsprites":
                        if cond_cls not in (0, 3, 6, 9, 12, 15):
                            continue
                    elif args.data == "utzappos_old":
                        if not (label_axis == 0 and cond_cls in (0, 1)):  # Boots, Sandals
                            continue
                    elif args.data == "utzappos":
                        if not check_cond_cls_utzappos(label_axis, cond_cls):
                            continue
                    elif args.data == "celeba":
                        if not check_cond_cls_celeba(label_axis, cond_cls):
                            continue
                    elif args.data == "cub":
                        if not check_cond_cls_cub(label_axis, cond_cls):
                            continue

                    _mask = init_y[:, label_axis] == cond_cls
                    plot_samples(init_x[_mask], f"{args.save_dir}/init_samples_{label_axis}_{cond_cls}.png")

            if all_binary:
                prod_cond_cls = product(*(range(2) for _ in range(label_dim)))
            else:
                prod_cond_cls = product(*(range(i) for i in label_shape))

            def next_cond_cls():
                label_combo = next(prod_cond_cls)
                cls_lbl = one_hot(torch.tensor(label_combo).to(args.device)[None])
                return cls_lbl.repeat_interleave(init_x.shape[0], 0)

            if args.data == "utzappos_old":
                label_combo_ = (2, 17, 272)  # Shoes/Sneakers and Athletic Shoes/Nike
                cls_lbl_ = one_hot(torch.tensor(label_combo_).to(args.device)[None])
                next_cond_cls_ = cls_lbl_.repeat_interleave(init_x.shape[0], 0)
                _disc_mask = (get_onehots(label_shape, unif_label_shape, init_y) ==
                              next_cond_cls_).view(init_y.shape[0], -1).all(1)
                cond_cls = 0  # idk what the number is, and it doesn't matter
                plot_samples(init_x[_disc_mask],
                             f"{args.save_dir}/init_samples{cond_cls}.png")
            elif args.data in ("utzappos", "celeba", "cub"):
                pass
            else:
                if all_binary:
                    _iterator = range(2 ** label_dim)
                else:
                    _iterator = range(all_labels + (label_dim == 1))
                for cond_cls in _iterator:
                    _disc_mask = (one_hot(init_y) == next_cond_cls()).view(init_y.shape[0], -1).all(1)
                    plot_samples(init_x[_disc_mask],
                                 f"{args.save_dir}/init_samples{cond_cls}.png")

            if "utzappos" in args.data or args.data in ("celeba", "cub") or "8gaussians_hierarch" in args.data:
                if all_binary:
                    if "utzappos" in args.data or args.data in ("celeba", "cub"):
                        _iterator = product(range(label_dim), range(label_dim))
                    else:
                        # in the case of toy data, just do the same thing if it weren't paramaterized as binary
                        _iterator = product(range(2), range(2))
                else:
                    _iterator = product(range(label_shape[0]), range(label_shape[1]))
                for cond_cls0, cond_cls1 in _iterator:
                    if args.data == "utzappos_old":
                        if (cond_cls0, cond_cls1) not in [(2, 11), (2, 12), (2, 13), (2, 15), (2, 17), (0, 1), (0, 2)]:
                            continue
                    elif args.data == "utzappos":
                        if not check_cond_cls_mul_utzappos(cond_cls0, cond_cls1):
                            continue
                    elif args.data == "celeba":
                        if not check_cond_cls_mul_celeba(cond_cls0, cond_cls1):
                            continue
                    elif args.data == "cub":
                        if not check_cond_cls_mul_cub(cond_cls0, cond_cls1):
                            continue

                    if all_binary and ("utzappos" in args.data or args.data in ("celeba", "cub")):
                        _mask = torch.logical_and(init_y[:, cond_cls0] == 1, init_y[:, cond_cls1] == 1)
                    else:
                        _mask = torch.logical_and(init_y[:, 0] == cond_cls0, init_y[:, 1] == cond_cls1)
                    plot_samples(init_x[_mask], f"{args.save_dir}/init_samples_mul_{cond_cls0}_{cond_cls1}.png")

            if args.eval is not None and args.data in ("celeba", "utzappos"):

                if args.dset_split_type == "zero_shot":
                    # use the whole test dataset, since we're plotting held out attribute combinations!
                    # TODO: does this work for the attribute combinations we're not holding out?
                    init_x_, init_y_ = zero_shot_tst_dset[:]
                else:
                    init_x_, init_y_ = init_x, init_y

                # convert keys from names to indices
                for celeba_cls_combo in custom_cls_combos:
                    celeba_cls_indices = [(dset_label_info[celeba_cls_name][0], celeba_cls_val)
                                          for celeba_cls_name, celeba_cls_val in celeba_cls_combo.items()]
                    _mask = reduce(torch.logical_and,
                                   map(partial(cond_attributes_from_labels, init_y_), celeba_cls_indices))

                    cls_descriptor_str = "_".join(map(str,
                                                      [val for pair in celeba_cls_indices for val in pair]))

                    plot_samples(init_x_[_mask][:args.batch_size],
                                 f"{args.save_dir}/init_samples_custom_{cls_descriptor_str}.png")
        else:
            for cond_cls in range(all_labels + (label_dim == 1)):
                plot_samples(init_x[init_y == cond_cls], f"{args.save_dir}/init_samples_{cond_cls}.png")

    normal_x_init_dist = torch.distributions.Normal(init_x.mean(0), init_x.std(0))
    logger(f"normal | "
           f"loc | "
           f"mean: {normal_x_init_dist.loc.mean():.2f} | "
           f"max: {normal_x_init_dist.loc.max():.2f} | "
           f"min: {normal_x_init_dist.loc.min():.2f} | "
           f"scale | "
           f"mean: {normal_x_init_dist.scale.mean():.2f} | "
           f"max: {normal_x_init_dist.scale.max():.2f} | "
           f"min: {normal_x_init_dist.scale.min():.2f} | ")
    if args.init_dist == "norm":
        x_init_dist = torch.distributions.Normal(init_x.mean(0), init_x.std(0))
        logger(f"normal | "
               f"loc | "
               f"mean: {x_init_dist.loc.mean():.2f} | "
               f"max: {x_init_dist.loc.max():.2f} | "
               f"min: {x_init_dist.loc.min():.2f} | "
               f"scale | "
               f"mean: {x_init_dist.scale.mean():.2f} | "
               f"max: {x_init_dist.scale.max():.2f} | "
               f"min: {x_init_dist.scale.min():.2f} | ")
    elif args.init_dist == "unif":
        if args.data == "mwa":
            _init_x_ones = torch.ones(init_x.shape[1:])
            x_init_dist = torch.distributions.Uniform(-_init_x_ones, _init_x_ones)
        # elif args.data == "celeba":
        #     _init_x_ones = torch.ones(init_x.shape[1:])
        #     x_init_dist = torch.distributions.Uniform(torch.zeros_like(_init_x_ones), _init_x_ones)
        else:
            x_init_dist = torch.distributions.Uniform(init_x.min(0).values, init_x.max(0).values)
        logger(f"unif | "
               f"low | "
               f"mean: {x_init_dist.low.mean():.2f} | "
               f"max: {x_init_dist.low.max():.2f} | "
               f"min: {x_init_dist.low.min():.2f} | "
               f"high | "
               f"mean: {x_init_dist.high.mean():.2f} | "
               f"max: {x_init_dist.high.max():.2f} | "
               f"min: {x_init_dist.high.min():.2f} | ")
    else:
        assert False

    if label_dim > 1:
        # multiclass
        if label_shape != (-1,):
            if all_binary:
                probs_ = init_y.float().mean(0)
                if args.unif_init_b:
                    b_init_dist = torch.distributions.Bernoulli(probs=torch.ones_like(probs_) * .5)
                else:
                    b_init_dist = torch.distributions.Bernoulli(probs=trn_dset_freqs.squeeze())
            else:
                assert args.unif_init_b
                if unif_label_shape is not None:
                    b_init_dist = torch.distributions.OneHotCategorical(probs=get_onehots(label_shape, unif_label_shape,
                                                                                          torch.ones_like(init_y[0])))
                else:
                    b_init_dists = [torch.distributions.OneHotCategorical(probs=torch.ones(label_axis_dim,
                                                                                           device=init_y.device))
                                    for label_axis_dim in label_shape]

                    class StructuredSampler:
                        def __init__(self, init_dists):
                            self.init_dists = init_dists

                        def sample(self, shape):
                            samples = [init_dist.sample(shape) for init_dist in self.init_dists]
                            return torch.cat(samples, dim=1)

                    b_init_dist = StructuredSampler(b_init_dists)
        else:
            if args.unif_init_b:
                assert False
                # b_init_dist = torch.distributions.OneHotCategorical(probs=torch.ones_like(init_y))
            else:
                b_init_dist = torch.distributions.OneHotCategorical(probs=init_y.bincount(minlength=label_dim).float())
    elif label_dim == 0:
        # dummy value
        b_init_dist = torch.distributions.Bernoulli(probs=torch.zeros(args.batch_size))
    else:
        b_init_dist = torch.distributions.Bernoulli(probs=init_y.float().mean())

    if args.weighted_ce:
        if args.data not in ("utzappos", "celeba", "cub"):
            raise NotImplementedError(f"Weighted cross-entropy is not implemented for {args.data}.")

        # dset_freqs has the frequency of label `1`
        cls_weights = torch.cat([1 / (1 - trn_dset_freqs), 1 / trn_dset_freqs], dim=1)
        # normalize so that class weights sum to 1
        cls_weights = cls_weights / cls_weights.sum(1, keepdim=True)
    else:
        cls_weights = None

    if args.transform:
        if args.data == "celeba":
            if args.celeba_all_transform:
                transforms = [kornia.augmentation.RandomResizedCrop((args.img_size, args.img_size), scale=(0.08, 1.0)),
                              kornia.augmentation.RandomHorizontalFlip(),
                              get_color_distortion(0.5),
                              GaussianBlur(kernel_size=5)]
            elif args.celeba_no_color_transform:
                transforms = [kornia.augmentation.RandomResizedCrop((args.img_size, args.img_size), scale=(0.08, 1.0)),
                              kornia.augmentation.RandomHorizontalFlip(),
                              GaussianBlur(kernel_size=5)]
            else:
                transforms = [kornia.augmentation.RandomHorizontalFlip(),
                              GaussianBlur(kernel_size=5)]
            transform = torchvision.transforms.Compose(transforms)
        elif args.data == "utzappos":
            if args.utzappos_blur_transform:
                transforms = []
            else:
                transforms = [kornia.augmentation.RandomResizedCrop((args.img_size, args.img_size), scale=(0.08, 1.0)),
                              get_color_distortion(0.5)]
            transforms.append(GaussianBlur(kernel_size=5))
            transform = torchvision.transforms.Compose(transforms)
        elif args.data == "cub":
            transforms = [kornia.augmentation.RandomResizedCrop((args.img_size, args.img_size), scale=(0.08, 1.0)),
                          kornia.augmentation.RandomHorizontalFlip(),
                          get_color_distortion(0.5),
                          GaussianBlur(kernel_size=5)]
            transform = torchvision.transforms.Compose(transforms)
        else:
            raise ValueError(f"Unrecognized dataset {args.data} for transforms.")
    else:
        transform = lambda x: x

    def init_replay_buffer():
        buffer_x_init = init_x_random(args.buffer_size, to_device=False)
        buffer_y_init = init_b_random(args.buffer_size, to_device=False, label_samples=True)
        if args.yd_buffer is not None:
            if args.yd_buffer == "replay":
                return ReplayBuffer(args.buffer_size, buffer_x_init), \
                       ReplayBuffer(args.buffer_size, buffer_y_init)
            elif args.yd_buffer == "reservoir":
                return ReservoirBuffer(args.buffer_size, buffer_x_init), \
                       ReservoirBuffer(args.buffer_size, buffer_y_init)
            else:
                raise ValueError(f"Unrecognized yd_buffer {args.yd_buffer}")
        else:
            return buffer_x_init, buffer_y_init

    if args.mode == "sup":
        replay_buffer, y_replay_buffer = [], []
    else:
        replay_buffer, y_replay_buffer = init_replay_buffer()
    init_sampling, step_sampling, set_sampling = get_sample_q(init_x_random=init_x_random,
                                                              init_b_random=init_b_random,
                                                              reinit_freq=args.reinit_freq,
                                                              step_size=args.step_size,
                                                              sigma=args.sigma,
                                                              device=args.device,
                                                              transform=transform,
                                                              one_hot_b=one_hot,
                                                              only_transform_buffer=args.only_transform_buffer)

    optim = torch.optim.Adam(logp_net.parameters(), args.lr, betas=(args.beta1, args.beta2))

    start_itr = 0
    duration = 0
    data_duration = 0
    log_num_added = None
    tst_accs = {}

    # appeasing the linter
    kl_loss = torch.tensor(0.).to(args.device)
    logp_y_x = torch.tensor(0.).to(args.device)
    logp_y_x_fake = torch.tensor(0.).to(args.device)
    bp_loss = torch.tensor(0.).to(args.device)
    log_p_xy = torch.tensor(0.).to(args.device)
    logp_fake = torch.tensor(0.).to(args.device)
    eval_logp_y_x = torch.tensor(0.).to(args.device)
    eval_logp_fake = torch.tensor(0.).to(args.device)
    eval_logp_y_x_fake = torch.tensor(0.).to(args.device)
    eval_kl_loss = torch.tensor(0.).to(args.device)
    if all_binary:
        fix_label_axis_mask = torch.arange(label_dim, device=args.device)
    else:
        fix_label_axis_mask = torch.arange(len(label_shape), device=args.device)
    x_neg_kl, b_neg_kl = None, None

    if args.model == "poj" and unif_label_shape is None and not all_binary and not args.mode == "uncond":
        fix_label_axes_ = torch.zeros(1, device=args.device, dtype=torch.int64)[:, None]

        def add_axis_mask(axis_dim_):
            return get_onehot_struct_mask(diff_label_axes, struct_mask, (fix_label_axes_ + axis_dim_))

        poj_logit_mask = torch.cat([add_axis_mask(i) for i in range(len(label_shape))], dim=1)
    else:
        poj_logit_mask = None

    try:
        if args.eval is None and args.resume is None:
            ckpt_path = args.ckpt_path
        elif args.eval is not None:
            logger(f"========= EVALUATION MODE =========")
            ckpt_path = args.eval
        elif args.resume is not None:
            if os.path.exists(args.ckpt_path):
                logger(f"========= OVERRIDING MANUAL RESUME, FOUND PREEMPTION CKPT =========")
                ckpt_path = args.ckpt_path
            else:
                logger(f"========= MANUAL RESUME =========")
                ckpt_path = args.resume
        else:
            raise ValueError("Undefined checkpoint option")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        start_itr = ckpt["itr"] + 1  # last completed iteration is saved
        logp_net.load_state_dict(ckpt["models"]["logp_net"])
        ema_logp_net.load_state_dict(ckpt["models"]["ema_logp_net"])
        optim.load_state_dict(ckpt["optimizer"]["logp_net_optimizer"])
        replay_buffer = ckpt["replay_buffer"]
        y_replay_buffer = ckpt["y_replay_buffer"]
        if "tst_accs" in ckpt:
            tst_accs = ckpt["tst_accs"]
        else:
            logger("============ WARNING: NO TST ACCS FOUND IN CKPT ============")

        logger(f"Loaded checkpoint from {start_itr} at {ckpt_path}")
    except IOError as error:
        if args.eval is None:
            logger(f"Could not load checkpoint {args.ckpt_path}")
        else:
            raise error(f"Could not load model for evaluation {args.eval}")

    def accuracy_b(b, y):
        pred = label(b)
        missing_weights = (y != -1)
        return ((y.float() == pred) * missing_weights).float().sum() / missing_weights.sum()

    def f1_b(b, y, **kwargs):
        pred = label(b)
        return f1_score_missing(y_true=y.float(), y_pred=pred.float(), **kwargs)

    def ece_b(b, y, **kwargs):
        # convert raw unnormalized logits to probabilities
        p_mean = logit_log_prob(b, logit_logsumexp(b)).exp().detach()

        return get_calibration(y=y, p_mean=p_mean, **kwargs)["ece"]

    def ece_plot_b(b, y, plot_itr, individual=False, micro=False):
        # convert raw unnormalized logits to probabilities
        p_mean = logit_log_prob(b, logit_logsumexp(b)).exp().detach()

        ece_info = get_calibration(y=y, p_mean=p_mean, individual=individual, micro=micro)
        if micro:
            _plot_ece_hist(args.save_dir, f"micro_{plot_itr}", ece_info["reliability_diag"], ece_info["ece"],
                           y.mean(), dset_label_info)
        else:
            for label_dim_ in range(y.shape[1]):
                _plot_ece_hist(args.save_dir, f"{label_dim_}_{plot_itr}",
                               tuple(el[:, label_dim_] for el in ece_info["reliability_diag"]),
                               ece_info["ece"][label_dim_], y[:, label_dim_].mean(), dset_label_info)

    def ap_b(b, y, individual=False, micro=False):
        # AP is invariant to scale, but converting from softmax to binary probability changes relative score
        b = logit_log_prob(b, logit_logsumexp(b)).exp()
        # index the probs at the positive class
        b = logit_log_prob_ind(b, torch.ones_like(y).long()).detach()

        return ap_score(y_true=y, y_pred=b, individual=individual, micro=micro)

    def auroc_b(b, y, individual=False, micro=False):
        # AUROC is invariant to scale, but converting from softmax to binary probability changes relative score
        b = logit_log_prob(b, logit_logsumexp(b)).exp()
        # index the probs at the positive class
        b = logit_log_prob_ind(b, torch.ones_like(y).long()).detach()

        return auroc_score(y_true=y, y_pred=b, individual=individual, micro=micro)

    def pr_curve_b(b, y, plot_itr, micro=False):
        # AP is invariant to scale, but converting from softmax to binary probability changes relative score
        b = logit_log_prob(b, logit_logsumexp(b)).exp()
        # index the probs at the positive class
        b = logit_log_prob_ind(b, torch.ones_like(y).long())

        assert b.shape == y.shape

        assert len(y.shape) == 2

        if micro:
            y = y.view(-1)
            b = b.view(-1)
            precision, recall, _ = precision_recall_curve(y_true=y.detach().cpu().numpy(),
                                                          probas_pred=b.detach().cpu().numpy())
            ap = average_precision_score(y_true=y.detach().cpu().numpy(),
                                         y_score=b.detach().cpu().numpy())
            _plot_pr_curve(args.save_dir, f"micro_{plot_itr}", precision, recall, ap, y.mean(), dset_label_info)
        else:
            y = y.detach().cpu().numpy()
            b = b.detach().cpu().numpy()
            for label_dim_ in range(y.shape[1]):
                precision, recall, _ = precision_recall_curve(y_true=y[:, label_dim_], probas_pred=b[:, label_dim_])
                ap = average_precision_score(y_true=y[:, label_dim_],
                                             y_score=b[:, label_dim_])
                _plot_pr_curve(args.save_dir, f"{label_dim_}_{plot_itr}", 
                               precision, recall, ap, y[:, label_dim_].mean(), dset_label_info)

    def roc_curve_b(b, y, plot_itr, micro=False):
        # AUROC is invariant to scale, but converting from softmax to binary probability changes relative score
        b = logit_log_prob(b, logit_logsumexp(b)).exp()
        # index the probs at the positive class
        b = logit_log_prob_ind(b, torch.ones_like(y).long())

        assert b.shape == y.shape

        assert len(y.shape) == 2

        if micro:
            y = y.view(-1)
            b = b.view(-1)
            fpr, tpr, _ = roc_curve(y_true=y.detach().cpu().numpy(), y_score=b.detach().cpu().numpy())
            auroc = roc_auc_score(y_true=y.detach().cpu().numpy(), y_score=b.detach().cpu().numpy())
            _plot_roc_curve(args.save_dir, f"micro_{plot_itr}", fpr, tpr, auroc, y.mean(), dset_label_info)
        else:
            y = y.detach().cpu().numpy()
            b = b.detach().cpu().numpy()
            for label_dim_ in range(y.shape[1]):
                fpr, tpr, _ = roc_curve(y_true=y[:, label_dim_], y_score=b[:, label_dim_])
                auroc = roc_auc_score(y_true=y[:, label_dim_], y_score=b[:, label_dim_])
                _plot_roc_curve(args.save_dir, f"{label_dim_}_{plot_itr}",
                                fpr, tpr, auroc, y[:, label_dim_].mean(), dset_label_info)

    eval_x, eval_y = data_batch()
    if args.zero_shot:
        tst_eval_x, tst_eval_y = data_val_batch()
    elif args.data == "mnist" or args.dsprites_test or args.full_test:
        tst_eval_x, tst_eval_y = data_test_batch()
    else:
        tst_eval_x, tst_eval_y = None, None

    if args.zero_shot:
        num_tst_batches = dset_lengths['val'] // args.batch_size + int(dset_lengths['val'] % args.batch_size != 0)
        num_zs_tst_batches = dset_lengths['tst'] // args.batch_size + int(dset_lengths['tst'] % args.batch_size != 0)
        logger(f"==== VAL LEN {dset_lengths['val']} BS {args.batch_size} NB {num_tst_batches} ====")
        logger(f"==== TEST LEN {dset_lengths['tst']} BS {args.batch_size} NB {num_zs_tst_batches} ====")
        # reset the iterator
        for _ in range(num_tst_batches - 1):
            data_val_batch()
    elif args.full_test:
        num_zs_tst_batches = -1
        if args.data == "mnist":
            assert 10000 % args.batch_size == 0
            num_tst_batches = 10000 // args.batch_size
        elif args.data == "utzappos":
            num_tst_batches = UTZAPPOS_TEST_LEN // args.batch_size + int(UTZAPPOS_TEST_LEN % args.batch_size != 0)
            logger(f"==== TEST LEN {UTZAPPOS_TEST_LEN} BS {args.batch_size} NB {num_tst_batches} ====")
        elif args.data == "celeba":
            if args.dset_split_type == "zero_shot":
                num_tst_batches = tst_dset_len // args.batch_size + int(tst_dset_len % args.batch_size != 0)
            else:
                num_tst_batches = CELEBA_TEST_LEN // args.batch_size + int(CELEBA_TEST_LEN % args.batch_size != 0)
            logger(f"==== TEST LEN {CELEBA_TEST_LEN} BS {args.batch_size} NB {num_tst_batches} ====")
        elif args.data == "cub":
            num_tst_batches = CUB_TEST_LEN // args.batch_size + int(CUB_TEST_LEN % args.batch_size != 0)
            logger(f"==== TEST LEN {CUB_TEST_LEN} BS {args.batch_size} NB {num_tst_batches} ====")
        else:
            raise ValueError(f"Unrecognized dataset {args.data}")
        # reset the iterator
        for i in range(num_tst_batches - 1):
            skip_batch_, _ = data_test_batch()
            if args.dset_split_type == "zero_shot":
                if i == num_tst_batches - 2:
                    assert skip_batch_.shape[0] == tst_dset_len % args.batch_size
                else:
                    assert skip_batch_.shape[0] == args.batch_size
    else:
        num_tst_batches = num_zs_tst_batches = -1

    eval_x_init = init_x_random()
    eval_b_init = init_b_random()

    plot_samples(eval_x_init, f"{args.save_dir}/init_dist.png")

    if args.full_test:
        num_trn_batches = trn_dset_len // args.batch_size
        num_reset_itrs = start_itr % num_trn_batches

        logger(f"======================= RESET ITERATOR {num_reset_itrs} itrs =======================")

        start_reset = time.time()
        for _ in range(num_reset_itrs):
            data_batch()
        reset_time = time.time() - start_reset

        logger(f"======================= RESETTING TOOK {reset_time:.4e} s =======================")

    for itr in range(start_itr, args.n_iters):
        if args.eval is None:
            data_start_time = time.time()
            batch_x, batch_y = data_batch()
            data_duration += time.time() - data_start_time

            # make sure drop last behaves correctly when using SubsetRandomSampler
            assert len(batch_x) == args.batch_size

            start_time = time.time()

            if args.warmup_itrs_from_restart > 0:
                if (itr - start_itr) < args.warmup_itrs_from_restart:
                    lr = args.lr * ((itr - start_itr) + 1) / float(args.warmup_itrs_from_restart)
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr
            else:
                if itr < args.warmup_itrs:
                    lr = args.lr * (itr + 1) / float(args.warmup_itrs)
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr

            if len(args.lr_itr_at) > 0:
                bisect_lr_itr = bisect(args.lr_itr_at, itr)
                if bisect_lr_itr > 0:
                    lr = args.lr_at[bisect_lr_itr - 1]
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr

            x_neg_init = init_x_random()
            b_neg_init = init_b_random()

            if label_dim > 1:
                missing_y_mask = batch_y == -1
                if args.model == "joint":
                    if missing_y_mask.any():
                        if unif_label_shape:
                            resample_fix_label_axis = -1
                        else:
                            resample_fix_label_axis = missing_y_mask
                        # TODO: config the sampler here (number of steps, short-run vs. PCD etc.)
                        resampled_batch_y = sample_b(logp_net, batch_x, one_hot(batch_y),
                                                     fix_label_axis=resample_fix_label_axis)
                        logp_real = energy(logp_net, batch_x, resampled_batch_y)
                    else:
                        logp_real = energy(logp_net, batch_x, one_hot(batch_y))
                else:
                    logits = logp_net(batch_x)
                    logits_logsumexp = logit_logsumexp(logits)
                    if args.poj_joint:
                        # index the logits according to the observed classes, 0 for unobserved
                        log_probs = logit_log_prob_ind_missing(shape_logits(logits), batch_y, missing_y_mask)
                        # use indexed log probs for observed, logsumexp (marginalized) for unobserved
                        logp_real = (log_probs + logits_logsumexp * missing_y_mask).sum(-1)
                    else:
                        logp_real = logits_logsumexp.sum(-1)

                        if args.p_y_x > 0:
                            logp_y_x = logit_log_p_y_x(logits, batch_y, missing_y_mask, logits_logsumexp)
            elif label_dim == 0:
                logp_real = energy(logp_net, batch_x)
            else:
                if args.model == "joint":
                    logp_real = energy(logp_net, batch_x, batch_y[:, None])
                else:
                    raise NotImplementedError

            if args.mode != "sup":
                if label_dim == 0 or (args.model == "poj" and not args.poj_joint):
                    *x_neg, log_num_added = uncond_sample_x(logp_net, x_neg_init,
                                                            truncated_bp=args.truncated_bp, full_bp=args.full_bp,
                                                            transform_every=None, return_num_added=True)
                    if args.truncated_bp or args.full_bp:
                        x_neg, x_neg_kl = x_neg
                    else:
                        x_neg, = x_neg
                    logp_fake = energy(logp_net, x_neg)
                else:
                    set_gibbs_steps = args.gibbs_steps
                    if len(args.sgld_steps_itr_at) > 0:
                        bisect_sgld_steps_itr = bisect(args.sgld_steps_itr_at, itr)
                        if bisect_sgld_steps_itr > 0:
                            set_gibbs_steps = args.gibbs_steps_at[bisect_sgld_steps_itr - 1]

                    (*x_neg, log_num_added), b_neg = sample_x_b(logp_net, x_neg_init, b_neg_init,
                                                                gibbs_steps=set_gibbs_steps,
                                                                truncated_bp=args.truncated_bp, full_bp=args.full_bp,
                                                                return_num_added=True)
                    if args.truncated_bp or args.full_bp:
                        x_neg, x_neg_kl = x_neg
                    else:
                        x_neg, = x_neg
                    if args.poj_joint:
                        logits_neg = logp_net(x_neg)
                        logp_fake = logit_log_prob_ind(shape_logits(logits_neg), b_neg.long()).sum(-1)
                    else:
                        logp_fake = energy(logp_net, x_neg, b_neg)

                    if args.p_y_x > 0:
                        b_x_neg = sample_b(logp_net, batch_x, b_neg_init)
                        logp_y_x_fake = energy(logp_net, batch_x, b_x_neg)
                        logp_y_x = logp_real.mean() - logp_y_x_fake.mean()

                if args.kl > 0 and (itr > args.kl_start or (args.yd_buffer is not None and len(replay_buffer) > 1000)):

                    # get samples for kl before updating the buffer
                    kl_samples, _ = init_sampling(replay_buffer, torch.zeros_like(x_neg_init)[:100])

                    # dequantize the samples (we've already checked we're using images)
                    kl_samples = deq_x(kl_samples)

                    # maximize entropy term
                    dist_matrix = torch.norm(x_neg_kl.view(x_neg_kl.shape[0], -1)[:, None, :] -
                                             kl_samples.view(kl_samples.shape[0], -1)[None, :, :],
                                             p=2, dim=-1)
                    mins = dist_matrix.min(dim=1).values
                    mins_fin = mins > 0
                    num_0 = (mins_fin == 0).sum()
                    if num_0 > 0:
                        logger(f"=== NN 0 for {num_0} examples!! ===")

                    if args.truncated_bp or args.full_bp:
                        logp_net.requires_grad_(False)
                        if label_dim == 0 or args.poj_joint:
                            bp_loss = energy(logp_net, x_neg_kl)
                        else:
                            bp_loss = energy(logp_net, x_neg_kl, b_neg_kl)
                        logp_net.requires_grad_(True)

                    kl_loss = args.kl * torch.log(mins[mins_fin]).mean() + args.bp * bp_loss.mean()

            if args.mode != "sup":
                # otherwise, log_p_xy is set to a dummy constant of 0
                log_p_xy = logp_real.mean() - logp_fake.mean()

            obj = log_p_xy \
                  + args.p_y_x * logp_y_x \
                  + kl_loss \
                  - args.p_control * torch.pow(logp_real, 2).mean() \
                  - args.n_control * torch.pow(logp_fake, 2).mean()
            loss = -obj
            optim.zero_grad()
            loss.backward()

            if args.clip_grad_norm > 0:
                clip_grad_norm_(logp_net.parameters(), args.clip_grad_norm)

            optim.step()

            if loss.abs().item() > 1e8:
                logger(f"Loss diverged at {itr} with {loss.abs().item()} | "
                       f"log p (real) = {logp_real.mean().item():.4f}, "
                       f"log p (fake) = {logp_fake.mean().item():.4f}, "
                       f"log p y_x = {logp_y_x.mean().item():.4f}, "
                       f"log p y_x (fake) = {logp_y_x_fake.mean().item():.4f}, "
                       f"kl = {kl_loss.mean().item():.4f}, "
                       f"p control = {torch.pow(logp_real, 2).mean().item():.4f}, "
                       f"n control = {torch.pow(logp_fake, 2).mean().item():.4f},"
                       f"num added = {log_num_added}")
                assert False

            duration += time.time() - start_time

            if args.log_ema:
                logger(f"Params the same before ema: {ema_params(logp_net, ema_logp_net)}")
            ema_model(logp_net, ema_logp_net, args.ema)
            if args.log_ema:
                logger(f"Params the same after ema: {ema_params(logp_net, ema_logp_net)}")

        if itr % args.print_every == 0 or itr % args.plot_every == 0 or args.eval is not None:
            ema_logp_net.eval()

            if args.log_ema:
                logger(f"Params the same before log: {ema_params(logp_net, ema_logp_net)}")

            if label_dim > 1:
                eval_missing_y_mask = eval_y == -1
                if args.model == "joint":
                    if eval_missing_y_mask.any():
                        if unif_label_shape:
                            resample_fix_label_axis = -1
                        else:
                            resample_fix_label_axis = eval_missing_y_mask
                        resampled_eval_y = sample_b(logp_net, eval_x, one_hot(eval_y),
                                                     fix_label_axis=resample_fix_label_axis)
                        eval_logp_real = energy(logp_net, eval_x, resampled_eval_y)
                    else:
                        eval_logp_real = energy(ema_logp_net, eval_x, one_hot(eval_y))
                else:
                    eval_logits = logp_net(eval_x)
                    eval_logits_logsumexp = logit_logsumexp(eval_logits)
                    if args.poj_joint:
                        eval_log_probs = logit_log_prob_ind_missing(shape_logits(eval_logits), eval_y,
                                                                    eval_missing_y_mask)
                        eval_logp_real = (eval_log_probs + eval_logits_logsumexp * eval_missing_y_mask).sum(-1)
                    else:
                        eval_logp_real = eval_logits_logsumexp.sum(-1)

                        eval_logp_y_x = logit_log_p_y_x(eval_logits, eval_y,
                                                        eval_missing_y_mask, eval_logits_logsumexp)
            elif label_dim == 0:
                eval_logp_real = energy(ema_logp_net, eval_x)
            else:
                if args.model == "joint":
                    eval_logp_real = energy(ema_logp_net, eval_x, eval_y[:, None])
                else:
                    raise NotImplementedError

            if args.mode != "sup":
                if label_dim == 0 or (args.model == "poj" and not args.poj_joint):
                    eval_x_neg = uncond_sample_x(ema_logp_net, eval_x_init, update_buffer=False,
                                                 return_steps=args.return_steps,
                                                 steps_batch_ind=args.steps_batch_ind)
                    if args.return_steps > 0:
                        eval_x_neg, eval_x_neg_steps = eval_x_neg
                    else:
                        eval_x_neg_steps = None
                    ar, lr, lf, la, mt, pt, h, eval_b = -1, -1, -1, -1, -1, -1, -1, torch.zeros(args.batch_size)
                    eval_logp_fake = energy(ema_logp_net, eval_x_neg)

                    if args.model == "poj":
                        eval_b = logp_net(eval_x_neg)

                    eval_logp_y_x_fake = torch.tensor(0.)
                else:
                    set_gibbs_steps = args.gibbs_steps
                    if len(args.sgld_steps_itr_at) > 0:
                        bisect_sgld_steps_itr = bisect(args.sgld_steps_itr_at, itr)
                        if bisect_sgld_steps_itr > 0:
                            set_gibbs_steps = args.gibbs_steps_at[bisect_sgld_steps_itr - 1]

                    eval_x_neg, eval_b, *info = sample_x_b(ema_logp_net, eval_x_init, eval_b_init,
                                                           gibbs_steps=set_gibbs_steps,
                                                           verbose=True, update_buffer=False,
                                                           return_steps=args.return_steps,
                                                           steps_batch_ind=args.steps_batch_ind)
                    if args.return_steps > 0:
                        eval_x_neg, eval_x_neg_steps = eval_x_neg
                    else:
                        eval_x_neg_steps = None
                    ar, lr, lf, la, mt, pt, h = info
                    if args.poj_joint:
                        eval_logits_neg = logp_net(eval_x_neg)
                        eval_logp_fake = logit_log_prob_ind(shape_logits(eval_logits_neg), eval_b.long()).sum(-1)
                    else:
                        eval_logp_fake = energy(ema_logp_net, eval_x_neg, eval_b)

                    eval_b_x = sample_b(ema_logp_net, eval_x, eval_b_init)
                    eval_logp_y_x_fake = energy(ema_logp_net, eval_x, eval_b_x)

                eval_kl_samples, _ = init_sampling(replay_buffer, torch.zeros_like(eval_x_neg)[:100])
                eval_dist_matrix = torch.norm(eval_x_neg.view(eval_x_neg.shape[0], -1)[:, None, :] -
                                              eval_kl_samples.view(eval_kl_samples.shape[0], -1)[None, :, :],
                                              p=2, dim=-1)
                eval_kl_loss = torch.log(eval_dist_matrix.min(dim=1)[0]).mean()
            else:
                # lint
                ar = lr = lf = la = mt = pt = h = -1
                eval_x_neg = eval_x_neg_steps = eval_b = None

            if itr % args.print_every == 0:
                logger(f"Itr {itr} "
                       f"({data_duration / args.print_every:.2f}) "
                       f"({duration / args.print_every:.2f}) | "
                       f"lr = {optim.param_groups[0]['lr']:.4e} | "
                       f"log p (real) = {eval_logp_real.mean().item():.4f}, "
                       f"log p (fake) = {eval_logp_fake.mean().item():.4f}, "
                       f"log p y_x = {eval_logp_y_x.mean().item():.4f}, "
                       f"log p y_x (fake) = {eval_logp_y_x_fake.mean().item():.4f}, "
                       f"kl = {eval_kl_loss.mean().item():.4f}, "
                       f"p control = {torch.pow(eval_logp_real, 2).mean().item():.4f}, "
                       f"n control = {torch.pow(eval_logp_fake, 2).mean().item():.4f}, "
                       f"num added = {log_num_added} | "
                       f"ar = {ar:.2f}, "
                       f"lr = {lr:.2f}, "
                       f"lf = {lf:.2f}, "
                       f"la = {la:.2f}, "
                       f"mt = {mt:.2f}, "
                       f"pt = {pt:.2f}, "
                       f"h = {h:.2f}")
                # reset timers
                duration = 0
                data_duration = 0

            def pred_zero_shot(net, x, y):
                zero_shot_logits = shape_logits(net(x)).detach().cpu()
                # zero_shot_log_probs = logit_log_prob(zero_shot_logits, logit_logsumexp(zero_shot_logits))
                # zero_shot_pred = logits_ind_combos(zero_shot_log_probs, zero_shot_combos)
                zero_shot_pred = logits_ind_combos(zero_shot_logits, zero_shot_combos, include_marginals=True)
                zero_shot_y = lbl_in_combos(y, dset_label_info, zero_shot_combos)
                return zero_shot_pred.detach().cpu(), zero_shot_y.detach().cpu()

            def eval_cond_acc_print_hist(net, x, y, fixed_cond, fn_base):
                # get predictions of model
                y = y.detach().cpu()
                eval_cond_logits_ = shape_logits(net(x)).detach().cpu()
                eval_cond_log_probs = logit_log_prob(eval_cond_logits_, logit_logsumexp(eval_cond_logits_))
                eval_lp_ = onehot_poj_logits(eval_cond_log_probs)

                # find where predictions are equal to labels, only at fixed cond labels
                cond_acc_ = (eval_lp_ == y)[:, fixed_cond].float().mean()

                cond_log_prob_ = logit_log_prob_ind(eval_cond_log_probs, y.long())[:, fixed_cond].sum(-1)

                # log the output
                logger(f"{fn_base} acc: {cond_acc_.detach().cpu().item():.4f}, "
                       f"log_prob: {cond_log_prob_.mean().detach().cpu().item()}")

                # save a histogram
                plt.clf()
                plt.hist(cond_log_prob_.detach().cpu().numpy())
                plt.savefig(f"{args.save_dir}/eval_hist_{fn_base}.png")

                # save the values themselves so we can plot them together
                with open(f"{args.save_dir}/eval_save_{fn_base}.pickle", "wb") as f:
                    pickle.dump(cond_log_prob_.detach().cpu().numpy(), f)

            def log_full_tst(individual_f1=False, val=False):
                assert all_binary
                assert implies(val, args.zero_shot)

                eval_zero_shot = args.dset_split_type == "zero_shot" and args.eval is not None

                # energy f1 cannot be batched!
                if args.zero_shot:
                    if val:
                        test_len_ = dset_lengths['val']
                        num_tst_batches_ = num_tst_batches
                    else:
                        test_len_ = dset_lengths['tst']
                        num_tst_batches_ = num_zs_tst_batches
                else:
                    if "utzappos" in args.data:
                        test_len_ = UTZAPPOS_TEST_LEN
                    elif args.data == "celeba":
                        if eval_zero_shot:
                            test_len_ = tst_dset_len
                        else:
                            test_len_ = CELEBA_TEST_LEN
                    elif args.data == "cub":
                        test_len_ = CUB_TEST_LEN
                    else:
                        raise ValueError(f"Unrecognized data {args.data}")

                    if eval_zero_shot:
                        num_tst_batches_ = test_len_ // args.batch_size + int(test_len_ % args.batch_size != 0)
                    else:
                        num_tst_batches_ = num_tst_batches

                sample_acc = torch.zeros(num_tst_batches_)
                energy_acc = torch.zeros(num_tst_batches_)

                individual_y_pred = torch.zeros(test_len_, label_dim)
                individual_y_true = torch.zeros(test_len_, label_dim)
                individual_y_pred_score = torch.zeros(test_len_, 2 * label_dim)
                individual_y_pred_sample = torch.zeros(test_len_, label_dim)

                if eval_zero_shot:
                    individual_zero_shot_y_pred_score = torch.zeros(test_len_, len(zero_shot_combos))
                    individual_zero_shot_y_true = torch.zeros(test_len_, len(zero_shot_combos))
                else:
                    individual_zero_shot_y_pred_score = individual_zero_shot_y_true = None

                individual_y_zs = torch.zeros(test_len_, 3)
                zs_ranges = torch.zeros(3, 2) - 1

                for j in range(num_tst_batches_):
                    if args.zero_shot and val:
                        tst_eval_batch_ = data_val_batch(get_zs=args.zero_shot)
                    else:
                        tst_eval_batch_ = data_test_batch(get_zs=args.zero_shot)

                    if args.zero_shot:
                        tst_eval_x_, tst_eval_y_, tst_eval_zs_ = tst_eval_batch_
                    else:
                        tst_eval_x_, tst_eval_y_ = tst_eval_batch_
                        tst_eval_zs_ = None

                    if j == num_tst_batches_ - 1 and test_len_ % args.batch_size != 0:
                        # TODO: the last batch size needs to be bigger than the number of GPUs
                        assert tst_eval_x_.shape[0] == test_len_ % args.batch_size
                    else:
                        assert tst_eval_x_.shape[0] == args.batch_size

                    if args.model == "joint":
                        # sample-based metrics only make sense for joint model
                        tst_b_eval_n_ = sample_eval_b(ema_logp_net, tst_eval_x_, args.eval_samples)
                        sample_acc[j] = accuracy_b(tst_b_eval_n_, tst_eval_y_)

                        # save predictions for f1 computation at end
                        individual_y_pred_sample[j * args.batch_size: (j + 1) * args.batch_size] = tst_b_eval_n_

                    energy_acc[j], tst_eval_pred_, tst_eval_score_ = energy_eval_b(ema_logp_net,
                                                                                   tst_eval_x_, tst_eval_y_,
                                                                                   batched=True,
                                                                                   return_pred=True,
                                                                                   return_score=True,
                                                                                   detach_and_cpu=True,
                                                                                   check_shapes=True)
                    individual_y_pred[j * args.batch_size: (j + 1) * args.batch_size] = tst_eval_pred_
                    individual_y_true[j * args.batch_size: (j + 1) * args.batch_size] = tst_eval_y_
                    individual_y_pred_score[j * args.batch_size: (j + 1) * args.batch_size] = tst_eval_score_

                    if eval_zero_shot:
                        zs_pred, zs_y = pred_zero_shot(ema_logp_net, tst_eval_x_, tst_eval_y_)
                        individual_zero_shot_y_pred_score[j * args.batch_size: (j + 1) * args.batch_size] = zs_pred
                        individual_zero_shot_y_true[j * args.batch_size: (j + 1) * args.batch_size] = zs_y

                    if args.zero_shot:

                        _zs_lbl_inds = torch.cat([tst_eval_zs_[0][None], tst_eval_zs_[3][None], tst_eval_zs_[6][None]],
                                                 dim=1)

                        individual_y_zs[j * args.batch_size: (j + 1) * args.batch_size] = _zs_lbl_inds

                        if (zs_ranges < 0).any():
                            # uninitialized
                            zs_ranges[0][0] = tst_eval_zs_[1]
                            zs_ranges[0][1] = tst_eval_zs_[2]
                            zs_ranges[1][0] = tst_eval_zs_[4]
                            zs_ranges[1][1] = tst_eval_zs_[5]
                            zs_ranges[2][0] = tst_eval_zs_[7]
                            zs_ranges[2][2] = tst_eval_zs_[8]

                if args.dset_split_type == "zero_shot":
                    actual_num_tst_batches = tst_dset_len // args.batch_size + int(tst_dset_len % args.batch_size != 0)
                    num_reset_itrs_ = actual_num_tst_batches - num_tst_batches_
                    logger(f"======================= RESET TST ITERATOR {num_reset_itrs_} itrs =======================")
                    start_reset_ = time.time()
                    for _ in range(num_reset_itrs_):
                        data_test_batch()
                    reset_time_ = time.time() - start_reset_
                    logger(f"======================= RESETTING TST TOOK {reset_time_:.4e} s =======================")

                if args.model == "joint":
                    logger(f"\tFull Test Accuracy ({args.eval_samples}) = {torch.mean(sample_acc).item():.4f}")

                    sample_f1 = f1_b(individual_y_pred_sample, individual_y_true)
                    logger(f"\tFull Test F1({args.eval_samples}) = {sample_f1.item():.4f}")

                energy_f1 = f1_b(individual_y_pred, individual_y_true, individual=individual_f1)
                micro_energy_f1 = f1_b(individual_y_pred, individual_y_true, micro=True)
                if individual_f1:
                    # print indices of any F1 scores that are 0
                    logger("\t" + ", ".join([str(ind_) for ind_, f1_ in enumerate(energy_f1) if f1_ == 0]))

                energy_ece = ece_b(individual_y_pred_score, individual_y_true, individual=individual_f1)
                micro_energy_ece = ece_b(individual_y_pred_score, individual_y_true, micro=True)

                energy_ap = ap_b(individual_y_pred_score, individual_y_true, individual=individual_f1)
                micro_energy_ap = ap_b(individual_y_pred_score, individual_y_true, micro=True)

                energy_auroc = auroc_b(individual_y_pred_score, individual_y_true, individual=individual_f1)
                micro_energy_auroc = auroc_b(individual_y_pred_score, individual_y_true, micro=True)

                # =============== PLOT CURVES ===============
                if args.eval is not None:
                    if args.save_test_predictions:
                        with open(f"{args.save_dir}/full_tst_preds.pickle", "wb") as f:
                            d_tst = {
                                "score": individual_y_pred_score.detach().cpu().numpy(),
                                "true": individual_y_true.detach().cpu().numpy()
                            }
                            pickle.dump(d_tst, f, protocol=4)

                    # only plot if evaluating
                    pr_curve_b(individual_y_pred_score, individual_y_true, itr)
                    pr_curve_b(individual_y_pred_score, individual_y_true, itr, micro=True)

                    ece_plot_b(individual_y_pred_score, individual_y_true, itr, individual=True)
                    ece_plot_b(individual_y_pred_score, individual_y_true, itr, micro=True)

                    roc_curve_b(individual_y_pred_score, individual_y_true, itr)
                    roc_curve_b(individual_y_pred_score, individual_y_true, itr, micro=True)

                if args.model == "joint":
                    raise NotImplementedError(f"Train metrics have not been implemented for joint model")

                # =============== COMPUTE TRN BATCH METRICS ===============

                trn_eval_x_, trn_eval_y_ = eval_x, eval_y

                energy_acc_trn, trn_eval_pred_, trn_eval_score_ = energy_eval_b(ema_logp_net,
                                                                                trn_eval_x_, trn_eval_y_,
                                                                                batched=True,
                                                                                return_pred=True,
                                                                                return_score=True)
                individual_y_pred_trn = trn_eval_pred_
                individual_y_true_trn = trn_eval_y_
                individual_y_pred_score_trn = trn_eval_score_

                energy_f1_trn = f1_b(individual_y_pred_trn, individual_y_true_trn, individual=individual_f1)
                micro_energy_f1_trn = f1_b(individual_y_pred_trn, individual_y_true_trn, micro=True)

                energy_ece_trn = ece_b(individual_y_pred_score_trn, individual_y_true_trn, individual=individual_f1)
                micro_energy_ece_trn = ece_b(individual_y_pred_score_trn, individual_y_true_trn, micro=True)

                energy_ap_trn = ap_b(individual_y_pred_score_trn, individual_y_true_trn, individual=individual_f1)
                micro_energy_ap_trn = ap_b(individual_y_pred_score_trn, individual_y_true_trn, micro=True)

                energy_auroc_trn = auroc_b(individual_y_pred_score_trn, individual_y_true_trn, individual=individual_f1)
                micro_energy_auroc_trn = auroc_b(individual_y_pred_score_trn, individual_y_true_trn, micro=True)

                # save tst acc for checkpointing
                tst_accs[itr] = torch.mean(energy_acc).item()

                logger(f"\tAverage Freq: {individual_y_true.mean():.4f}")
                logger(f"\t(ENERGY) Full Test Accuracy = {torch.mean(energy_acc).item():.4f} "
                       f"({torch.mean(energy_acc_trn).item():.4f})")

                logger(f"\t(ENERGY) Full Test F1 (macro) = {torch.mean(energy_f1).item():.4f} "
                       f"({torch.mean(energy_f1_trn).item():.4f})")
                logger(f"\t(ENERGY) Full Test F1 (micro) = {micro_energy_f1:.4f} ({micro_energy_f1_trn:.4f})")
                logger(f"\t(ENERGY) Full Test Ind. F1")
                for lbl_ind_, (f1_, f1_trn_, freq_, trn_freq_) in enumerate(zip(energy_f1, energy_f1_trn,
                                                                                individual_y_true.mean(0),
                                                                                trn_dset_freqs.squeeze())):
                    logger(f"\t{lbl_ind_:02d} F1: {f1_:.4f} ({f1_trn_:.4f}) "
                           f"Freq: {freq_:.4f} ({trn_freq_:.4f}) | "
                           f"{'better' if f1_ > freq_ else 'worse'} than random "
                           f"({'better' if f1_trn_ > trn_freq_ else 'worse'})")

                logger(f"\t(ENERGY) Full Test AP (macro) = {torch.mean(energy_ap).item():.4f} "
                       f"({torch.mean(energy_ap_trn).item():.4f})")
                logger(f"\t(ENERGY) Full Test AP (micro) = {micro_energy_ap:.4f} ({micro_energy_ap_trn:.4f})")
                logger(f"\t(ENERGY) Full Test Ind. AP")
                for lbl_ind_, (ap_, ap_trn_, freq_, trn_freq_) in enumerate(zip(energy_ap, energy_ap_trn,
                                                                                individual_y_true.mean(0).cpu(),
                                                                                trn_dset_freqs.squeeze().cpu())):
                    logger(f"\t{lbl_ind_:02d} AP: {ap_:.4f} ({ap_trn_:.4f}) "
                           f"Freq: {freq_:.4f} ({trn_freq_:.4f}) | "
                           f"{'better' if ap_ > freq_ else 'worse'} than random "
                           f"({'better' if ap_trn_ > trn_freq_ else 'worse'})")

                logger(f"\t(ENERGY) Full Test AUROC (macro) = {torch.mean(energy_auroc).item():.4f} "
                       f"({torch.mean(energy_auroc_trn).item():.4f})")
                logger(f"\t(ENERGY) Full Test AUROC (micro) = {micro_energy_auroc:.4f} ({micro_energy_auroc_trn:.4f})")
                logger(f"\t(ENERGY) Full Test Ind. AUROC")
                for lbl_ind_, (auroc_, auroc_trn_, freq_, trn_freq_) in enumerate(zip(energy_auroc, energy_auroc_trn,
                                                                                      individual_y_true.mean(0).cpu(),
                                                                                      trn_dset_freqs.squeeze().cpu())):
                    logger(f"\t{lbl_ind_:02d} AUROC: {auroc_:.4f} ({auroc_trn_:.4f}) "
                           f"Freq: {freq_:.4f} ({trn_freq_:.4f}) | "
                           f"{'better' if auroc_ > freq_ else 'worse'} than random "
                           f"({'better' if auroc_trn_ > trn_freq_ else 'worse'})")

                logger(f"\t(ENERGY) Full Test ECE (macro) = {torch.mean(energy_ece).item():.4e} "
                       f"({torch.mean(energy_ece_trn).item():.4e})")
                logger(f"\t(ENERGY) Full Test ECE (micro) = {micro_energy_ece.item():.4e} "
                       f"({micro_energy_ece_trn.item():.4e})")
                logger(f"\t(ENERGY) Full Test Ind. ECE")
                for lbl_ind_, (ece_, ece_trn_, freq_, trn_freq_) in enumerate(zip(energy_ece, energy_ece_trn,
                                                                                  individual_y_true.mean(0),
                                                                                  trn_dset_freqs.squeeze())):
                    logger(f"\t{lbl_ind_:02d} ECE: {ece_:.4e} ({ece_trn_:.4e}) Freq: {freq_:.4f} ({trn_freq_:.4f})")

                if args.zero_shot:
                    # compute metrics for zero shot
                    shaped_individual_y_pred_score = shape_logits(individual_y_pred_score)
                    zs_scores = torch.zeros(3, test_len_, label_dim)  # TODO: this is a jagged array
                    for i in range(3):
                        zs_ind_min, zs_ind_max = zs_ranges[i]
                        zs_scores[i] = cond_from_joint(shaped_individual_y_pred_score[:, zs_ind_min:(zs_ind_max + 1)])

                if eval_zero_shot:
                    # TODO: evaluate accuracy on subset of examples based on less "classes"
                    # filter to examples with mutually exclusive held-out label combinations
                    tst_labels_combos = lbl_in_combos(individual_y_true, dset_label_info, zero_shot_combos)
                    tst_labels_combos_filter = tst_labels_combos.sum(1) == 1
                    
                    individual_zero_shot_y_pred_score = individual_zero_shot_y_pred_score[tst_labels_combos_filter]
                    individual_zero_shot_y_true = individual_zero_shot_y_true[tst_labels_combos_filter]
                    
                    individual_zero_shot_y_pred_lbl = individual_zero_shot_y_pred_score.max(1).indices
                    individual_zero_shot_y_true_lbl = individual_zero_shot_y_true.max(1).indices

                    assert individual_zero_shot_y_pred_lbl.shape == individual_zero_shot_y_true_lbl.shape
                    
                    acc_zero_shot = (individual_zero_shot_y_pred_lbl == individual_zero_shot_y_true_lbl).float().mean()
                    logger(f"\t(ENERGY) Zero-Shot Test Accuracy = {acc_zero_shot.item():.4f}")
                    zs_cls_freqs = individual_zero_shot_y_true.mean(0) * 100
                    for zs_cls_freq, zs_cls in zip(zs_cls_freqs, zero_shot_combos):
                        logger(f"\t{zs_cls} Freq: {zs_cls_freq.detach().cpu().item():.2f}")

            if args.mode == "sup":
                a_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="acc")
                logger(f"\t(ENERGY) Accuracy = {a_m.item():.4f}")

                f_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="f1")
                logger(f"\t(ENERGY) F1 = {f_m.item():.4f}")

                e_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="ece")
                logger(f"\t(ENERGY) ECE = {e_m.item():.4e}")

                p_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="ap")
                logger(f"\t(ENERGY) AP = {p_m.item():.4e}")

                r_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="auroc")
                logger(f"\t(ENERGY) AUROC = {r_m.item():.4e}")

                if args.full_test:
                    log_full_tst(True)

            if (args.mode != "sup") and (itr % args.plot_every == 0) and (itr >= args.plot_after):
                plot_samples(eval_x_neg, f"{args.save_dir}/samples_{itr}.png")
                if args.return_steps > 0:
                    if args.steps_batch_ind is None:
                        for chain_itr in range(eval_x_neg_steps.shape[0]):
                            plot_samples(eval_x_neg_steps[chain_itr],
                                         f"{args.save_dir}/chain_{itr}_itr_{chain_itr}.png")
                    else:
                        plot_samples(eval_x_neg_steps, f"{args.save_dir}/chain_{itr}.png")

                if label_dim == 0:
                    if args.plot_uncond_fresh:
                        eval_x_neg_fresh = uncond_sample_x(ema_logp_net, eval_x_init, steps=args.test_k,
                                                           update_buffer=False,
                                                           new_replay_buffer=[], new_y_replay_buffer=[],
                                                           return_steps=args.return_steps,
                                                           steps_batch_ind=args.steps_batch_ind,
                                                           transform_every=args.transform_every)
                        if args.return_steps > 0:
                            eval_x_neg_fresh, eval_x_neg_fresh_steps = eval_x_neg_fresh
                            plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")
                            if args.steps_batch_ind is None:
                                for chain_itr in range(eval_x_neg_fresh_steps.shape[0]):
                                    plot_samples(eval_x_neg_fresh_steps[chain_itr],
                                                 f"{args.save_dir}/chain_fresh_{itr}_itr_{chain_itr}.png")
                            else:
                                plot_samples(eval_x_neg_fresh_steps, f"{args.save_dir}/chain_fresh_{itr}.png")
                        else:
                            plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")

                    if args.data not in IMG_DSETS:
                        ema_logp_net.cpu()

                        plt.clf()
                        plt_flow_density(partial(energy, ema_logp_net), plt.gca())
                        plt.savefig(f"{args.save_dir}/density_{itr}.png")

                        plt.clf()
                        plt_flow_density(partial(energy, ema_logp_net), plt.gca(), exp=False)
                        plt.savefig(f"{args.save_dir}/density_log_{itr}.png")

                else:
                    if args.model == "joint":
                        # compare the accuracy when sampling b given just x, and b given x and y
                        b_acc, *info = sample_b(ema_logp_net, eval_x, eval_b_init, verbose=True)
                        ar, lr, lf, la, mt, pt, h = info
                        if label_dim > 1:
                            b_pos = sample_b(ema_logp_net, eval_x, one_hot(eval_y).float(), steps=args.test_n_steps)
                        else:
                            b_pos = sample_b(ema_logp_net, eval_x, eval_y.float(), steps=args.test_n_steps)
                        logger(f"\tSample Accuracy = {accuracy_b(b_acc.squeeze(), eval_y):.4f}, "
                               f"Posterior Accuracy = {accuracy_b(b_pos.squeeze(), eval_y):.4f} | "
                               f"Sample F1 = {f1_b(b_acc.squeeze(), eval_y):.4f}, "
                               f"Posterior F1 = {f1_b(b_pos.squeeze(), eval_y):.4f} | "
                               f"ar = {ar:.2f}, "
                               f"lr = {lr:.2f}, "
                               f"lf = {lf:.2f}, "
                               f"la = {la:.2f}, "
                               f"mt = {mt:.2f}, "
                               f"pt = {pt:.2f}, "
                               f"h = {h:.2f}")

                        # check the accuracy of the model from one sampled prediction and several sampled predictions
                        b_eval_1 = sample_eval_b(ema_logp_net, eval_x, 1)
                        b_eval_n, bs_eval_n = sample_eval_b(ema_logp_net, eval_x, args.eval_samples,
                                                            return_individual=True)
                        logger(f"\t(SAMPLE) Accuracy (1) = {accuracy_b(b_eval_1, eval_y):.4f}, "
                               f"Accuracy ({args.eval_samples}) = {accuracy_b(b_eval_n, eval_y):.4f}")
                        logger(f"\t(SAMPLE) F1 (1) = {f1_b(b_eval_1, eval_y):.4f}, "
                               f"F1 ({args.eval_samples}) = {f1_b(b_eval_n, eval_y):.4f}")
                    else:
                        bs_eval_n = None  # linter

                    if args.data not in ["mwa", "celeba", "dsprites", "utzappos", "cub"] or args.model == "poj":
                        a_m, pred_scores = energy_eval_b(ema_logp_net, eval_x, eval_y, return_pred=True)
                        logger(f"\t(ENERGY) Accuracy = {a_m.item():.4f}")

                        f_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="f1")
                        logger(f"\t(ENERGY) F1 = {f_m.item():.4f}")

                        e_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="ece")
                        logger(f"\t(ENERGY) ECE = {e_m.item():.4e}")

                        p_m = energy_eval_b(ema_logp_net, eval_x, eval_y, metric="ap")
                        logger(f"\t(ENERGY) AP = {p_m.item():.4e}")

                        if args.full_test and not args.just_sampling:
                            log_full_tst(True)

                        if args.plot_energy_b:
                            assert args.model == "joint"
                            pred_scores = torch.nn.functional.softmax(pred_scores, dim=-1)
                            plt.clf()
                            if label_dim > 1:
                                _b_plot = label(bs_eval_n, check_shape=False)
                            else:
                                _b_plot = bs_eval_n[:, 0]
                            _b_plot = _b_plot.cpu().detach().numpy()
                            # unintuitive mpl behaviour
                            # https://stackoverflow.com/questions/3866520/how-can-i-plot-a-histogram-such-that-the-heights-of-the-bars-sum-to-1-in-matplot/16399202#16399202
                            plt.hist(_b_plot[0], weights=np.ones_like(_b_plot[0]) / len(_b_plot[0]), zorder=1)

                            plt.scatter(np.arange(pred_scores.shape[1]), pred_scores[0].cpu().detach().numpy(),
                                        c="red", zorder=2)
                            plt.savefig(f"{args.save_dir}/energy_b_{itr}.png")

                    if args.data == "mnist":
                        if args.model != "joint":
                            raise NotImplementedError

                        if args.full_test:
                            log_full_tst()
                        else:
                            tst_b_eval_1 = sample_eval_b(ema_logp_net, tst_eval_x, 1)
                            tst_b_eval_n, tst_bs_eval_n = sample_eval_b(ema_logp_net, tst_eval_x, args.eval_samples,
                                                                        return_individual=True)
                            logger(f"\t(SAMPLE) Test Accuracy (1) = {accuracy_b(tst_b_eval_1, tst_eval_y):.4f}, "
                                   f"Test Accuracy ({args.eval_samples}) = {accuracy_b(tst_b_eval_n, tst_eval_y):.4f}")

                            tst_a_m = energy_eval_b(ema_logp_net, tst_eval_x, tst_eval_y)
                            logger(f"\t(ENERGY) Test Accuracy = {tst_a_m.item():.4f}")

                    elif args.dsprites_test:
                        if args.model != "joint":
                            raise NotImplementedError

                        tst_b_eval_1 = sample_eval_b(ema_logp_net, tst_eval_x, 1)
                        tst_b_eval_n, tst_bs_eval_n = sample_eval_b(ema_logp_net, tst_eval_x, args.eval_samples,
                                                                    return_individual=True)
                        logger(f"\t(SAMPLE) Test Accuracy (1) = {accuracy_b(tst_b_eval_1, tst_eval_y):.4f}, "
                               f"Test Accuracy ({args.eval_samples}) = {accuracy_b(tst_b_eval_n, tst_eval_y):.4f}")

                    if args.model == "joint" or args.poj_joint:
                        labels_b = label(eval_b)
                        plt.clf()
                        if label_dim > 1:
                            plt.hist(labels_b.cpu().detach().numpy())
                        else:
                            plt.hist(eval_b[:, 0].cpu().detach().numpy())
                        plt.savefig(f"{args.save_dir}/b_{itr}.png")
                    else:
                        labels_b = label(onehot_poj_logits(eval_b))

                    # ================ plot conditioning on a single attribute ================

                    if label_shape != (-1,):
                        if all_binary:
                            _iterator = zip(range(label_dim), (2 for _ in range(label_dim)))
                        else:
                            _iterator = enumerate(label_shape)
                        for label_axis, label_axis_dim in _iterator:
                            for cond_cls in range(label_axis_dim):
                                if args.data == "dsprites":
                                    if cond_cls not in (0, 3, 6, 9, 12, 15):
                                        continue
                                elif args.data == "utzappos_old":
                                    if not (label_axis == 0 and cond_cls in (0, 1)):  # Boots, Sandals
                                        continue
                                elif args.data == "utzappos":
                                    if not check_cond_cls_utzappos(label_axis, cond_cls):
                                        continue
                                elif args.data == "celeba":
                                    if not check_cond_cls_celeba(label_axis, cond_cls):
                                        continue
                                elif args.data == "cub":
                                    if not check_cond_cls_cub(label_axis, cond_cls):
                                        continue

                                _mask = labels_b[:, label_axis] == cond_cls
                                plot_samples(eval_x_neg[_mask],
                                             f"{args.save_dir}/samples_{label_axis}_{cond_cls}_{itr}.png")

                                batch_size_ = args.batch_size * args.plot_cond_filter

                                if args.plot_cond_filter > 1:
                                    eval_x_init_, eval_b_init_ = init_x_random(batch_size_), init_b_random(batch_size_)
                                else:
                                    eval_x_init_, eval_b_init_ = eval_x_init, eval_b_init

                                if args.plot_cond_buffer:
                                    if isinstance(replay_buffer, ReplayBuffer):
                                        replay_buffer_ = replay_buffer._storage
                                        y_replay_buffer_ = y_replay_buffer._storage
                                    else:
                                        replay_buffer_ = replay_buffer
                                        y_replay_buffer_ = y_replay_buffer
                                    # next_cond_cls_ is repeated along batch dimension, so just index one of them
                                    cond_buffer_mask_ = y_replay_buffer_[:, label_axis] == cond_cls
                                    plot_samples(replay_buffer_[cond_buffer_mask_][:args.batch_size],
                                                 f"{args.save_dir}/buffer_{label_axis}_{cond_cls}_{itr}.png")

                                if args.plot_cond_init_buffer:

                                    if args.model != "joint":
                                        raise NotImplementedError

                                    x_cond = sample_x_b(ema_logp_net, eval_x_init,
                                                        format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                        fix_label_axis=label_axis,
                                                        update_buffer=False)[0]
                                    plot_samples(x_cond,
                                                 f"{args.save_dir}/samples_{label_axis}_{cond_cls}_cond_{itr}.png")

                                if args.plot_cond_continue_buffer:
                                    if isinstance(replay_buffer, ReplayBuffer):
                                        replay_buffer_ = replay_buffer._storage
                                        y_replay_buffer_ = y_replay_buffer._storage
                                    else:
                                        replay_buffer_ = replay_buffer
                                        y_replay_buffer_ = y_replay_buffer
                                    cond_buffer_mask_ = y_replay_buffer_[:, label_axis] == cond_cls

                                    x_init_buffer_ = replay_buffer_[cond_buffer_mask_][:batch_size_]
                                    mul_format_lbl_ = y_replay_buffer_[cond_buffer_mask_][:batch_size_]
                                    if x_init_buffer_.shape[0] > 0:
                                        x_cond_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                  mul_format_lbl_,
                                                                  fix_label_axis=label_axis,
                                                                  steps=args.test_k,
                                                                  update_buffer=False,
                                                                  new_replay_buffer=[], new_y_replay_buffer=[],
                                                                  return_steps=args.return_steps,
                                                                  steps_batch_ind=args.steps_batch_ind,
                                                                  gibbs_steps=args.test_gibbs_steps,
                                                                  gibbs_k_steps=args.test_gibbs_k_steps,
                                                                  gibbs_n_steps=args.test_gibbs_n_steps,
                                                                  transform_every=args.transform_every)[0]

                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, label_axis)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_{label_axis}_{cond_cls}_"
                                                         f"cond_continue_buffer_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_{label_axis}_{cond_cls}_"
                                                                 f"cond_continue_buffer_{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_{label_axis}_{cond_cls}_"
                                                             f"cond_continue_buffer_{itr}.png")
                                        else:
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, label_axis)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_{label_axis}_{cond_cls}_"
                                                         f"cond_continue_buffer_{itr}.png")

                                        if args.plot_cond_marginalize:
                                            x_cond_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                      mul_format_lbl_,
                                                                      fix_label_axis=label_axis,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every,
                                                                      marginalize_free_b=True)[0]
                                            if args.return_steps > 0:
                                                x_cond_fresh, fresh_steps = x_cond_fresh
                                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                               mul_format_lbl_, label_axis)
                                                plot_samples(x_cond_fresh,
                                                             f"{args.save_dir}/"
                                                             f"samples_{label_axis}_{cond_cls}_cond_continue_buffer_"
                                                             f"marginalize_{itr}.png")
                                                if args.steps_batch_ind is None:
                                                    for chain_itr in range(fresh_steps.shape[0]):
                                                        plot_samples(fresh_steps[chain_itr],
                                                                     f"{args.save_dir}/"
                                                                     f"chain_{label_axis}_{cond_cls}_"
                                                                     f"cond_continue_buffer_"
                                                                     f"marginalize_{itr}_itr_{chain_itr}.png")
                                                else:
                                                    plot_samples(fresh_steps,
                                                                 f"{args.save_dir}/"
                                                                 f"chain_{label_axis}_{cond_cls}_cond_continue_buffer_"
                                                                 f"marginalize_{itr}.png")
                                            else:
                                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                               mul_format_lbl_,
                                                                               label_axis)
                                                plot_samples(x_cond_fresh,
                                                             f"{args.save_dir}/"
                                                             f"samples_{label_axis}_{cond_cls}_cond_continue_buffer_"
                                                             f"marginalize_{itr}.png")

                                if args.plot_cond_continue_buffer_uncond:
                                    if isinstance(replay_buffer, ReplayBuffer):
                                        replay_buffer_ = replay_buffer._storage
                                        y_replay_buffer_ = y_replay_buffer._storage
                                    else:
                                        replay_buffer_ = replay_buffer
                                        y_replay_buffer_ = y_replay_buffer

                                    x_init_buffer_ = replay_buffer_[:args.batch_size]
                                    mul_format_lbl_ = y_replay_buffer_[:args.batch_size]
                                    if x_init_buffer_.shape[0] > 0:
                                        x_cond_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                  mul_format_lbl_,
                                                                  fix_label_axis=label_axis,
                                                                  steps=args.test_k,
                                                                  update_buffer=False,
                                                                  new_replay_buffer=[], new_y_replay_buffer=[],
                                                                  return_steps=args.return_steps,
                                                                  steps_batch_ind=args.steps_batch_ind,
                                                                  gibbs_steps=args.test_gibbs_steps,
                                                                  gibbs_k_steps=args.test_gibbs_k_steps,
                                                                  gibbs_n_steps=args.test_gibbs_n_steps,
                                                                  transform_every=args.transform_every)[0]

                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, label_axis)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_{label_axis}_{cond_cls}_"
                                                         f"cond_continue_buffer_uncond_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_{label_axis}_{cond_cls}_"
                                                                 f"cond_continue_buffer_uncond_"
                                                                 f"{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_{label_axis}_{cond_cls}_"
                                                             f"cond_continue_buffer_uncond_{itr}.png")
                                        else:
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, label_axis)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_{label_axis}_{cond_cls}_"
                                                         f"cond_continue_buffer_uncond_{itr}.png")

                                        if args.plot_cond_marginalize:
                                            x_cond_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                      mul_format_lbl_,
                                                                      fix_label_axis=label_axis,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every,
                                                                      marginalize_free_b=True)[0]
                                            if args.return_steps > 0:
                                                x_cond_fresh, fresh_steps = x_cond_fresh
                                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                               mul_format_lbl_, label_axis)
                                                plot_samples(x_cond_fresh,
                                                             f"{args.save_dir}/"
                                                             f"samples_{label_axis}_{cond_cls}_cond_"
                                                             f"continue_buffer_uncond_"
                                                             f"marginalize_{itr}.png")
                                                if args.steps_batch_ind is None:
                                                    for chain_itr in range(fresh_steps.shape[0]):
                                                        plot_samples(fresh_steps[chain_itr],
                                                                     f"{args.save_dir}/"
                                                                     f"chain_{label_axis}_{cond_cls}_"
                                                                     f"cond_continue_buffer_uncond_"
                                                                     f"marginalize_{itr}_itr_{chain_itr}.png")
                                                else:
                                                    plot_samples(fresh_steps,
                                                                 f"{args.save_dir}/"
                                                                 f"chain_{label_axis}_{cond_cls}_cond_"
                                                                 f"continue_buffer_uncond_"
                                                                 f"marginalize_{itr}.png")
                                            else:
                                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                               mul_format_lbl_,
                                                                               label_axis)
                                                plot_samples(x_cond_fresh,
                                                             f"{args.save_dir}/"
                                                             f"samples_{label_axis}_{cond_cls}_cond_"
                                                             f"continue_buffer_uncond_"
                                                             f"marginalize_{itr}.png")

                                if args.plot_cond_marginalize:
                                    x_cond_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                              format_lbl_cond(label_axis, eval_b_init_, cond_cls),
                                                              fix_label_axis=label_axis,
                                                              steps=args.test_k,
                                                              update_buffer=False,
                                                              new_replay_buffer=[], new_y_replay_buffer=[],
                                                              return_steps=args.return_steps,
                                                              steps_batch_ind=args.steps_batch_ind,
                                                              gibbs_steps=args.test_gibbs_steps,
                                                              gibbs_k_steps=args.test_gibbs_k_steps,
                                                              gibbs_n_steps=args.test_gibbs_n_steps,
                                                              transform_every=args.transform_every,
                                                              marginalize_free_b=True)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh, fresh_steps = x_cond_fresh
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       format_lbl_cond(label_axis, eval_b_init_,
                                                                                       cond_cls),
                                                                       label_axis)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_{label_axis}_{cond_cls}_cond_fresh_"
                                                     f"marginalize_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"chain_{label_axis}_{cond_cls}_"
                                                             f"cond_fresh_marginalize_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"chain_{label_axis}_{cond_cls}_cond_fresh_"
                                                         f"marginalize_{itr}.png")
                                    else:
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       format_lbl_cond(label_axis, eval_b_init_,
                                                                                       cond_cls),
                                                                       label_axis)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_{label_axis}_{cond_cls}_cond_fresh_"
                                                     f"marginalize_{itr}.png")

                                x_cond_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                          format_lbl_cond(label_axis, eval_b_init_, cond_cls),
                                                          fix_label_axis=label_axis,
                                                          steps=args.test_k,
                                                          update_buffer=False,
                                                          new_replay_buffer=[], new_y_replay_buffer=[],
                                                          return_steps=args.return_steps,
                                                          steps_batch_ind=args.steps_batch_ind,
                                                          gibbs_steps=args.test_gibbs_steps,
                                                          gibbs_k_steps=args.test_gibbs_k_steps,
                                                          gibbs_n_steps=args.test_gibbs_n_steps,
                                                          transform_every=args.transform_every)[0]
                                if args.return_steps > 0:
                                    x_cond_fresh, fresh_steps = x_cond_fresh
                                    x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                   format_lbl_cond(label_axis, eval_b_init_, cond_cls),
                                                                   label_axis)
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(fresh_steps.shape[0]):
                                            plot_samples(fresh_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"chain_{label_axis}_{cond_cls}_"
                                                         f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(fresh_steps,
                                                     f"{args.save_dir}/"
                                                     f"chain_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                else:
                                    x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                   format_lbl_cond(label_axis, eval_b_init_, cond_cls),
                                                                   label_axis)
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                                if args.plot_ais:

                                    if args.model != "joint":
                                        raise NotImplementedError

                                    x_cond_ais = sample_x_b(ema_logp_net, eval_x_init,
                                                            format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                            fix_label_axis=label_axis,
                                                            steps=args.test_k,
                                                            update_buffer=False,
                                                            new_replay_buffer=[], new_y_replay_buffer=[],
                                                            return_steps=args.return_steps,
                                                            steps_batch_ind=args.steps_batch_ind,
                                                            gibbs_steps=args.test_gibbs_steps,
                                                            gibbs_k_steps=args.test_gibbs_k_steps,
                                                            gibbs_n_steps=args.test_gibbs_n_steps,
                                                            transform_every=args.transform_every,
                                                            anneal=True)[0]
                                    if args.return_steps > 0:
                                        x_cond_ais, ais_steps = x_cond_ais
                                        plot_samples(x_cond_ais,
                                                     f"{args.save_dir}/"
                                                     f"ais_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                        if args.steps_batch_ind:
                                            for chain_itr in range(ais_steps.shape[0]):
                                                plot_samples(ais_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"ais_chain_{label_axis}_{cond_cls}_"
                                                             f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(ais_steps,
                                                         f"{args.save_dir}/"
                                                         f"ais_chain_"
                                                         f"{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                    else:
                                        plot_samples(x_cond_ais,
                                                     f"{args.save_dir}/"
                                                     f"ais_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                                    x_cond_ais2 = sample_x(ema_logp_net, eval_x_init,
                                                           format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                           steps=args.test_k,
                                                           update_buffer=False, new_replay_buffer=[],
                                                           new_y_replay_buffer=[],
                                                           return_steps=args.return_steps,
                                                           steps_batch_ind=args.steps_batch_ind,
                                                           anneal=True)

                                    if args.return_steps > 0:
                                        x_cond_ais2, ais_steps = x_cond_ais2
                                        plot_samples(x_cond_ais2,
                                                     f"{args.save_dir}/"
                                                     f"ais_2samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(ais_steps.shape[0]):
                                                plot_samples(ais_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"ais_2chain_{cond_cls}_"
                                                             f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(ais_steps,
                                                         f"{args.save_dir}/"
                                                         f"ais_2chain{cond_cls}_cond_fresh_{itr}.png")
                                    else:
                                        plot_samples(x_cond_ais2,
                                                     f"{args.save_dir}/"
                                                     f"ais_2samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                                if args.plot_temp:

                                    x_cond_fresh_temp = sample_x_b(ema_logp_net, eval_x_init,
                                                                   format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                                   fix_label_axis=label_axis,
                                                                   steps=args.test_k,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every,
                                                                   temp=True)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh_temp, fresh_steps = x_cond_fresh_temp
                                        plot_samples(x_cond_fresh_temp,
                                                     f"{args.save_dir}/"
                                                     f"temp_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"temp_chain_{label_axis}_{cond_cls}_"
                                                             f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"temp_chain_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                    else:
                                        plot_samples(x_cond_fresh_temp,
                                                     f"{args.save_dir}/"
                                                     f"temp_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                                if args.plot_same:

                                    x_cond_fresh_same = sample_x_b(ema_logp_net,
                                                                   repeat_x(eval_x_init),
                                                                   format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                                   fix_label_axis=label_axis,
                                                                   steps=args.test_k,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh_same, fresh_steps = x_cond_fresh_same
                                        plot_samples(x_cond_fresh_same,
                                                     f"{args.save_dir}/"
                                                     f"same_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"same_chain_{label_axis}_{cond_cls}_"
                                                             f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"same_chain_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                    else:
                                        plot_samples(x_cond_fresh_same,
                                                     f"{args.save_dir}/"
                                                     f"same_samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                                if args.plot_cond_fix_b:

                                    if args.model != "joint":
                                        raise NotImplementedError

                                    # plot fresh conditional samples, fixing b to be random
                                    x_cond_fresh2 = sample_x(ema_logp_net, eval_x_init,
                                                             format_lbl_cond(label_axis, eval_b_init, cond_cls),
                                                             steps=args.test_k,
                                                             update_buffer=False, new_replay_buffer=[],
                                                             new_y_replay_buffer=[],
                                                             return_steps=args.return_steps,
                                                             steps_batch_ind=args.steps_batch_ind)

                                    if args.return_steps > 0:
                                        x_cond_fresh2, fresh_steps = x_cond_fresh2
                                        plot_samples(x_cond_fresh2,
                                                     f"{args.save_dir}/"
                                                     f"2samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"2chain_{label_axis}_{cond_cls}_"
                                                             f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"2chain_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")
                                    else:
                                        plot_samples(x_cond_fresh2,
                                                     f"{args.save_dir}/"
                                                     f"2samples_{label_axis}_{cond_cls}_cond_fresh_{itr}.png")

                    # ================ plot unconditional fresh samples ================

                    if args.plot_uncond_fresh:
                        if args.model == "joint" or args.poj_joint:
                            eval_x_neg_fresh = sample_x_b(ema_logp_net, eval_x_init, eval_b_init,
                                                          update_buffer=False,
                                                          new_replay_buffer=[], new_y_replay_buffer=[],
                                                          return_steps=args.return_steps,
                                                          steps_batch_ind=args.steps_batch_ind,
                                                          steps=args.test_k,
                                                          gibbs_steps=args.test_gibbs_steps,
                                                          gibbs_k_steps=args.test_gibbs_k_steps,
                                                          gibbs_n_steps=args.test_gibbs_n_steps,
                                                          transform_every=args.transform_every)[0]

                            if args.return_steps > 0:
                                eval_x_neg_fresh, eval_x_neg_fresh_steps = eval_x_neg_fresh
                                plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(eval_x_neg_fresh_steps.shape[0]):
                                        plot_samples(eval_x_neg_fresh_steps[chain_itr],
                                                     f"{args.save_dir}/chain_fresh_chain_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(eval_x_neg_fresh_steps, f"{args.save_dir}/chain_fresh_{itr}.png")
                            else:
                                plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")

                            if args.plot_cond_marginalize:
                                eval_x_neg_fresh = uncond_sample_x(ema_logp_net, eval_x_init, steps=args.test_k,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   transform_every=args.transform_every)
                                if args.return_steps > 0:
                                    eval_x_neg_fresh, eval_x_neg_fresh_steps = eval_x_neg_fresh
                                    plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_"
                                                                   f"marginalize_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(eval_x_neg_fresh_steps.shape[0]):
                                            plot_samples(eval_x_neg_fresh_steps[chain_itr],
                                                         f"{args.save_dir}/chain_fresh_"
                                                         f"marginalize_chain_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(eval_x_neg_fresh_steps, f"{args.save_dir}/chain_fresh_"
                                                                             f"marginalize_{itr}.png")
                                else:
                                    plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_"
                                                                   f"marginalize_{itr}.png")

                            if args.plot_cond_continue_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer

                                x_init_buffer_ = replay_buffer_[:args.batch_size]
                                b_init_buffer_ = y_replay_buffer_[:args.batch_size]
                                x_cond_fresh = sample_x_b(ema_logp_net, x_init_buffer_, b_init_buffer_,
                                                          steps=args.test_k,
                                                          update_buffer=False,
                                                          new_replay_buffer=[], new_y_replay_buffer=[],
                                                          return_steps=args.return_steps,
                                                          steps_batch_ind=args.steps_batch_ind,
                                                          gibbs_steps=args.test_gibbs_steps,
                                                          gibbs_k_steps=args.test_gibbs_k_steps,
                                                          gibbs_n_steps=args.test_gibbs_n_steps,
                                                          transform_every=args.transform_every)[0]

                                if args.return_steps > 0:
                                    x_cond_fresh, fresh_steps = x_cond_fresh
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_fresh_"
                                                 f"continue_buffer_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(fresh_steps.shape[0]):
                                            plot_samples(fresh_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"chain_fresh_"
                                                         f"continue_buffer_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(fresh_steps,
                                                     f"{args.save_dir}/"
                                                     f"chain_fresh_"
                                                     f"continue_buffer_{itr}.png")
                                else:
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_fresh_"
                                                 f"continue_buffer_{itr}.png")

                                if args.plot_cond_marginalize:
                                    eval_x_neg_fresh = uncond_sample_x(ema_logp_net, x_init_buffer_,
                                                                       steps=args.test_k,
                                                                       update_buffer=False,
                                                                       new_replay_buffer=[], new_y_replay_buffer=[],
                                                                       return_steps=args.return_steps,
                                                                       steps_batch_ind=args.steps_batch_ind,
                                                                       transform_every=args.transform_every)
                                    if args.return_steps > 0:
                                        eval_x_neg_fresh, eval_x_neg_fresh_steps = eval_x_neg_fresh
                                        plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_"
                                                                       f"continue_buffer_marginalize_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(eval_x_neg_fresh_steps.shape[0]):
                                                plot_samples(eval_x_neg_fresh_steps[chain_itr],
                                                             f"{args.save_dir}/chain_fresh_"
                                                             f"continue_buffer_"
                                                             f"marginalize_chain_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(eval_x_neg_fresh_steps, f"{args.save_dir}/chain_fresh_"
                                                                                 f"continue_buffer_"
                                                                                 f"marginalize_{itr}.png")
                                    else:
                                        plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_"
                                                                       f"continue_buffer_marginalize_{itr}.png")

                            if args.plot_temp:
                                eval_x_neg_fresh_temp = sample_x_b(ema_logp_net, eval_x_init, eval_b_init,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   steps=args.test_k,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every,
                                                                   temp=True)[0]

                                if args.return_steps > 0:
                                    eval_x_neg_fresh_temp, eval_x_neg_fresh_temp_steps = eval_x_neg_fresh_temp
                                    plot_samples(eval_x_neg_fresh_temp, f"{args.save_dir}/temp_samples_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(eval_x_neg_fresh_temp_steps.shape[0]):
                                            plot_samples(eval_x_neg_fresh_temp_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"temp_chain_fresh_chain_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(eval_x_neg_fresh_temp_steps,
                                                     f"{args.save_dir}/temp_chain_fresh_{itr}.png")
                                else:
                                    plot_samples(eval_x_neg_fresh_temp, f"{args.save_dir}/temp_samples_fresh_{itr}.png")

                            if args.plot_same:
                                eval_x_neg_fresh_same = sample_x_b(ema_logp_net,
                                                                   repeat_x(eval_x_init),
                                                                   repeat_x(eval_b_init),
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   steps=args.test_k,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every)[0]

                                if args.return_steps > 0:
                                    eval_x_neg_fresh_same, eval_x_neg_fresh_same_steps = eval_x_neg_fresh_same
                                    plot_samples(eval_x_neg_fresh_same, f"{args.save_dir}/same_samples_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(eval_x_neg_fresh_same_steps.shape[0]):
                                            plot_samples(eval_x_neg_fresh_same_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"same_chain_fresh_chain_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(eval_x_neg_fresh_same_steps,
                                                     f"{args.save_dir}/same_chain_fresh_{itr}.png")
                                else:
                                    plot_samples(eval_x_neg_fresh_same, f"{args.save_dir}/same_samples_fresh_{itr}.png")
                        else:
                            eval_x_neg_fresh = uncond_sample_x(ema_logp_net, eval_x_init, steps=args.test_k,
                                                               update_buffer=False,
                                                               new_replay_buffer=[], new_y_replay_buffer=[],
                                                               return_steps=args.return_steps,
                                                               steps_batch_ind=args.steps_batch_ind,
                                                               transform_every=args.transform_every)
                            if args.return_steps > 0:
                                eval_x_neg_fresh, eval_x_neg_fresh_steps = eval_x_neg_fresh
                                plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(eval_x_neg_fresh_steps.shape[0]):
                                        plot_samples(eval_x_neg_fresh_steps[chain_itr],
                                                     f"{args.save_dir}/chain_fresh_chain_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(eval_x_neg_fresh_steps, f"{args.save_dir}/chain_fresh_{itr}.png")
                            else:
                                plot_samples(eval_x_neg_fresh, f"{args.save_dir}/samples_fresh_{itr}.png")

                    # ================ plot conditioning on two attributes ================
                    if args.data in ["8gaussians_hierarch", "utzappos", "utzappos_old", "celeba", "cub"]:
                        if all_binary:
                            _iterator = product(range(label_dim), range(label_dim))
                        else:
                            _iterator = product(range(label_shape[0]), range(label_shape[1]))
                        for cond_cls0, cond_cls1 in _iterator:
                            if args.data == "utzappos_old":
                                if (cond_cls0, cond_cls1) not in [(2, 11), (2, 12), (2, 13), (2, 15), (2, 17), (0, 1),
                                                                  (0, 2)]:
                                    continue
                            elif args.data == "utzappos":
                                if not check_cond_cls_mul_utzappos(cond_cls0, cond_cls1):
                                    continue
                            elif args.data == "celeba":
                                if not check_cond_cls_mul_celeba(cond_cls0, cond_cls1):
                                    continue
                            elif args.data == "cub":
                                if not check_cond_cls_mul_cub(cond_cls0, cond_cls1):
                                    continue

                            if all_binary:
                                _mask = torch.logical_and(labels_b[:, cond_cls0] == 1, labels_b[:, cond_cls1] == 1)
                            else:
                                _mask = torch.logical_and(labels_b[:, 0] == cond_cls0, labels_b[:, 1] == cond_cls1)
                            plot_samples(
                                eval_x_neg[_mask],
                                f"{args.save_dir}/samples_mul_{cond_cls0}_{cond_cls1}_{itr}.png")

                            if args.plot_cond_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                cond_buffer_mask_ = torch.logical_and(y_replay_buffer_[:, cond_cls0] == 1,
                                                                      y_replay_buffer_[:, cond_cls1] == 1)
                                plot_samples(replay_buffer_[cond_buffer_mask_][:args.batch_size],
                                             f"{args.save_dir}/buffer_mul_{cond_cls0}_{cond_cls1}_{itr}.png")

                            batch_size_ = args.batch_size * args.plot_cond_filter

                            if args.plot_cond_filter > 1:
                                eval_x_init_, eval_b_init_ = init_x_random(batch_size_), init_b_random(batch_size_)
                            else:
                                eval_x_init_, eval_b_init_ = eval_x_init, eval_b_init

                            # compose multiple times to fix different axes
                            if all_binary:
                                mul_format_lbl = format_lbl_cond(cond_cls0, eval_b_init_, 1)
                                mul_format_lbl = format_lbl_cond(cond_cls1, mul_format_lbl, 1)
                                fix_label_axis_ = [cond_cls0, cond_cls1]
                            else:
                                mul_format_lbl = format_lbl_cond(0, eval_b_init_, cond_cls0)
                                mul_format_lbl = format_lbl_cond(1, mul_format_lbl, cond_cls1)
                                fix_label_axis_ = [0, 1]

                            if args.plot_cond_continue_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                cond_buffer_mask_ = torch.logical_and(y_replay_buffer_[:, cond_cls0] == 1,
                                                                      y_replay_buffer_[:, cond_cls1] == 1)
                                x_init_buffer_ = replay_buffer_[cond_buffer_mask_][:batch_size_]
                                mul_format_lbl_ = y_replay_buffer_[cond_buffer_mask_][:batch_size_]
                                if x_init_buffer_.shape[0] > 0:
                                    x_cond_mul_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                  mul_format_lbl_,
                                                                  fix_label_axis=fix_label_axis_,
                                                                  steps=args.test_k,
                                                                  update_buffer=False,
                                                                  new_replay_buffer=[], new_y_replay_buffer=[],
                                                                  return_steps=args.return_steps,
                                                                  steps_batch_ind=args.steps_batch_ind,
                                                                  gibbs_steps=args.test_gibbs_steps,
                                                                  gibbs_k_steps=args.test_gibbs_k_steps,
                                                                  gibbs_n_steps=args.test_gibbs_n_steps,
                                                                  transform_every=args.transform_every)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                     f"cond_continue_buffer_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                             f"cond_continue_buffer_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"chain_mul_{cond_cls0}_{cond_cls1}_"
                                                         f"cond_continue_buffer_{itr}.png")

                                        if args.eval_cond_acc:
                                            eval_cond_acc_print_hist(ema_logp_net, x_cond_fresh, mul_format_lbl_,
                                                                     fix_label_axis_,
                                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                                     f"cond_continue_buffer_{itr}")
                                    else:
                                        x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_mul_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                     f"cond_continue_buffer_{itr}.png")

                                        if args.eval_cond_acc:
                                            eval_cond_acc_print_hist(ema_logp_net, x_cond_mul_fresh, mul_format_lbl_,
                                                                     fix_label_axis_,
                                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                                     f"cond_continue_buffer_{itr}")

                                    if args.plot_cond_marginalize:
                                        x_cond_mul_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                      mul_format_lbl_,
                                                                      fix_label_axis=fix_label_axis_,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every,
                                                                      marginalize_free_b=True)[0]
                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_cond_continue_buffer_"
                                                         f"marginalize_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                                 f"cond_continue_buffer_"
                                                                 f"marginalize_{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_mul_{cond_cls0}_{cond_cls1}_cond_continue_buffer_"
                                                             f"marginalize_{itr}.png")

                                            if args.eval_cond_acc:
                                                eval_cond_acc_print_hist(ema_logp_net, x_cond_fresh, mul_format_lbl_,
                                                                         fix_label_axis_,
                                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                                         f"cond_continue_buffer_"
                                                                         f"marginalize_{itr}")
                                        else:
                                            x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                               mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_mul_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_cond_continue_buffer_"
                                                         f"marginalize_{itr}.png")

                                            if args.eval_cond_acc:
                                                eval_cond_acc_print_hist(ema_logp_net, x_cond_mul_fresh,
                                                                         mul_format_lbl_, fix_label_axis_,
                                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                                         f"cond_continue_buffer_"
                                                                         f"marginalize_{itr}")

                            if args.plot_cond_continue_buffer_uncond:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                x_init_buffer_ = replay_buffer_[:batch_size_]
                                mul_format_lbl_ = y_replay_buffer_[:batch_size_]
                                if x_init_buffer_.shape[0] > 0:
                                    x_cond_mul_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                  mul_format_lbl_,
                                                                  fix_label_axis=fix_label_axis_,
                                                                  steps=args.test_k,
                                                                  update_buffer=False,
                                                                  new_replay_buffer=[], new_y_replay_buffer=[],
                                                                  return_steps=args.return_steps,
                                                                  steps_batch_ind=args.steps_batch_ind,
                                                                  gibbs_steps=args.test_gibbs_steps,
                                                                  gibbs_k_steps=args.test_gibbs_k_steps,
                                                                  gibbs_n_steps=args.test_gibbs_n_steps,
                                                                  transform_every=args.transform_every)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                     f"cond_continue_buffer_uncond_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                             f"cond_continue_buffer_uncond_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"chain_mul_{cond_cls0}_{cond_cls1}_"
                                                         f"cond_continue_buffer_uncond_{itr}.png")
                                    else:
                                        x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_mul_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_mul_{cond_cls0}_{cond_cls1}_"
                                                     f"cond_continue_buffer_{itr}.png")

                                    if args.plot_cond_marginalize:
                                        x_cond_mul_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                      mul_format_lbl_,
                                                                      fix_label_axis=fix_label_axis_,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every,
                                                                      marginalize_free_b=True)[0]
                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_cond_"
                                                         f"continue_buffer_uncond_"
                                                         f"marginalize_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                                 f"cond_continue_buffer_uncond_"
                                                                 f"marginalize_{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_mul_{cond_cls0}_{cond_cls1}_cond_"
                                                             f"continue_buffer_uncond_"
                                                             f"marginalize_{itr}.png")
                                        else:
                                            x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                               mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_mul_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_mul_{cond_cls0}_{cond_cls1}_cond_"
                                                         f"continue_buffer_uncond_"
                                                         f"marginalize_{itr}.png")

                            if args.plot_cond_marginalize:
                                x_cond_mul_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                              mul_format_lbl,
                                                              fix_label_axis=fix_label_axis_,
                                                              steps=args.test_k,
                                                              update_buffer=False,
                                                              new_replay_buffer=[], new_y_replay_buffer=[],
                                                              return_steps=args.return_steps,
                                                              steps_batch_ind=args.steps_batch_ind,
                                                              gibbs_steps=args.test_gibbs_steps,
                                                              gibbs_k_steps=args.test_gibbs_k_steps,
                                                              gibbs_n_steps=args.test_gibbs_n_steps,
                                                              transform_every=args.transform_every,
                                                              marginalize_free_b=True)[0]
                                if args.return_steps > 0:
                                    x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                    x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                   mul_format_lbl, fix_label_axis_)
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_"
                                                 f"marginalize_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(fresh_steps.shape[0]):
                                            plot_samples(fresh_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                         f"cond_fresh_marginalize_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(fresh_steps,
                                                     f"{args.save_dir}/"
                                                     f"chain_mul_{cond_cls0}_{cond_cls1}_cond_fresh_"
                                                     f"marginalize_{itr}.png")

                                    if args.eval_cond_acc:
                                        eval_cond_acc_print_hist(ema_logp_net, x_cond_fresh, mul_format_lbl, 
                                                                 fix_label_axis_,
                                                                 f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_"
                                                                 f"marginalize_{itr}")
                                else:
                                    x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                       mul_format_lbl, fix_label_axis_)
                                    plot_samples(x_cond_mul_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_"
                                                 f"marginalize_{itr}.png")

                                    if args.eval_cond_acc:
                                        eval_cond_acc_print_hist(ema_logp_net, x_cond_mul_fresh, mul_format_lbl,
                                                                 fix_label_axis_,
                                                                 f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_"
                                                                 f"marginalize_{itr}")

                            x_cond_mul_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                          mul_format_lbl,
                                                          fix_label_axis=fix_label_axis_,
                                                          steps=args.test_k,
                                                          update_buffer=False,
                                                          new_replay_buffer=[], new_y_replay_buffer=[],
                                                          return_steps=args.return_steps,
                                                          steps_batch_ind=args.steps_batch_ind,
                                                          gibbs_steps=args.test_gibbs_steps,
                                                          gibbs_k_steps=args.test_gibbs_k_steps,
                                                          gibbs_n_steps=args.test_gibbs_n_steps,
                                                          transform_every=args.transform_every)[0]
                            if args.return_steps > 0:
                                x_cond_fresh, fresh_steps = x_cond_mul_fresh
                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                               mul_format_lbl, fix_label_axis_)
                                plot_samples(x_cond_fresh,
                                             f"{args.save_dir}/"
                                             f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(fresh_steps.shape[0]):
                                        plot_samples(fresh_steps[chain_itr],
                                                     f"{args.save_dir}/"
                                                     f"chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                     f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(fresh_steps,
                                                 f"{args.save_dir}/"
                                                 f"chain_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")

                                if args.eval_cond_acc:
                                    eval_cond_acc_print_hist(ema_logp_net, x_cond_fresh, mul_format_lbl,
                                                             fix_label_axis_,
                                                             f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}")
                            else:
                                x_cond_mul_fresh = filter_post_hoc(ema_logp_net, x_cond_mul_fresh,
                                                                   mul_format_lbl, fix_label_axis_)
                                plot_samples(x_cond_mul_fresh,
                                             f"{args.save_dir}/"
                                             f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")

                                if args.eval_cond_acc:
                                    eval_cond_acc_print_hist(ema_logp_net, x_cond_mul_fresh, mul_format_lbl,
                                                             fix_label_axis_,
                                                             f"samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}")

                            if args.plot_temp:
                                x_cond_mul_fresh_temp = sample_x_b(ema_logp_net, eval_x_init,
                                                                   mul_format_lbl,
                                                                   fix_label_axis=fix_label_axis_,
                                                                   steps=args.test_k,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every,
                                                                   temp=True)[0]
                                if args.return_steps > 0:
                                    x_cond_mul_fresh_temp, cond_mul_fresh_temp_steps = x_cond_mul_fresh_temp
                                    plot_samples(x_cond_mul_fresh_temp,
                                                 f"{args.save_dir}/"
                                                 f"temp_samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(cond_mul_fresh_temp_steps.shape[0]):
                                            plot_samples(cond_mul_fresh_temp_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"temp_chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                         f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(cond_mul_fresh_temp_steps,
                                                     f"{args.save_dir}/"
                                                     f"temp_chain_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")
                                else:
                                    plot_samples(x_cond_mul_fresh_temp,
                                                 f"{args.save_dir}/"
                                                 f"temp_samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")

                            if args.plot_same:
                                x_cond_mul_fresh_same = sample_x_b(ema_logp_net,
                                                                   repeat_x(eval_x_init),
                                                                   mul_format_lbl,
                                                                   fix_label_axis=fix_label_axis_,
                                                                   steps=args.test_k,
                                                                   update_buffer=False,
                                                                   new_replay_buffer=[], new_y_replay_buffer=[],
                                                                   return_steps=args.return_steps,
                                                                   steps_batch_ind=args.steps_batch_ind,
                                                                   gibbs_steps=args.test_gibbs_steps,
                                                                   gibbs_k_steps=args.test_gibbs_k_steps,
                                                                   gibbs_n_steps=args.test_gibbs_n_steps,
                                                                   transform_every=args.transform_every)[0]
                                if args.return_steps > 0:
                                    x_cond_mul_fresh_same, cond_mul_fresh_same_steps = x_cond_mul_fresh_same
                                    plot_samples(x_cond_mul_fresh_same,
                                                 f"{args.save_dir}/"
                                                 f"same_samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(cond_mul_fresh_same_steps.shape[0]):
                                            plot_samples(cond_mul_fresh_same_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"same_chain_mul_chain_{cond_cls0}_{cond_cls1}_"
                                                         f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(cond_mul_fresh_same_steps,
                                                     f"{args.save_dir}/"
                                                     f"same_chain_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")
                                else:
                                    plot_samples(x_cond_mul_fresh_same,
                                                 f"{args.save_dir}/"
                                                 f"same_samples_mul_{cond_cls0}_{cond_cls1}_cond_fresh_{itr}.png")

                    # ================ plot conditioning on all attributes ================

                    if label_shape != (-1,):
                        if all_binary:
                            prod_cond_cls = product(*(range(2) for _ in range(label_dim)))
                        else:
                            prod_cond_cls = product(*(range(i) for i in label_shape))

                        def next_cond_cls():
                            label_combo = next(prod_cond_cls)
                            cls_lbl = one_hot(torch.tensor(label_combo).to(args.device)[None])
                            return cls_lbl.repeat_interleave(eval_x_init.shape[0], 0)
                    else:
                        next_cond_cls = None

                    if args.data == "utzappos_old":
                        _iterator = range(1)  # just one example to plot
                    else:
                        if all_binary:
                            _iterator = range(2 ** label_dim)
                        else:
                            _iterator = range(all_labels + (label_dim == 1))

                    for cond_cls in _iterator:

                        if all_labels > 10 and not args.data == "utzappos_old":
                            # don't do too many conditionals!
                            break

                        if label_shape == (-1,):
                            next_cond_cls_ = None  # for linting

                            plot_samples(eval_x_neg[labels_b.squeeze() == cond_cls],
                                         f"{args.save_dir}/samples{cond_cls}_{itr}.png")

                            if args.plot_cond_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    raise NotImplementedError
                                plot_samples(replay_buffer[y_replay_buffer == cond_cls],
                                             f"{args.save_dir}/buffer{cond_cls}_{itr}.png")

                            if args.plot_cond_init_buffer:
                                assert args.model == "joint"
                                x_cond = sample_x(ema_logp_net, eval_x_init,
                                                  format_lbl(bs=eval_y.shape[0], cls=cond_cls, device=args.device),
                                                  update_buffer=False,
                                                  return_steps=args.return_steps,
                                                  steps_batch_ind=args.steps_batch_ind)
                                if args.return_steps > 0:
                                    x_cond, cond_steps = x_cond
                                    plot_samples(x_cond, f"{args.save_dir}/samples{cond_cls}_cond_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(cond_steps.shape[0]):
                                            plot_samples(cond_steps[chain_itr],
                                                         f"{args.save_dir}/chain_"
                                                         f"{cond_cls}_cond_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(cond_steps, f"{args.save_dir}/chain{cond_cls}_cond_{itr}.png")
                                else:
                                    plot_samples(x_cond, f"{args.save_dir}/samples{cond_cls}_cond_{itr}.png")

                            x_cond_fresh = sample_x(ema_logp_net, eval_x_init,
                                                    format_lbl(bs=eval_y.shape[0], cls=cond_cls,
                                                               device=args.device),
                                                    steps=args.test_k,
                                                    update_buffer=False,
                                                    new_replay_buffer=[], new_y_replay_buffer=[],
                                                    return_steps=args.return_steps,
                                                    steps_batch_ind=args.steps_batch_ind)

                        else:

                            if args.data == "utzappos_old":
                                label_combo_ = (2, 17, 272)  # Shoes/Sneakers and Athletic Shoes/Nike
                                cls_lbl_ = one_hot(torch.tensor(label_combo_).to(args.device)[None])
                                next_cond_cls_ = cls_lbl_.repeat_interleave(init_x.shape[0], 0)
                                cond_cls = 0  # idk what the number is, and it doesn't matter
                            else:
                                next_cond_cls_ = next_cond_cls()

                            if args.plot_cond_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                # next_cond_cls_ is repeated along batch dimension, so just index one of them
                                cond_buffer_mask_ = (y_replay_buffer_ == label(next_cond_cls_[0][None])).all(-1)
                                plot_samples(replay_buffer_[cond_buffer_mask_],
                                             f"{args.save_dir}/buffer{cond_cls}_{itr}.png")

                            if args.model == "poj" and not args.poj_joint:
                                _disc_mask = (onehot_poj_logits(eval_b) == next_cond_cls_).all(1)
                                if unif_label_shape is not None:
                                    # if uniform label shape, we have 2 dimensions to call .all() over
                                    _disc_mask = _disc_mask.all(1)
                            else:
                                _mask_shape = (eval_b.shape[0], label_dim)
                                _disc_mask = (eval_b.view(*_mask_shape) == next_cond_cls_.view(*_mask_shape)).all(1)
                            plot_samples(eval_x_neg[_disc_mask],
                                         f"{args.save_dir}/samples{cond_cls}_{itr}.png")

                            x_cond_fresh = sample_x(ema_logp_net, eval_x_init, next_cond_cls_,
                                                    steps=args.test_k,
                                                    update_buffer=False,
                                                    new_replay_buffer=[], new_y_replay_buffer=[],
                                                    return_steps=args.return_steps,
                                                    steps_batch_ind=args.steps_batch_ind)

                        if args.return_steps > 0:
                            x_cond_fresh, fresh_steps = x_cond_fresh
                            plot_samples(x_cond_fresh, f"{args.save_dir}/samples{cond_cls}_cond_fresh_{itr}.png")
                            if args.steps_batch_ind is None:
                                for chain_itr in range(fresh_steps.shape[0]):
                                    plot_samples(fresh_steps[chain_itr],
                                                 f"{args.save_dir}/chain_"
                                                 f"{cond_cls}_cond_fresh_{itr}_itr_{chain_itr}.png")
                            else:
                                plot_samples(fresh_steps, f"{args.save_dir}/chain{cond_cls}_cond_fresh_{itr}.png")
                        else:
                            plot_samples(x_cond_fresh, f"{args.save_dir}/samples{cond_cls}_cond_fresh_{itr}.png")

                        if args.plot_ais:

                            assert args.model == "joint"

                            if args.data in ("utzappos", "celeba", "cub"):
                                raise NotImplementedError

                            if label_shape != (-1,):
                                x_cond_ais = sample_x(ema_logp_net, eval_x_init, next_cond_cls_,
                                                      steps=args.test_k,
                                                      anneal=True,
                                                      update_buffer=False, new_replay_buffer=[], new_y_replay_buffer=[],
                                                      return_steps=args.return_steps,
                                                      steps_batch_ind=args.steps_batch_ind)
                            else:
                                x_cond_ais = sample_x(ema_logp_net, eval_x_init,
                                                      format_lbl(bs=eval_y.shape[0], cls=cond_cls, device=args.device),
                                                      steps=args.test_k,
                                                      anneal=True,
                                                      update_buffer=False, new_replay_buffer=[], new_y_replay_buffer=[],
                                                      return_steps=args.return_steps,
                                                      steps_batch_ind=args.steps_batch_ind)
                            if args.return_steps > 0:
                                x_cond_ais, ais_steps = x_cond_ais
                                plot_samples(x_cond_ais, f"{args.save_dir}/samples{cond_cls}_cond_ais_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(ais_steps.shape[0]):
                                        plot_samples(ais_steps[chain_itr],
                                                     f"{args.save_dir}/chain_"
                                                     f"{cond_cls}_cond_ais_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(ais_steps, f"{args.save_dir}/chain{cond_cls}_cond_ais_{itr}.png")
                            else:
                                plot_samples(x_cond_ais, f"{args.save_dir}/samples{cond_cls}_cond_ais_{itr}.png")

                        if args.plot_temp:

                            if args.data in ("utzappos", "celeba", "cub"):
                                raise NotImplementedError

                            if label_shape != (-1,):
                                x_cond_temp = sample_x(ema_logp_net, eval_x_init, next_cond_cls_,
                                                       steps=args.test_k,
                                                       temp=True,
                                                       update_buffer=False, new_replay_buffer=[],
                                                       new_y_replay_buffer=[],
                                                       return_steps=args.return_steps,
                                                       steps_batch_ind=args.steps_batch_ind)
                            else:
                                x_cond_temp = sample_x(ema_logp_net, eval_x_init,
                                                       format_lbl(bs=eval_y.shape[0], cls=cond_cls, device=args.device),
                                                       steps=args.test_k,
                                                       temp=True,
                                                       update_buffer=False, new_replay_buffer=[],
                                                       new_y_replay_buffer=[],
                                                       return_steps=args.return_steps,
                                                       steps_batch_ind=args.steps_batch_ind)
                            if args.return_steps > 0:
                                x_cond_temp, temp_steps = x_cond_temp
                                plot_samples(x_cond_temp, f"{args.save_dir}/temp_samples{cond_cls}_cond_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(temp_steps.shape[0]):
                                        plot_samples(temp_steps[chain_itr],
                                                     f"{args.save_dir}/temp_chain_"
                                                     f"{cond_cls}_cond_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(temp_steps, f"{args.save_dir}/temp_chain{cond_cls}_cond_{itr}.png")
                            else:
                                plot_samples(x_cond_temp, f"{args.save_dir}/temp_samples{cond_cls}_cond_{itr}.png")

                        if args.plot_same:

                            if args.data in ("utzappos", "celeba", "cub"):
                                raise NotImplementedError

                            if label_shape != (-1,):
                                x_cond_same = sample_x(ema_logp_net,
                                                       repeat_x(eval_x_init),
                                                       next_cond_cls_,
                                                       steps=args.test_k,
                                                       update_buffer=False, new_replay_buffer=[],
                                                       new_y_replay_buffer=[],
                                                       return_steps=args.return_steps,
                                                       steps_batch_ind=args.steps_batch_ind)
                            else:
                                x_cond_same = sample_x(ema_logp_net,
                                                       repeat_x(eval_x_init),
                                                       format_lbl(bs=eval_y.shape[0], cls=cond_cls, device=args.device),
                                                       steps=args.test_k,
                                                       update_buffer=False, new_replay_buffer=[],
                                                       new_y_replay_buffer=[],
                                                       return_steps=args.return_steps,
                                                       steps_batch_ind=args.steps_batch_ind)
                            if args.return_steps > 0:
                                x_cond_same, same_steps = x_cond_same
                                plot_samples(x_cond_same, f"{args.save_dir}/same_samples{cond_cls}_cond_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(same_steps.shape[0]):
                                        plot_samples(same_steps[chain_itr],
                                                     f"{args.save_dir}/same_chain_"
                                                     f"{cond_cls}_cond_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(same_steps, f"{args.save_dir}/same_chain{cond_cls}_cond_{itr}.png")
                            else:
                                plot_samples(x_cond_same, f"{args.save_dir}/same_samples{cond_cls}_cond_{itr}.png")

                    # ================ CUSTOM ATTRIBUTE PLOTTING ================

                    if args.eval is not None and args.data in ("celeba", "utzappos"):

                        # convert keys from names to indices
                        for celeba_cls_combo in custom_cls_combos:
                            celeba_cls_indices = [(dset_label_info[celeba_cls_name][0], celeba_cls_val)
                                                  for celeba_cls_name, celeba_cls_val in celeba_cls_combo.items()]
                            _mask = reduce(torch.logical_and,
                                           map(partial(cond_attributes_from_labels, labels_b), celeba_cls_indices))

                            batch_size_ = args.batch_size * args.plot_cond_filter

                            if args.plot_cond_filter > 1:
                                eval_x_init_, eval_b_init_ = init_x_random(batch_size_), init_b_random(batch_size_)
                            else:
                                eval_x_init_, eval_b_init_ = eval_x_init, eval_b_init

                            mul_format_lbl = eval_b_init
                            for celeba_cls_index, celeba_cls_val in celeba_cls_indices:
                                mul_format_lbl = format_lbl_cond(celeba_cls_index, eval_b_init_, celeba_cls_val)
                            fix_label_axis_ = list(map(itemgetter(0), celeba_cls_indices))

                            cls_descriptor_str = "_".join(map(str,
                                                              [val for pair in celeba_cls_indices for val in pair]))

                            plot_samples(eval_x_neg[_mask],
                                         f"{args.save_dir}/samples_custom_{cls_descriptor_str}_{itr}.png")

                            if args.plot_cond_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                cond_buffer_mask_ = reduce(torch.logical_and,
                                                           map(partial(cond_attributes_from_labels, y_replay_buffer_),
                                                               celeba_cls_indices))
                                plot_samples(replay_buffer_[cond_buffer_mask_][:args.batch_size],
                                             f"{args.save_dir}/buffer_custom_{cls_descriptor_str}_{itr}.png")

                            if args.plot_cond_continue_buffer:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                cond_buffer_mask_ = reduce(torch.logical_and,
                                                           map(partial(cond_attributes_from_labels,
                                                                       y_replay_buffer_),
                                                               celeba_cls_indices))
                                x_init_buffer_ = replay_buffer_[cond_buffer_mask_][:batch_size_]
                                mul_format_lbl_ = y_replay_buffer_[cond_buffer_mask_][:batch_size_]

                                if x_init_buffer_.shape[0] > 0:
                                    x_cond_custom_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                     mul_format_lbl_,
                                                                     fix_label_axis=fix_label_axis_,
                                                                     steps=args.test_k,
                                                                     update_buffer=False,
                                                                     new_replay_buffer=[], new_y_replay_buffer=[],
                                                                     return_steps=args.return_steps,
                                                                     steps_batch_ind=args.steps_batch_ind,
                                                                     gibbs_steps=args.test_gibbs_steps,
                                                                     gibbs_k_steps=args.test_gibbs_k_steps,
                                                                     gibbs_n_steps=args.test_gibbs_n_steps,
                                                                     transform_every=args.transform_every)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_custom_{cls_descriptor_str}_"
                                                     f"cond_continue_buffer_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"chain_custom_chain_{cls_descriptor_str}_"
                                                             f"cond_continue_buffer_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"chain_custom_{cls_descriptor_str}_"
                                                         f"cond_continue_buffer_{itr}.png")
                                    else:
                                        x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                              mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_custom_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_custom_{cls_descriptor_str}_"
                                                     f"cond_continue_buffer_{itr}.png")

                                    if args.plot_cond_marginalize:
                                        x_cond_custom_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                         mul_format_lbl_,
                                                                         fix_label_axis=fix_label_axis_,
                                                                         steps=args.test_k,
                                                                         update_buffer=False,
                                                                         new_replay_buffer=[], new_y_replay_buffer=[],
                                                                         return_steps=args.return_steps,
                                                                         steps_batch_ind=args.steps_batch_ind,
                                                                         gibbs_steps=args.test_gibbs_steps,
                                                                         gibbs_k_steps=args.test_gibbs_k_steps,
                                                                         gibbs_n_steps=args.test_gibbs_n_steps,
                                                                         transform_every=args.transform_every,
                                                                         marginalize_free_b=True)[0]
                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_custom_{cls_descriptor_str}_cond_continue_buffer_"
                                                         f"marginalize_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_custom_chain_{cls_descriptor_str}_"
                                                                 f"cond_continue_buffer_"
                                                                 f"marginalize_{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_custom_{cls_descriptor_str}_cond_continue_buffer_"
                                                             f"marginalize_{itr}.png")
                                        else:
                                            x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                                  mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_custom_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_custom_{cls_descriptor_str}_cond_continue_buffer_"
                                                         f"marginalize_{itr}.png")

                            if args.plot_cond_continue_buffer_uncond:
                                if isinstance(replay_buffer, ReplayBuffer):
                                    replay_buffer_ = replay_buffer._storage
                                    y_replay_buffer_ = y_replay_buffer._storage
                                else:
                                    replay_buffer_ = replay_buffer
                                    y_replay_buffer_ = y_replay_buffer
                                x_init_buffer_ = replay_buffer_[:batch_size_]
                                mul_format_lbl_ = y_replay_buffer_[:batch_size_]

                                if x_init_buffer_.shape[0] > 0:
                                    x_cond_custom_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                     mul_format_lbl_,
                                                                     fix_label_axis=fix_label_axis_,
                                                                     steps=args.test_k,
                                                                     update_buffer=False,
                                                                     new_replay_buffer=[], new_y_replay_buffer=[],
                                                                     return_steps=args.return_steps,
                                                                     steps_batch_ind=args.steps_batch_ind,
                                                                     gibbs_steps=args.test_gibbs_steps,
                                                                     gibbs_k_steps=args.test_gibbs_k_steps,
                                                                     gibbs_n_steps=args.test_gibbs_n_steps,
                                                                     transform_every=args.transform_every)[0]
                                    if args.return_steps > 0:
                                        x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                        x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                       mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_custom_{cls_descriptor_str}_"
                                                     f"cond_continue_buffer_uncond_{itr}.png")
                                        if args.steps_batch_ind is None:
                                            for chain_itr in range(fresh_steps.shape[0]):
                                                plot_samples(fresh_steps[chain_itr],
                                                             f"{args.save_dir}/"
                                                             f"chain_custom_chain_{cls_descriptor_str}_"
                                                             f"cond_continue_buffer_uncond_{itr}_itr_{chain_itr}.png")
                                        else:
                                            plot_samples(fresh_steps,
                                                         f"{args.save_dir}/"
                                                         f"chain_custom_{cls_descriptor_str}_"
                                                         f"cond_continue_buffer_uncond_{itr}.png")
                                    else:
                                        x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                              mul_format_lbl_, fix_label_axis_)
                                        plot_samples(x_cond_custom_fresh,
                                                     f"{args.save_dir}/"
                                                     f"samples_custom_{cls_descriptor_str}_"
                                                     f"cond_continue_buffer_uncond_{itr}.png")

                                    if args.plot_cond_marginalize:
                                        x_cond_custom_fresh = sample_x_b(ema_logp_net, x_init_buffer_,
                                                                         mul_format_lbl_,
                                                                         fix_label_axis=fix_label_axis_,
                                                                         steps=args.test_k,
                                                                         update_buffer=False,
                                                                         new_replay_buffer=[], new_y_replay_buffer=[],
                                                                         return_steps=args.return_steps,
                                                                         steps_batch_ind=args.steps_batch_ind,
                                                                         gibbs_steps=args.test_gibbs_steps,
                                                                         gibbs_k_steps=args.test_gibbs_k_steps,
                                                                         gibbs_n_steps=args.test_gibbs_n_steps,
                                                                         transform_every=args.transform_every,
                                                                         marginalize_free_b=True)[0]
                                        if args.return_steps > 0:
                                            x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                            x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                           mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_custom_{cls_descriptor_str}_cond_"
                                                         f"continue_buffer_uncond_"
                                                         f"marginalize_{itr}.png")
                                            if args.steps_batch_ind is None:
                                                for chain_itr in range(fresh_steps.shape[0]):
                                                    plot_samples(fresh_steps[chain_itr],
                                                                 f"{args.save_dir}/"
                                                                 f"chain_custom_chain_{cls_descriptor_str}_"
                                                                 f"cond_continue_buffer_uncond_"
                                                                 f"marginalize_{itr}_itr_{chain_itr}.png")
                                            else:
                                                plot_samples(fresh_steps,
                                                             f"{args.save_dir}/"
                                                             f"chain_custom_{cls_descriptor_str}_cond_"
                                                             f"continue_buffer_uncond_"
                                                             f"marginalize_{itr}.png")
                                        else:
                                            x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                                  mul_format_lbl_, fix_label_axis_)
                                            plot_samples(x_cond_custom_fresh,
                                                         f"{args.save_dir}/"
                                                         f"samples_custom_{cls_descriptor_str}_cond_"
                                                         f"continue_buffer_uncond_"
                                                         f"marginalize_{itr}.png")

                            if args.plot_cond_marginalize:
                                x_cond_custom_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                                 mul_format_lbl,
                                                                 fix_label_axis=fix_label_axis_,
                                                                 steps=args.test_k,
                                                                 update_buffer=False,
                                                                 new_replay_buffer=[], new_y_replay_buffer=[],
                                                                 return_steps=args.return_steps,
                                                                 steps_batch_ind=args.steps_batch_ind,
                                                                 gibbs_steps=args.test_gibbs_steps,
                                                                 gibbs_k_steps=args.test_gibbs_k_steps,
                                                                 gibbs_n_steps=args.test_gibbs_n_steps,
                                                                 transform_every=args.transform_every,
                                                                 marginalize_free_b=True)[0]
                                if args.return_steps > 0:
                                    x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                    x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                                   mul_format_lbl, fix_label_axis_)
                                    plot_samples(x_cond_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_custom_{cls_descriptor_str}_cond_fresh_"
                                                 f"marginalize_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(fresh_steps.shape[0]):
                                            plot_samples(fresh_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"chain_custom_chain_{cls_descriptor_str}_"
                                                         f"cond_fresh_marginalize_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(fresh_steps,
                                                     f"{args.save_dir}/"
                                                     f"chain_custom_{cls_descriptor_str}_cond_fresh_"
                                                     f"marginalize_{itr}.png")
                                else:
                                    x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                          mul_format_lbl, fix_label_axis_)
                                    plot_samples(x_cond_custom_fresh,
                                                 f"{args.save_dir}/"
                                                 f"samples_custom_{cls_descriptor_str}_cond_fresh_"
                                                 f"marginalize_{itr}.png")

                            x_cond_custom_fresh = sample_x_b(ema_logp_net, eval_x_init_,
                                                             mul_format_lbl,
                                                             fix_label_axis=fix_label_axis_,
                                                             steps=args.test_k,
                                                             update_buffer=False,
                                                             new_replay_buffer=[], new_y_replay_buffer=[],
                                                             return_steps=args.return_steps,
                                                             steps_batch_ind=args.steps_batch_ind,
                                                             gibbs_steps=args.test_gibbs_steps,
                                                             gibbs_k_steps=args.test_gibbs_k_steps,
                                                             gibbs_n_steps=args.test_gibbs_n_steps,
                                                             transform_every=args.transform_every)[0]
                            if args.return_steps > 0:
                                x_cond_fresh, fresh_steps = x_cond_custom_fresh
                                x_cond_fresh = filter_post_hoc(ema_logp_net, x_cond_fresh,
                                                               mul_format_lbl, fix_label_axis_)
                                plot_samples(x_cond_fresh,
                                             f"{args.save_dir}/"
                                             f"samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                                if args.steps_batch_ind is None:
                                    for chain_itr in range(fresh_steps.shape[0]):
                                        plot_samples(fresh_steps[chain_itr],
                                                     f"{args.save_dir}/"
                                                     f"chain_custom_chain_{cls_descriptor_str}_"
                                                     f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                else:
                                    plot_samples(fresh_steps,
                                                 f"{args.save_dir}/"
                                                 f"chain_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                            else:
                                x_cond_custom_fresh = filter_post_hoc(ema_logp_net, x_cond_custom_fresh,
                                                                      mul_format_lbl, fix_label_axis_)
                                plot_samples(x_cond_custom_fresh,
                                             f"{args.save_dir}/"
                                             f"samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")

                            if args.plot_temp:
                                x_cond_custom_fresh_temp = sample_x_b(ema_logp_net, eval_x_init,
                                                                      mul_format_lbl,
                                                                      fix_label_axis=fix_label_axis_,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every,
                                                                      temp=True)[0]
                                if args.return_steps > 0:
                                    x_cond_custom_fresh_temp, cond_custom_fresh_temp_steps = x_cond_custom_fresh_temp
                                    plot_samples(x_cond_custom_fresh_temp,
                                                 f"{args.save_dir}/"
                                                 f"temp_samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(cond_custom_fresh_temp_steps.shape[0]):
                                            plot_samples(cond_custom_fresh_temp_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"temp_chain_custom_chain_{cls_descriptor_str}_"
                                                         f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(cond_custom_fresh_temp_steps,
                                                     f"{args.save_dir}/"
                                                     f"temp_chain_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                                else:
                                    plot_samples(x_cond_custom_fresh_temp,
                                                 f"{args.save_dir}/"
                                                 f"temp_samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")

                            if args.plot_same:
                                x_cond_custom_fresh_same = sample_x_b(ema_logp_net,
                                                                      repeat_x(eval_x_init),
                                                                      mul_format_lbl,
                                                                      fix_label_axis=fix_label_axis_,
                                                                      steps=args.test_k,
                                                                      update_buffer=False,
                                                                      new_replay_buffer=[], new_y_replay_buffer=[],
                                                                      return_steps=args.return_steps,
                                                                      steps_batch_ind=args.steps_batch_ind,
                                                                      gibbs_steps=args.test_gibbs_steps,
                                                                      gibbs_k_steps=args.test_gibbs_k_steps,
                                                                      gibbs_n_steps=args.test_gibbs_n_steps,
                                                                      transform_every=args.transform_every)[0]
                                if args.return_steps > 0:
                                    x_cond_custom_fresh_same, cond_custom_fresh_same_steps = x_cond_custom_fresh_same
                                    plot_samples(x_cond_custom_fresh_same,
                                                 f"{args.save_dir}/"
                                                 f"same_samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                                    if args.steps_batch_ind is None:
                                        for chain_itr in range(cond_custom_fresh_same_steps.shape[0]):
                                            plot_samples(cond_custom_fresh_same_steps[chain_itr],
                                                         f"{args.save_dir}/"
                                                         f"same_chain_custom_chain_{cls_descriptor_str}_"
                                                         f"cond_fresh_{itr}_itr_{chain_itr}.png")
                                    else:
                                        plot_samples(cond_custom_fresh_same_steps,
                                                     f"{args.save_dir}/"
                                                     f"same_chain_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")
                                else:
                                    plot_samples(x_cond_custom_fresh_same,
                                                 f"{args.save_dir}/"
                                                 f"same_samples_custom_{cls_descriptor_str}_cond_fresh_{itr}.png")

                    plt.clf()
                    ema_logp_net.cpu()

                    if args.data not in IMG_DSETS:

                        if label_shape != (-1,):
                            if all_binary:
                                prod_cond_cls = product(*(range(2) for _ in range(label_dim)))
                            else:
                                prod_cond_cls = product(*(range(i) for i in label_shape))

                            def next_cond_cls():
                                label_combo = next(prod_cond_cls)
                                return one_hot(torch.tensor(label_combo)[None])

                            if all_binary:
                                all_classes = [next_cond_cls() for _ in range(2 ** label_dim)]
                            else:
                                all_classes = [next_cond_cls() for _ in range(all_labels + label_dim == 1)]
                            all_classes = torch.cat(all_classes, dim=0)

                            def joint_energy(x):
                                x_sh = x.shape
                                bs = x_sh[0]
                                x = x.repeat_interleave(all_labels, 0)
                                if unif_label_shape is None:
                                    repeat_classes = all_classes.repeat((bs, 1))
                                else:
                                    repeat_classes = all_classes.repeat((bs, 1, 1))
                                e = energy(ema_logp_net, x, repeat_classes)
                                e = e.view(bs, all_labels)
                                return e.logsumexp(dim=1)

                            if args.model != "joint":
                                joint_energy = partial(energy, ema_logp_net)

                            plt_flow_density(joint_energy, plt.gca())
                            plt.savefig(f"{args.save_dir}/density_joint_{itr}.png")

                            plt_flow_density(joint_energy, plt.gca(), exp=False)
                            plt.savefig(f"{args.save_dir}/density_log_joint_{itr}.png")
                        else:
                            all_classes = None

                        if all_binary:
                            _iterator = range(2 ** label_dim)
                        else:
                            _iterator = range(all_labels + (label_dim == 1))

                        for cond_cls in _iterator:

                            if label_shape != (-1,):
                                cond_cls_struct = all_classes[cond_cls]
                            else:
                                cond_cls_struct = None  # linter

                            if args.model == "joint":
                                if label_shape != (-1,):
                                    norm = plt_flow_density(
                                        lambda x: energy(ema_logp_net, x,
                                                         cond_cls_struct.repeat_interleave(x.shape[0], 0)),
                                        plt.gca(), return_norm=True)
                                    plt.savefig(f"{args.save_dir}/density{cond_cls}_cond_{itr}.png")

                                    plt_flow_density(
                                        lambda x: energy(ema_logp_net, x,
                                                         cond_cls_struct.repeat_interleave(x.shape[0], 0)),
                                        plt.gca(), exp=False)
                                    plt.savefig(f"{args.save_dir}/density_log_{cond_cls}_cond_{itr}.png")
                                else:
                                    norm = plt_flow_density(
                                        lambda x: energy(ema_logp_net, x, format_lbl(bs=x.shape[0], cls=cond_cls)),
                                        plt.gca(), return_norm=True)
                                    plt.savefig(f"{args.save_dir}/density{cond_cls}_cond_{itr}.png")

                                    plt_flow_density(
                                        lambda x: energy(ema_logp_net, x, format_lbl(bs=x.shape[0], cls=cond_cls)),
                                        plt.gca(), exp=False)
                                    plt.savefig(f"{args.save_dir}/density_log_{cond_cls}_cond_{itr}.png")
                            else:
                                if label_shape == (-1,):
                                    cond_cls_struct = format_lbl(bs=1, cls=cond_cls).squeeze()

                                # put on device since diff_label_axes is there
                                cond_cls_struct = cond_cls_struct.to(args.device)
                                # put back on cpu for plotting
                                cond_cls_struct = label_onehot_poj(cond_cls_struct).squeeze().type(torch.int64).cpu()

                                def cond_energy(x):
                                    cond_logits = ema_logp_net(x)
                                    cond_cls_struct_ = cond_cls_struct[None].repeat_interleave(x.shape[0], 0)
                                    return logit_log_prob_ind(shape_logits(cond_logits),
                                                              cond_cls_struct_).sum(-1)

                                norm = plt_flow_density(cond_energy, plt.gca(), return_norm=True)
                                plt.savefig(f"{args.save_dir}/density{cond_cls}_cond_{itr}.png")

                                plt_flow_density(cond_energy, plt.gca(), exp=False)
                                plt.savefig(f"{args.save_dir}/density_log_{cond_cls}_cond_{itr}.png")

                            logger(f"\tnorm {cond_cls} = {norm.item():.4f}")

            ema_logp_net.to(args.device)
            ema_logp_net.train()

            if args.log_ema:
                logger(f"Params the same after log: {ema_params(logp_net, ema_logp_net)}")

            if args.eval is not None:
                break

        if itr % args.ckpt_every == 0 and itr != 0:
            save_ckpt(itr, logger, args.device,
                      ckpt_path=args.ckpt_path,
                      data={
                          "models": {
                              "logp_net": logp_net,
                              "ema_logp_net": ema_logp_net,
                          },
                          "optimizer": {
                              "logp_net_optimizer": optim,
                          },
                          "replay_buffer": replay_buffer,
                          "y_replay_buffer": y_replay_buffer,
                          "itr": itr,
                          "tst_accs": tst_accs
                      })

        if itr % args.ckpt_recent_every == 0 and itr != 0:
            save_ckpt(itr, logger, args.device,
                      save_dir=args.save_dir,
                      most_recent=args.ckpt_recent,
                      prefix="recent",
                      data={
                          "models": {
                              "logp_net": logp_net,
                              "ema_logp_net": ema_logp_net,
                          },
                          "optimizer": {
                              "logp_net_optimizer": optim,
                          },
                          "replay_buffer": replay_buffer,
                          "y_replay_buffer": y_replay_buffer,
                          "itr": itr,
                          "tst_accs": tst_accs
                      })

        if itr % args.save_best_every == 0 and itr != 0:
            save_ckpt(itr, logger, args.device,
                      save_dir=args.save_dir,
                      save_best=args.save_best,
                      tst_accs=tst_accs,
                      prefix="best",
                      data={
                          "models": {
                              "logp_net": logp_net,
                              "ema_logp_net": ema_logp_net,
                          },
                          "optimizer": {
                              "logp_net_optimizer": optim,
                          },
                          "replay_buffer": replay_buffer,
                          "y_replay_buffer": y_replay_buffer,
                          "itr": itr,
                          "tst_accs": tst_accs
                      })

        if (itr % args.save_every == 0 and itr != 0 and itr >= args.save_after) or itr == args.save_at:
            save_ckpt(itr, logger, args.device,
                      save_dir=args.save_dir,
                      overwrite=False,
                      prefix="ckpt",
                      data={
                          "models": {
                              "logp_net": logp_net,
                              "ema_logp_net": ema_logp_net,
                          },
                          "optimizer": {
                              "logp_net_optimizer": optim,
                          },
                          "replay_buffer": replay_buffer,
                          "y_replay_buffer": y_replay_buffer,
                          "itr": itr,
                          "tst_accs": tst_accs
                      })


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Zero Shot Learning yeet")
    parser.add_argument("--data", type=str, default="moons", choices=["moons", "circles", "rings", "checkerboard",
                                                                      "8gaussians",
                                                                      "8gaussians_struct", "8gaussians_struct_missing",
                                                                      "8gaussians_multi", "8gaussians_hierarch",
                                                                      "8gaussians_hierarch_missing",
                                                                      "8gaussians_hierarch_binarized_missing",
                                                                      "8gaussians_hierarch_binarized", "rings_struct"]
                                                                     + IMG_DSETS)
    parser.add_argument("--mode", type=str, default="cond", choices=["cond", "uncond", "sup"])
    parser.add_argument("--model", type=str, default="joint", choices=["joint", "poj"])
    parser.add_argument("--uncond_poj", action="store_true", default=False)
    parser.add_argument("--zero_shot", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="tmp")
    parser.add_argument("--log_filename", type=str, default="log.txt")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lr_at", nargs="+", type=float, default=[])
    parser.add_argument("--lr_itr_at", nargs="+", type=int, default=[])
    parser.add_argument("--beta1", type=float, default=.9)
    parser.add_argument("--beta2", type=float, default=.999)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--k_at", nargs="+", type=int, default=[])
    parser.add_argument("--sgld_steps_itr_at", nargs="+", type=int, default=[])
    parser.add_argument("--sigma", type=float, default=.01)
    parser.add_argument("--step_size", type=float, default=1.)
    parser.add_argument("--temp", type=float, default=2., help="Discrete sampling temperature")
    parser.add_argument("--plot_temp_sigma_start", type=float, default=-1., help="Discrete sampling temperature")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=100)
    parser.add_argument("--ckpt_recent_every", type=int, default=1000)
    parser.add_argument("--ckpt_recent", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=10000000000)
    parser.add_argument("--init_scale", type=float, default=1)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--spectral", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default="tmp/ck.pt")
    parser.add_argument("--ema", type=float, default=0, help="Weight for Exponential Moving Average")
    parser.add_argument("--log_ema", action="store_true", default=False)
    parser.add_argument("--n_visible", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--gibbs_steps", type=int, default=5)
    parser.add_argument("--gibbs_steps_at", nargs="+", type=int, default=[])
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sampling", type=str, default="sr", choices=["sr", "pcd"])
    parser.add_argument("--init_dist", type=str, default="unif", choices=["unif", "norm"])
    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--warmup_itrs_from_restart", type=int, default=0)
    parser.add_argument("--p_control", type=float, default=0)
    parser.add_argument("--n_control", type=float, default=0)
    parser.add_argument("--cnn", action="store_true", default=False)
    parser.add_argument("--small_cnn", action="store_true", default=False)
    parser.add_argument("--img_size", type=int, default=32, choices=[32, 64, 128])
    parser.add_argument("--p_y_x", type=float, default=0)
    parser.add_argument("--first_gibbs", type=str, default="dis", choices=["dis", "cts"])
    parser.add_argument("--no_shift_logit", action="store_true", default=False)
    parser.add_argument("--other_reverse_changes", action="store_true", default=False)
    parser.add_argument("--interleave", action="store_true", default=False)
    parser.add_argument("--mnist_act", type=str, default="elu", choices=["elu", "lrelu"])
    parser.add_argument("--gibbs_k_steps", type=int, default=1)
    parser.add_argument("--gibbs_n_steps", type=int, default=1)
    parser.add_argument("--test_k", type=int, default=-1)
    parser.add_argument("--test_gibbs_k_steps", type=int, default=-1)
    parser.add_argument("--test_gibbs_n_steps", type=int, default=-1)
    parser.add_argument("--test_gibbs_steps", type=int, default=-1)
    parser.add_argument("--logit", action="store_true", default=False)
    parser.add_argument("--plot_cond_buffer", action="store_true", default=False)
    parser.add_argument("--plot_cond_marginalize", action="store_true", default=False)
    parser.add_argument("--plot_cond_init_buffer", action="store_true", default=False)
    parser.add_argument("--plot_cond_continue_buffer", action="store_true", default=False)
    parser.add_argument("--plot_cond_continue_buffer_uncond", action="store_true", default=False)
    parser.add_argument("--plot_cond_filter", type=int, default=1)
    parser.add_argument("--plot_ais", action="store_true", default=False)
    parser.add_argument("--plot_temp", action="store_true", default=False)
    parser.add_argument("--plot_same", action="store_true", default=False)
    parser.add_argument("--plot_cond_fix_b", action="store_true", default=False)
    parser.add_argument("--return_steps", type=int, default=0, help="Leaps to log chain")
    parser.add_argument("--steps_batch_ind", type=int, default=-1, help="Batch index when logging chain")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--save_at", type=int, default=-1)
    parser.add_argument("--save_best", type=int, default=1)
    parser.add_argument("--save_best_every", type=int, default=1000)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--plot_after", type=int, default=0)
    parser.add_argument("--img_sigma", type=float, default=-1)
    parser.add_argument("--plot_uncond_fresh", action="store_true", default=False)
    parser.add_argument("--plot_energy_b", action="store_true", default=False)
    parser.add_argument("--unif_init_b", action="store_true", default=False)
    parser.add_argument("--test_n_steps", type=int, default=-1)
    parser.add_argument("--cond_arch", action="store_true", default=False)
    parser.add_argument("--small_mlp", action="store_true", default=False)
    parser.add_argument("--small_mlp_nhidden", type=int, default=256)
    parser.add_argument("--dsprites_test", action="store_true", default=False)
    parser.add_argument("--full_test", action="store_true", default=False)
    parser.add_argument("--cond_mode", type=str, default="dot", choices=["dot", "cos", "cnn-mlp"])
    parser.add_argument("--n_f", type=int, default=8)
    parser.add_argument("--transform_every", type=int, default=40)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--only_transform_buffer", action="store_true", default=False)
    parser.add_argument("--multiscale", action="store_true", default=False)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--spec_norm", action="store_true", default=False)
    parser.add_argument("--self_attn", action="store_true", default=False)
    parser.add_argument("--alias", action="store_true", default=False)
    parser.add_argument("--kl", type=float, default=0)
    parser.add_argument("--bp", type=float, default=0)
    parser.add_argument("--kl_start", type=int, default=20)
    parser.add_argument("--truncated_bp", action="store_true", default=False)
    parser.add_argument("--full_bp", action="store_true", default=False)
    parser.add_argument("--yd_buffer", type=str, default=None, choices=["replay", "reservoir"])
    parser.add_argument("--eval", type=str, default=None, help="Path to weights for evaluation.")
    parser.add_argument("--resume", type=str, default=None, help="Path to weights for manually restart of training.")
    parser.add_argument("--weighted_ce", action="store_true", default=False)
    parser.add_argument("--utzappos_drop_infreq", type=float, default=0.1)
    parser.add_argument("--celeba_drop_infreq", type=float, default=10.)
    parser.add_argument("--cub_drop_infreq", type=float, default=10)
    parser.add_argument("--dset_split_type", type=str, default="balanced", choices=["random", "balanced", "zero_shot"])
    parser.add_argument("--eval_filter_to_tst_combos", action="store_true", default=False)
    parser.add_argument("--eval_cond_acc", action="store_true", default=False)
    parser.add_argument("--dset_balanced_split_ratio", type=float, default=0.5)
    parser.add_argument("--poj_joint", action="store_true", default=False)
    parser.add_argument("--clamp_data", action="store_true", default=False)
    parser.add_argument("--clamp_samples", action="store_true", default=False)
    parser.add_argument("--clip_grad_norm", type=float, default=-1)
    parser.add_argument("--multi_gpu", action="store_true", default=False, help="Multiple GPUs buddyyyyyy")
    parser.add_argument("--old_multi_gpu", action="store_true", default=False, help="Multiple GPUs buddyyyyyy")
    parser.add_argument("--utzappos_blur_transform", action="store_true", default=False, help="Just use blur transform")
    parser.add_argument("--celeba_all_transform", action="store_true", default=False, help="Use all transforms")
    parser.add_argument("--celeba_no_color_transform", action="store_true", default=False, help="No color transforms")
    parser.add_argument("--just_sampling", action="store_true", default=False, help="Just sampling for eval.")
    parser.add_argument("--save_test_predictions", action="store_true", default=False, help="Save prediction scores.")

    parse_args = parser.parse_args()

    assert implies(parse_args.just_sampling, parse_args.eval is not None)
    assert implies(parse_args.just_sampling, parse_args.mode == "cond")

    assert implies(parse_args.model != "joint" and not parse_args.uncond_poj, parse_args.mode in ("cond", "sup"))
    assert implies(parse_args.poj_joint, parse_args.model == "poj")
    assert implies(parse_args.mode == "sup", parse_args.model == "poj")
    assert implies(parse_args.p_y_x > 0, parse_args.mode in ("cond", "sup"))
    assert implies(parse_args.model != "joint" and not parse_args.poj_joint, parse_args.p_y_x > 0)
    assert implies(parse_args.uncond_poj, parse_args.mode == "uncond" and parse_args.model == "poj")
    assert implies(parse_args.uncond_poj, parse_args.data in ("utzappos", "celeba")), "NOT IMPLEMENTED!!"

    assert implies(parse_args.model == "joint" or parse_args.poj_joint, parse_args.p_y_x == 0), "DON'T DO THAT BUDDYYYY"

    assert implies(parse_args.mode == "sup", parse_args.ema == 0)
    assert implies(parse_args.mode != "sup", parse_args.ema >= .99)

    assert implies(parse_args.cnn, parse_args.data in IMG_DSETS)

    assert implies(parse_args.img_size == 128, parse_args.data == "cub")

    assert implies(parse_args.zero_shot, parse_args.data == "utzappos")

    assert implies("8gaussians_" in parse_args.data, parse_args.mode in ("cond", "sup"))

    assert not (parse_args.truncated_bp and parse_args.full_bp)

    assert implies(parse_args.logit, parse_args.data == "mnist" and not parse_args.cnn)

    assert parse_args.resume is None or parse_args.eval is None

    assert implies(parse_args.warmup_itrs_from_restart > 0, parse_args.resume is not None)

    assert len(parse_args.lr_at) == len(parse_args.lr_itr_at)
    assert np.all(np.diff([parse_args.lr] + parse_args.lr_at) < 0), "LRs must decay"
    if len(parse_args.lr_itr_at) > 0:
        assert parse_args.lr_itr_at[0] > 0
        assert parse_args.lr_itr_at[0] > parse_args.warmup_itrs
        assert np.all(np.diff(parse_args.lr_itr_at) > 0), "LR itrs must increase"

    assert implies(len(parse_args.gibbs_steps_at) > 0, parse_args.poj_joint and not parse_args.uncond_poj), \
        "Not implemented everywhere boi!"
    assert len(parse_args.k_at) == len(parse_args.gibbs_steps_at) == len(parse_args.sgld_steps_itr_at)
    assert np.all(np.diff([parse_args.k] + parse_args.k_at) > 0), "Number of steps must increase"
    assert np.all(np.diff([parse_args.gibbs_steps] + parse_args.gibbs_steps_at) > 0), "Number of steps must increase"
    if len(parse_args.sgld_steps_itr_at) > 0:
        assert parse_args.sgld_steps_itr_at[0] > 0
        assert np.all(np.diff(parse_args.sgld_steps_itr_at) > 0), "Number of steps itrs must increase"

    assert parse_args.clamp_data == parse_args.clamp_samples, "These should always be the same (at least for now)"
    assert implies(parse_args.clamp_data or parse_args.clamp_samples, parse_args.data in ["utzappos", "celeba", "cub"])

    assert (parse_args.kl > 0) == (parse_args.bp > 0), "Both kl and bp should be weighted, or neither"
    assert implies(parse_args.kl > 0, parse_args.data in IMG_DSETS), "KL loss only defined on images"

    assert implies(parse_args.utzappos_blur_transform, parse_args.data == "utzappos")
    assert implies(parse_args.utzappos_blur_transform, parse_args.transform)

    assert implies(parse_args.celeba_all_transform, parse_args.data == "celeba")
    assert implies(parse_args.celeba_no_color_transform, parse_args.data == "celeba")
    assert implies(parse_args.celeba_all_transform, parse_args.transform)
    assert implies(parse_args.celeba_no_color_transform, parse_args.transform)
    assert not(parse_args.celeba_no_color_transform and parse_args.celeba_all_transform)

    assert implies(parse_args.only_transform_buffer, parse_args.transform)

    assert implies(parse_args.dset_split_type == "zero_shot", parse_args.data == "celeba")

    assert implies(parse_args.save_best > 0 and parse_args.eval is None,
                   parse_args.save_best_every == parse_args.print_every)

    if parse_args.utzappos_drop_infreq < 0:
        parse_args.utzappos_drop_infreq = None

    if parse_args.cub_drop_infreq < 0:
        parse_args.cub_drop_infreq = None

    if parse_args.step_size < 0:
        parse_args.step_size = (parse_args.sigma ** 2) / 2.

    if parse_args.temp < 0:
        parse_args.temp = ((parse_args.sigma ** 2) / 2.) / parse_args.step_size

    if parse_args.plot_temp_sigma_start < 0:
        parse_args.plot_temp_sigma_start = (2 * parse_args.step_size) ** .5

    if parse_args.test_k < 0:
        parse_args.test_k = parse_args.k

    if parse_args.test_gibbs_k_steps < 0:
        parse_args.test_gibbs_k_steps = parse_args.gibbs_k_steps

    if parse_args.test_gibbs_n_steps < 0:
        parse_args.test_gibbs_n_steps = parse_args.gibbs_n_steps

    if parse_args.test_gibbs_steps < 0:
        parse_args.test_gibbs_steps = parse_args.gibbs_steps

    if parse_args.img_sigma < 0:
        parse_args.img_sigma = parse_args.sigma

    if parse_args.test_n_steps < 0:
        parse_args.test_n_steps = parse_args.n_steps

    if parse_args.steps_batch_ind < 0:
        if parse_args.data in IMG_DSETS:
            parse_args.steps_batch_ind = 0  # log chain for the first index in the batch
        else:
            parse_args.steps_batch_ind = None  # log chain for all samples in the batch

    makedirs(parse_args.save_dir)
    with open(f'{parse_args.save_dir}/params.txt', 'w') as args_file:
        json.dump(parse_args.__dict__, args_file, sort_keys=True, indent=4)

    # add device to the namespace after saving args to txt since it's not serializable
    parse_args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger = get_logger(parse_args)

    if parse_args.eval is not None:
        logger(f"===================== CODE SEED {parse_args.seed} =====================")
        torch.manual_seed(parse_args.seed)
        np.random.seed(parse_args.seed)

    main(parse_args)
