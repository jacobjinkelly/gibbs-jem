"""
Load datasets.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from .cub import get_cub_dataset
from .mwa import MNISTWithAttributes
from .toy import ALL_TOY_DSETS, CTS_TOY_DSETS, LBL_TOY_DSETS, LBL_CTS_TOY_DSETS, process_data
from .utils import makedirs, wrap_args


@wrap_args
def get_data(dataset,
             num_data,
             img_size,
             batch_size, eval_batch_size,
             save_dir, device,
             obs_minimum, img_sigma,
             root="data/", **_):
    """
    Return dataloaders and attributes for train, validation, and test *class* splits, and plotting function.
    """

    if dataset in ALL_TOY_DSETS:
        cts = dataset in CTS_TOY_DSETS + LBL_CTS_TOY_DSETS
        lbl = dataset in LBL_TOY_DSETS + LBL_CTS_TOY_DSETS
        trn, encoder = process_data(dataset, num_data, device, cts=cts, labels=lbl, return_encoder=True)
        val = process_data(dataset, num_data, device, cts=cts)
        tst = process_data(dataset, num_data, device, cts=cts)

        trn_att = val_att = tst_att = None

        if cts:
            data_size = (2, )
        else:
            data_size = (2 * encoder.nbits, )
        nums_attributes = [1]

        def density(logp_net, exp=True):
            """
            Compute self-normalized density on a grid.
            """
            npts = 100

            # prepare points to evaluate energy on
            side = np.linspace(-4, 4, npts)
            xx, yy = np.meshgrid(side, side)
            x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
            x = torch.from_numpy(x).type(torch.float32).to(device)

            # get energy and normalize
            logpx = logp_net(x)
            logpx = logpx - logpx.logsumexp(0)
            logpx = logpx.cpu().detach().numpy().reshape(npts, npts)
            if exp:
                logpx = np.exp(logpx)
            logpx = logpx / logpx.sum()

            return logpx

        def plot(logp_net, x, *path, manual=False):
            if not manual:
                *prev_path, itr = path
                path = prev_path + [f"{itr:07}.png"]
            path = os.path.join(save_dir, *path)
            makedirs(os.path.dirname(path))

            if cts:
                x = x.cpu().numpy()
                energy = logp_net
            else:
                # round values to 0 or 1
                x = torch.argmin(torch.cat(((x - 0).abs()[None], (x - 1).abs()[None])), dim=0)

                # decode 0 and 1 to discretized bins in 2 dimensional space
                x = encoder.decode_batch(x).cpu().numpy()

                # rename function, otherwise we get some weird infinite recursion
                energy = lambda x_: logp_net(encoder.encode_batch(x_))

            plt.clf()

            ax = plt.subplot(2, 2, 1, aspect="equal", title='samples')
            ax.scatter(x[:, 0], x[:, 1], s=1)
            ax.set_xlim((-4, 4))
            ax.set_ylim((-4, 4))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax = plt.subplot(2, 2, 2, aspect="equal", title="sample hist")
            ax.hist2d(x[:, 0], x[:, 1], range=[[-4, 4], [-4, 4]], bins=100)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax = plt.subplot(2, 2, 3, aspect="equal", title="log density")
            ax.imshow(density(energy, exp=False))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            ax = plt.subplot(2, 2, 4, aspect="equal", title="density")
            ax.imshow(density(energy))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            plt.savefig(path)

        def process_batch(batch):
            x_train, = batch
            x_train = x_train.to(device)

            x_train = x_train + img_sigma * torch.randn_like(x_train)

            return x_train, None, None, None

        # get init dist
        x_plus_noise, *_ = process_batch(trn.tensors)
        mu = x_plus_noise.mean(0)
        std = x_plus_noise.std(0)
        init_dist = torch.distributions.Normal(mu, std)

    else:
        if dataset == "CUB":
            data_size = (3, img_size, img_size)
            nums_attributes = [2] * 312

            init_dist = torch.distributions.Uniform(-torch.ones(data_size), torch.ones(data_size))

            normalize = lambda x: 2 * x - 1
            plot_transform = lambda x: (x + 1) / 2  # to [0, 1] for plotting

            trn_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            tst_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize
            ])

            trn, trn_att = get_cub_dataset(root, "trn",
                                           transform=trn_transforms)
            val, val_att = get_cub_dataset(root, "val",
                                           transform=tst_transforms)
            tst, tst_att = get_cub_dataset(root, "tst",
                                           transform=tst_transforms)

        elif dataset == "MWA":
            data_size = (3, img_size, img_size)
            nums_attributes = [2] * 41
            normalize = lambda x: 2 * x - 1
            plot_transform = lambda x: (x + 1) / 2  # to [0, 1] for plotting

            init_dist = torch.distributions.Uniform(-torch.ones(data_size), torch.ones(data_size))

            trn = MNISTWithAttributes(root, True, img_size, transform=normalize)
            val = tst = MNISTWithAttributes(root, True, img_size, transform=normalize)
            trn_att = val_att = tst_att = None

        else:
            raise ValueError

        def plot(image, *path, manual=False):
            """
            Plot image image at path
            """
            if not manual:
                *prev_path, itr = path
                path = prev_path + [f"{itr:07}.png"]
            path = os.path.join(save_dir, *path)
            makedirs(os.path.dirname(path))
            save_image(plot_transform(image.view(image.size(0), *data_size)),
                       path,
                       normalize=False,
                       nrow=int(image.size(0) ** .5))

        if dataset == "CUB":
            trn_att = trn_att.to(device)
            val_att = val_att.to(device)
            tst_att = tst_att.to(device)

        def process_batch(batch):
            x_train, y_train, a_train, c_train = batch
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            a_train = a_train.to(device)
            c_train = c_train.to(device)

            c_train = (c_train > obs_minimum).float()
            x_train = x_train + img_sigma * torch.randn_like(x_train)

            return x_train, y_train, a_train, c_train

    data_config = {
        "data_size": data_size,
        "nums_attributes": nums_attributes,
        "init_dist": init_dist
    }

    trn_ld, trn_eval_ld, val_ld, tst_ld = get_loaders(trn, val, tst, batch_size, eval_batch_size)

    return (trn_ld, trn_att), (trn_eval_ld, ), (val_ld, val_att), (tst_ld, tst_att), \
        process_batch, plot, data_config


def get_loaders(trn, val, tst, batch_size, eval_batch_size):
    trn_ld = DataLoader(trn, batch_size=batch_size, shuffle=True, drop_last=True)

    trn_eval_ld = DataLoader(trn, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    val_ld = DataLoader(val, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    tst_ld = DataLoader(tst, batch_size=eval_batch_size, shuffle=False, drop_last=False)

    return trn_ld, trn_eval_ld, val_ld, tst_ld
