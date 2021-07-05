"""
dSprites dataset
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None):
        assert len(tensors) == 2
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def process_dsprites(root):
    dsprites = np.load(
        os.path.join(root, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'),
        allow_pickle=True, encoding='latin1')

    orientations = np.array([0, 9, 10, 19, 20, 29, 30, 39])
    shapes = np.array([1])

    imgs = dsprites["imgs"]
    lcs = dsprites["latents_classes"]

    orientations_mask = np.isin(lcs[:, 3], orientations)
    shapes_mask = np.isin(lcs[:, 1], shapes)
    mask = np.logical_and(orientations_mask, shapes_mask)

    lcs = lcs[:, 4:6]
    lcs = lcs[mask]

    # group together adjacent x, y coordinate classes
    lcs = lcs // 2

    return imgs[mask], lcs


def get_dsprites_dset(root, transform=None, test=False, test_size=2500):
    x, y = process_dsprites(root)

    x = torch.from_numpy(x).float()[:, None]
    x = torch.nn.functional.interpolate(x, size=32, mode='bilinear')

    y = torch.from_numpy(y)

    if test:
        indices = torch.randperm(len(y))
        trn_ind, tst_ind = indices[test_size:], indices[:test_size]
        x_test = x[tst_ind]
        y_test = y[tst_ind]

        x_train = x[trn_ind]
        y_train = y[trn_ind]
        return CustomTensorDataset(x_train, y_train, transform=transform), \
               CustomTensorDataset(x_test, y_test, transform=transform)
    else:
        return CustomTensorDataset(x, y, transform=transform)


if __name__ == "__main__":
    dset, tst = get_dsprites_dset("../data/", test=True)
    print(len(dset), len(tst))
