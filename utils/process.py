"""
Modified image folder dataset loader
"""

import argparse
import math
import os
import pickle
from collections import OrderedDict, defaultdict
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


CELEBA_ZERO_SHOT_ATTRIBUTES = ["Male",
                               "No_Beard",
                               "Bangs",
                               "Young",
                               "Smiling",
                               "Mouth_Slightly_Open",
                               "Black_Hair",
                               "Brown_Hair",
                               "Blond_Hair"]

CELEBA_ZERO_SHOT_COMBOS = [{"Male": 1, "No_Beard": 0},
                           {"Male": 1, "No_Beard": 1},
                           {"Male": 1, "Bangs": 0},
                           {"Male": 1, "Bangs": 1},
                           {"Male": 1, "Young": 0},
                           {"Male": 1, "Young": 1},
                           {"Male": 1, "Smiling": 0},
                           {"Male": 1, "Smiling": 1},
                           {"Male": 1, "Mouth_Slightly_Open": 0},
                           {"Male": 1, "Mouth_Slightly_Open": 1},
                           {"Male": 1, "Black_Hair": 1},
                           {"Male": 1, "Brown_Hair": 1},
                           {"Male": 1, "Blond_Hair": 1},
                           {"Male": 0, "Bangs": 0},
                           {"Male": 0, "Bangs": 1},
                           {"Male": 0, "Young": 0},
                           {"Male": 0, "Young": 1},
                           {"Male": 0, "Smiling": 0},
                           {"Male": 0, "Smiling": 1},
                           {"Male": 0, "Mouth_Slightly_Open": 0},
                           {"Male": 0, "Mouth_Slightly_Open": 1},
                           {"Male": 0, "Black_Hair": 1},
                           {"Male": 0, "Brown_Hair": 1},
                           {"Male": 0, "Blond_Hair": 1},
                           {"No_Beard": 1, "Bangs": 1},
                           {"No_Beard": 1, "Bangs": 0},
                           {"No_Beard": 1, "Young": 1},
                           {"No_Beard": 1, "Young": 0},
                           {"No_Beard": 1, "Smiling": 1},
                           {"No_Beard": 1, "Smiling": 0},
                           {"No_Beard": 1, "Mouth_Slightly_Open": 1},
                           {"No_Beard": 1, "Mouth_Slightly_Open": 0},
                           {"No_Beard": 1, "Black_Hair": 1},
                           {"No_Beard": 1, "Brown_Hair": 1},
                           {"No_Beard": 1, "Blond_Hair": 1},
                           {"No_Beard": 0, "Bangs": 1},
                           {"No_Beard": 0, "Bangs": 0},
                           {"No_Beard": 0, "Young": 1},
                           {"No_Beard": 0, "Young": 0},
                           {"No_Beard": 0, "Smiling": 1},
                           {"No_Beard": 0, "Smiling": 0},
                           {"No_Beard": 0, "Mouth_Slightly_Open": 1},
                           {"No_Beard": 0, "Mouth_Slightly_Open": 0},
                           {"No_Beard": 0, "Black_Hair": 1},
                           {"No_Beard": 0, "Brown_Hair": 1},
                           {"No_Beard": 0, "Blond_Hair": 1},
                           {"Bangs": 1, "Young": 1},
                           {"Bangs": 1, "Young": 0},
                           {"Bangs": 1, "Smiling": 1},
                           {"Bangs": 1, "Smiling": 0},
                           {"Bangs": 1, "Mouth_Slightly_Open": 1},
                           {"Bangs": 1, "Mouth_Slightly_Open": 0},
                           {"Bangs": 1, "Black_Hair": 1},
                           {"Bangs": 1, "Brown_Hair": 1},
                           {"Bangs": 1, "Blond_Hair": 1},
                           {"Bangs": 0, "Young": 1},
                           {"Bangs": 0, "Young": 0},
                           {"Bangs": 0, "Smiling": 1},
                           {"Bangs": 0, "Smiling": 0},
                           {"Bangs": 0, "Mouth_Slightly_Open": 1},
                           {"Bangs": 0, "Mouth_Slightly_Open": 0},
                           {"Bangs": 0, "Black_Hair": 1},
                           {"Bangs": 0, "Brown_Hair": 1},
                           {"Bangs": 0, "Blond_Hair": 1},
                           {"Young": 1, "Smiling": 1},
                           {"Young": 1, "Smiling": 0},
                           {"Young": 1, "Mouth_Slightly_Open": 1},
                           {"Young": 1, "Mouth_Slightly_Open": 0},
                           {"Young": 1, "Black_Hair": 1},
                           {"Young": 1, "Brown_Hair": 1},
                           {"Young": 1, "Blond_Hair": 1},
                           {"Young": 0, "Smiling": 1},
                           {"Young": 0, "Smiling": 0},
                           {"Young": 0, "Mouth_Slightly_Open": 1},
                           {"Young": 0, "Mouth_Slightly_Open": 0},
                           {"Young": 0, "Black_Hair": 1},
                           {"Young": 0, "Brown_Hair": 1},
                           {"Young": 0, "Blond_Hair": 1},
                           {"Smiling": 1, "Mouth_Slightly_Open": 1},
                           {"Smiling": 1, "Mouth_Slightly_Open": 0},
                           {"Smiling": 1, "Black_Hair": 1},
                           {"Smiling": 1, "Brown_Hair": 1},
                           {"Smiling": 1, "Blond_Hair": 1},
                           {"Smiling": 0, "Mouth_Slightly_Open": 1},
                           {"Smiling": 0, "Mouth_Slightly_Open": 0},
                           {"Smiling": 0, "Black_Hair": 1},
                           {"Smiling": 0, "Brown_Hair": 1},
                           {"Smiling": 0, "Blond_Hair": 1},
                           {"Mouth_Slightly_Open": 1, "Black_Hair": 1},
                           {"Mouth_Slightly_Open": 1, "Brown_Hair": 1},
                           {"Mouth_Slightly_Open": 1, "Blond_Hair": 1},
                           {"Mouth_Slightly_Open": 0, "Black_Hair": 1},
                           {"Mouth_Slightly_Open": 0, "Brown_Hair": 1},
                           {"Mouth_Slightly_Open": 0, "Blond_Hair": 1},
                           ]


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, zs=False, transform=None):
        if zs:
            assert len(tensors) == 3
        else:
            assert len(tensors) == 2
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.zs = zs

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        if self.zs:
            zs = self.tensors[2][index]
            return x, y, zs
        else:
            return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def order_set(seq):
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_loader(dataset, batch_size, split="trn"):
    dataloader_ = partial(DataLoader, dataset=dataset, batch_size=batch_size)
    if split == "trn":
        return dataloader_(shuffle=True, drop_last=True)
    elif split == "tst":
        return dataloader_(shuffle=False, drop_last=False)
    else:
        raise ValueError(f"Unrecognized split {split}.")


def get_data_gen(device, zs_mode=False, *args, **kwargs):
    loader = get_loader(*args, **kwargs)

    def inf_gen():
        while True:
            if zs_mode:
                for images, targets, zs in loader:
                    yield images.to(device), targets.to(device), zs
            else:
                for images, targets in loader:
                    yield images.to(device), targets.to(device)

    gen = inf_gen()

    def data_batch(get_zs=False):
        if zs_mode:
            images, targets, zs = next(gen)
            if get_zs:
                return images, targets, zs
            else:
                return images, targets
        else:
            return next(gen)

    return data_batch


def log_dset_label_info(logger, dset_label_info):
    for col_name, (col_num, col_freq) in dset_label_info.items():
        logger(f"{col_num} {col_name} {col_freq:.4f}")


def get_label_name_from_ind(label_dim, dset_label_info):
    for col_name, (col_num, _) in dset_label_info.items():
        if label_dim == col_num:
            return col_name
    raise ValueError(f"Could not find {label_dim} in label info of len {len(dset_label_info)}")


def find_duplicates_in_dsets(dset1, dset2, tuple_format=False, itself=False):
    if tuple_format:
        dset1_iter = zip(*dset1)
    else:
        dset1_iter = dset1
        dset2 = dset2[:]
    for tst_img, tst_label in dset1_iter:
        same_label_mask = (tst_label == dset2[1]).all(1)
        assert len(same_label_mask.shape) == 1
        if same_label_mask.any():
            same_img_and_label_mask = (tst_img == dset2[0][same_label_mask]).all(-1).all(-1).all(-1)
            assert len(same_img_and_label_mask.shape) == 1
            if (itself and same_img_and_label_mask.sum() > 1) or (not itself and same_img_and_label_mask.any()):
                print(f"Number of exact label matches: {same_label_mask.sum().item()}")
                print(f"Number of exact image and label matches: {same_img_and_label_mask.sum().item()}")


class CelebAIDImageFolder(datasets.ImageFolder):
    def __init__(self, attr_fn, pkl_fn, img_size, drop_infreq=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        pkl_fn = CelebAIDImageFolder.get_cache_name(pkl_fn, img_size, drop_infreq)

        try:
            with open(pkl_fn, 'rb') as f:
                self.img_ids, self._label_info = pickle.load(f)
        except FileNotFoundError:
            self.img_ids, self._label_info = CelebAIDImageFolder._read_images(attr_fn, drop_infreq=drop_infreq)
            with open(pkl_fn, 'wb') as handle:
                # use protocol 4 to be backwards compatible up to Python 3.4, but still allow large pickle files
                pickle.dump((self.img_ids, self._label_info), handle, protocol=4)

        assert len(self.samples) == len(self.img_ids)

        # throw out the class parsed by the parent class
        self.samples = [path for path, _cls in self.samples]

    @staticmethod
    def _update_cache_name(pkl_fn, name):
        """
        Add <name> to <pkl_fn>.
        """
        pkl_fn, _ = pkl_fn.split(".pickle")
        return f"{pkl_fn}_{name}.pickle"

    @staticmethod
    def get_cache_name(pkl_fn, img_size, drop_infreq):
        assert pkl_fn.endswith(".pickle")

        pkl_fn = CelebAIDImageFolder._update_cache_name(pkl_fn, f"img{img_size}")

        if drop_infreq is not None:
            pkl_fn = CelebAIDImageFolder._update_cache_name(pkl_fn, f"drop_infreq{drop_infreq:.4f}")

        return pkl_fn

    @staticmethod
    def _read_images(attr_fn, drop_infreq):
        """
        Get a dictionary mapping paths to attributes.
        """
        img_ids = pd.read_csv(attr_fn, delim_whitespace=True, skiprows=1)

        assert len(img_ids) == 202599

        for col in img_ids.columns:
            # convert labels to binary from -1, 1
            img_ids[col] = (img_ids[col] + 1) // 2

        class_freqs = [img_ids[col_name].values.mean() * 100 for col_name in img_ids.columns]

        for col_num, (column, freq) in enumerate(zip(img_ids.columns, class_freqs)):
            if drop_infreq is not None:
                if freq < drop_infreq:
                    img_ids.drop(column, axis=1, inplace=True)

        # recompute class freqs now that we've dropped some of them
        class_freqs = [img_ids[col_name].values.mean() * 100 for col_name in img_ids.columns]

        _label_info = dict(zip(img_ids.columns, enumerate(class_freqs)))

        # transpose to get index as column, create dict mapping {path_name: list}
        img_ids = img_ids.T.to_dict(orient="list")

        # transform type of values in img_ids dict
        img_ids = {k: tuple(v) for k, v in img_ids.items()}

        return img_ids, _label_info

    @staticmethod
    def get_img_id(path):
        # define the ID of an image
        return os.path.basename(path)

    def __getitem__(self, index):
        """
        Given an index for the dataset, return the element at that index
        """
        # load in image path and target
        path = self.samples[index]

        # load the image sample
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(self.img_ids[CelebAIDImageFolder.get_img_id(path)], dtype=torch.int64)

        return sample, target


class UTZapposIDImageFolder(datasets.ImageFolder):
    def __init__(self, attr_fn, pkl_fn, img_size, observed=False, binarized=False, drop_infreq=None, log=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        pkl_fn = UTZapposIDImageFolder.get_cache_name(pkl_fn, img_size, observed, binarized, drop_infreq)

        try:
            with open(pkl_fn, 'rb') as f:
                self.img_ids, self._label_info = pickle.load(f)
        except FileNotFoundError:
            self.img_ids, self._label_info = UTZapposIDImageFolder._read_images(attr_fn, observed, binarized,
                                                                                drop_infreq=drop_infreq, log=log)
            with open(pkl_fn, 'wb') as handle:
                # use protocol 4 to be backwards compatible up to Python 3.4, but still allow large pickle files
                pickle.dump((self.img_ids, self._label_info), handle, protocol=4)

        # throw out the class
        _samples = [path for path, _cls in self.samples]

        # the basename of the file corresponds to its ID
        # remove duplicate IDs by constructing dict with IDs as keys
        # use an OrderedDict for reproducibility between runs, assuming self.samples is always the same order
        _samples = OrderedDict(zip(map(UTZapposIDImageFolder.get_img_id, _samples), _samples))

        # remove any samples which were filtered out when constructing img_ids
        # use the full path and throw out the ID for now
        self.samples = [v for k, v in _samples.items() if k in self.img_ids]

    @staticmethod
    def _update_cache_name(pkl_fn, name):
        """
        Add <name> to <pkl_fn>.
        """
        pkl_fn, _ = pkl_fn.split(".pickle")
        return f"{pkl_fn}_{name}.pickle"

    @staticmethod
    def get_cache_name(pkl_fn, img_size, observed, binarized, drop_infreq):
        assert pkl_fn.endswith(".pickle")

        pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, f"img{img_size}")

        if observed:
            pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, "observed")

        if binarized:
            pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, "binarized")

        if drop_infreq is not None:
            pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, f"drop_infreq{drop_infreq:.4f}")

        return pkl_fn

    @staticmethod
    def _get_unique_multicol(img_ids, col):
        return order_set(list(chain.from_iterable(map(lambda x: x.split(";"), filter(lambda x: isinstance(x, str),
                                                                                     img_ids[col].values)))))

    @staticmethod
    def _get_option_col(img_ids, col_val, col_name):
        """
        Return a column with binarized version of img_ids[col_name] for attribute value col_val.
        """
        def _binarize_attribute(val):
            if isinstance(val, str):
                return 1 if col_val in val else 0
            return -1  # if it's not str, then it's missing

        return img_ids[col_name].apply(_binarize_attribute)

    @staticmethod
    def _read_images(attr_fn, observed, binarized, drop_infreq=None, log=False):
        """
        Get a dictionary mapping paths to attributes.
        """
        # columns with several options separated by ";", which we binarize by manual processing
        _manually_binarized_cols = ['Gender', 'Material', 'Closure', 'Insole', 'ToeStyle']
        # columns to be binarized normally (need to save them now since col names will change)
        _binarized_cols = ['Category', 'SubCategory', 'HeelHeight']
        # columns with several options separated by ";". take the first option if we're not binarizing
        _multiple_val_cols = ['Insole', 'Closure', 'Material', 'ToeStyle']
        # columns dropped because they have too many missing values
        _observed_drop_cols = ['HeelHeight', 'Insole', 'ToeStyle']
        # drop rows with missing values in these columns (Category and SubCategory don't actually have any)
        _observed_drop_rows = ['Gender', 'Material', 'Closure', 'Category', 'SubCategory']

        img_ids = pd.read_csv(attr_fn)

        assert set(_manually_binarized_cols + _binarized_cols) == set(col for col in img_ids.columns if col != 'CID')

        assert set(_observed_drop_rows + _observed_drop_cols) == set(col for col in img_ids.columns if col != 'CID')
        assert len(set(_observed_drop_rows).intersection(set(_observed_drop_cols))) == 0

        def split_cid(cid):
            return '.'.join(cid.split('-')) + ".jpg"

        def process_multiple(category):
            if not isinstance(category, str):
                assert isinstance(category, float)
                assert math.isnan(category)
                return category
            return category.split(';')[0]  # return the prominent category if there are multiple

        img_ids['CID'] = img_ids['CID'].apply(split_cid)

        img_ids = img_ids.set_index('CID')

        _label_info = None

        if binarized:
            for column in _manually_binarized_cols:
                if observed:
                    if column in _observed_drop_cols:
                        # don't binarize columns we'll drop
                        continue
                    # may as well drop the rows with missing values now
                    img_ids = img_ids[(~img_ids[column].isnull())]

                options = UTZapposIDImageFolder._get_unique_multicol(img_ids, column)
                for option in options:
                    # add new column
                    img_ids[f"{column}_{option}"] = UTZapposIDImageFolder._get_option_col(img_ids, option, column)
                img_ids.drop(column, axis=1, inplace=True)
        else:
            for column in _multiple_val_cols:
                img_ids[column] = img_ids[column].apply(process_multiple)

        if not binarized:
            for column in img_ids.columns:

                # convert to categorical data-type, even if we binarize it later on
                categorical_column = img_ids[column].astype('category').cat

                print(f"{column} ({len(categorical_column.categories)})")
                print('\t' + '\n\t'.join(map(str, enumerate(categorical_column.categories))))

                # use the numerical codes (-1 for nan)
                img_ids[column] = categorical_column.codes

        if observed:
            # drop columns with too many unobserved variables
            img_ids.drop(_observed_drop_cols, axis=1, inplace=True)

            if not binarized:  # if binarized, then these columns don't exist
                # drop any rows which still have missing values
                for col in _observed_drop_rows:
                    img_ids = img_ids[img_ids[col] == -1]

        if binarized:
            # turn categorical columns into binarized columns
            for column in _binarized_cols:

                if column not in img_ids.columns:
                    continue  # this column has been removed (e.g. due to not enough observed values)

                binarized_columns = pd.get_dummies(img_ids[column], prefix=column)
                img_ids.drop(column, axis=1, inplace=True)
                img_ids = img_ids.join(binarized_columns, how="outer")

            # classes with f1=0 on fixed batch of train data
            # unbalanced_classes = [7, 10, 11, 12, 14, 18, 20, 22, 25, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40,
            #                       41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            #                       62, 63, 64, 65, 66, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 91, 93, 95, 96, 99,
            #                       104, 106, 107, 109]
            unbalanced_classes = [10, 18, 20, 29, 35, 36, 37, 38, 39, 40, 41, 46, 47, 50, 51, 52, 55, 57, 59, 60, 61,
                                  63, 65, 66, 79, 83, 84, 85, 91, 93, 95, 104, 107, 109]

            def class_freq(col_name):
                # get the percentage of positive labels for a given column
                return img_ids[col_name].values.mean() * 100

            class_freqs = list(map(class_freq, img_ids.columns))

            f1_0_freqs = [freq for col_num, freq in enumerate(class_freqs) if col_num in unbalanced_classes]
            if log:
                print(min(f1_0_freqs), max(f1_0_freqs), np.mean(f1_0_freqs), np.median(f1_0_freqs))
            for col_num, (column, freq) in enumerate(zip(img_ids.columns, class_freqs)):
                if drop_infreq is not None:
                    if freq < drop_infreq:
                        img_ids.drop(column, axis=1, inplace=True)
                if log:
                    print(f"{col_num} {column}\t{freq:.4f}")

            new_class_freqs = list(map(class_freq, img_ids.columns))
            if log:
                print(min(new_class_freqs), max(new_class_freqs), np.mean(new_class_freqs), np.median(new_class_freqs))

            # log the columns at the end
            _label_info = dict(zip(img_ids.columns, enumerate(new_class_freqs)))
            for column, (column_num, freq) in _label_info.items():
                print(f"{column_num} {column}\t{freq:.4f}")

            if log:
                # log the frequencies in sorted order
                argsort_freqs = np.argsort(new_class_freqs)
                for ind in argsort_freqs:
                    print(f"{ind} {img_ids.columns[ind]}\t{new_class_freqs[ind]}")

        # transpose to get index as column, create dict mapping {path_name: list}
        img_ids = img_ids.T.to_dict(orient="list")

        # transform type of values in img_ids dict
        img_ids = {k: tuple(v) for k, v in img_ids.items()}

        return img_ids, _label_info

    @staticmethod
    def get_img_id(path):
        # define the ID of an image
        return os.path.basename(path)

    def __getitem__(self, index):
        """
        Given an index for the dataset, return the element at that index
        """
        # load in image path and target
        path = self.samples[index]

        # load the image sample
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(self.img_ids[UTZapposIDImageFolder.get_img_id(path)], dtype=torch.int64)

        return sample, target


class IDImageFolder(datasets.ImageFolder):
    def __init__(self, attributes_file=None,
                 images_file=None,
                 pkl_file=None,
                 bounding_boxes_file=None,
                 *args, **kwargs):

        assert ((attributes_file is None or images_file is None) != (pkl_file is None)), \
            "images and attributes files XOR pkl file"

        super().__init__(*args, **kwargs)

        if pkl_file is None:
            self.img_ids = IDImageFolder._read_images(images_file)
            self.im_attributes, self.im_certainties = IDImageFolder._read_attributes(attributes_file)
            self.bounding_boxes = IDImageFolder._read_bounding_boxes(bounding_boxes_file)
        else:
            with open(pkl_file, 'rb') as f:
                self.img_ids, self.im_attributes, self.im_certainties, self.bounding_boxes = pickle.load(f)

    @staticmethod
    def _read_images(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            im_ids, fnames = zip(*lines)
            im_ids = [int(im_id) - 1 for im_id in im_ids]
            d = {im_fname: im_id for im_id, im_fname in zip(im_ids, fnames)}
            return d

    @staticmethod
    def _read_attributes(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            n_att = 312
            n_im = int(len(lines) / n_att)
            att_tensor = np.zeros((n_im, n_att), dtype=int)
            cert_tensor = np.zeros((n_im, n_att), dtype=int)
            for line in lines:
                ls = line.strip().split()
                im_id, att_id, att_val, cert_val, time = int(ls[0]) - 1, int(ls[1]) - 1, \
                                                         int(ls[2]), int(ls[3]), float(ls[4])
                att_tensor[im_id, att_id] = att_val
                cert_tensor[im_id, att_id] = cert_val

        return att_tensor, cert_tensor

    @staticmethod
    def _read_bounding_boxes(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            n_im = len(lines)
            bounding_boxes = np.zeros((n_im, 4), dtype=float)
            for line in lines:
                im_id, *bounding_box = map(lambda x: int(float(x)), line.strip().split())
                im_id -= 1  # convert im_id from 1-index to 0-index
                bounding_boxes[im_id, :] = bounding_box

        return bounding_boxes

    def _crop_bounding_box(self, im_id, sample):
        x, y, width, height = self.bounding_boxes[im_id]

        left = x
        right = x + width
        top = y
        bottom = y + height

        sample = sample.crop((left, top, right, bottom))

        return sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # load in image and target
        path, target = self.samples[index]
        sample = self.loader(path)

        # load in attributes
        spath = "/".join(path.split('/')[-2:])
        im_id = self.img_ids[spath]
        att = self.im_attributes[im_id]
        cert = self.im_certainties[im_id]

        # crop the image to the bounding box
        sample = self._crop_bounding_box(im_id, sample)

        # transform the sample
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, att, cert


def utzappos_tensor_dset(img_size, observed, binarized, drop_infreq,
                         cache_fn, *dset_args, transform=None, **dset_kwargs):
    """
    Convert folder dataset to tensor dataset.
    """
    cache_fn = UTZapposIDImageFolder.get_cache_name(cache_fn, img_size, observed, binarized, drop_infreq)
    try:
        with open(cache_fn, 'rb') as f:
            dset_samples, dset_labels, dset_label_info = pickle.load(f)
    except FileNotFoundError:
        img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                        torchvision.transforms.ToTensor()])
        dset = UTZapposIDImageFolder(*dset_args, img_size=img_size, transform=img_transform,
                                     observed=observed, binarized=binarized, drop_infreq=drop_infreq,
                                     **dset_kwargs)
        dset_examples = [dset[ind] for ind in range(len(dset))]
        dset_samples, dset_labels = map(torch.stack, zip(*dset_examples))
        # find_duplicates_in_dsets((dset_samples, dset_labels), (dset_samples, dset_labels),
        #                          tuple_format=True, itself=True)
        dset_label_info = dset._label_info
        with open(cache_fn, 'wb') as handle:
            pickle.dump((dset_samples, dset_labels, dset_label_info), handle, protocol=4)

    return CustomTensorDataset(dset_samples, dset_labels, transform=transform), dset_label_info, cache_fn


def split_utzappos(dset, split_len, root, cache_fn,
                   split_type="random",
                   balanced_freq_thres=0.02,
                   balanced_split_ratio=-1):
    """
    Split UTZappos into train and test sets.
    """
    dataset_size = len(dset)
    if split_type == "random":
        indices = torch.randperm(dataset_size)
        trn_indices, tst_indices = indices[split_len:], indices[:split_len]
    elif split_type == "balanced":
        if balanced_split_ratio <= 0:
            balanced_split_ratio = split_len / dataset_size

        cache_fn = os.path.join(root, f"balanced_split_{balanced_split_ratio}_{balanced_freq_thres:.4f}_"
                                      f"{os.path.basename(cache_fn)}")
        try:
            with open(cache_fn, 'rb') as f:
                trn_indices, tst_indices = pickle.load(f)
        except FileNotFoundError:
            if balanced_split_ratio <= 0:
                balanced_split_ratio = split_len / dataset_size
            dset_samples, dset_labels = dset[:]
            rare_classes = dset_labels.float().mean(0) < balanced_freq_thres
            if rare_classes.any():
                trn_rare_indices, tst_rare_indices = zip(*map(partial(_get_rare_class_inds_split,
                                                                      dset_labels, balanced_split_ratio),
                                                              rare_classes.nonzero(as_tuple=False)))

                # aggregate the splits across attributes
                trn_rare_indices = list(chain.from_iterable(trn_rare_indices))
                tst_rare_indices = list(chain.from_iterable(tst_rare_indices))

                # remove duplicate indices (different attributes may place the same index in the same split)
                trn_rare_indices, tst_rare_indices = set(trn_rare_indices), set(tst_rare_indices)

                # remove conflicting indices (different attributes may place the same index in opposite splits)
                conflicting_rare_indices = trn_rare_indices.intersection(tst_rare_indices)
                trn_rare_indices = trn_rare_indices.difference(conflicting_rare_indices)
                tst_rare_indices = tst_rare_indices.difference(trn_rare_indices)

                # add conflicting indices to the test set (we want to make sure we have enough positive examples)
                tst_rare_indices = tst_rare_indices.union(conflicting_rare_indices)

                # aggregated split should be disjoint now
                assert len(trn_rare_indices.intersection(tst_rare_indices)) == 0

                # change types
                trn_rare_indices = torch.tensor(list(trn_rare_indices))
                tst_rare_indices = torch.tensor(list(tst_rare_indices))

                # track all indices split for rare classes
                rare_indices = torch.cat([trn_rare_indices, tst_rare_indices])
            else:
                trn_rare_indices = torch.tensor([], dtype=torch.int64)
                tst_rare_indices = torch.tensor([], dtype=torch.int64)
                rare_indices = torch.tensor([], dtype=torch.int64)

            # get indices not split already by rare classes
            nonrare_indices = torch.tensor([ind for ind in range(dataset_size) if ind not in rare_indices])

            assert len(nonrare_indices) + len(rare_indices) == dataset_size

            # shuffle
            nonrare_indices = nonrare_indices[torch.randperm(len(nonrare_indices))]

            # calculate remaining indices needed for the test set
            split_len_remaining = split_len - len(tst_rare_indices)
            assert split_len_remaining > 0, "Too many even rare class splits"

            trn_nonrare_indices = nonrare_indices[split_len_remaining:]
            tst_nonrare_indices = nonrare_indices[:split_len_remaining]

            # combine indices for rare and non-rare
            trn_indices = torch.cat([trn_rare_indices, trn_nonrare_indices])
            tst_indices = torch.cat([tst_rare_indices, tst_nonrare_indices])

            assert len(tst_indices) == split_len
            assert len(trn_indices) + len(tst_indices) == dataset_size
            assert set(trn_indices).isdisjoint(set(tst_indices))

            with open(cache_fn, 'wb') as handle:
                pickle.dump((trn_indices, tst_indices), handle, protocol=4)
    else:
        raise ValueError(f"Unrecognized split type {split_type}")

    return CustomTensorDataset(*dset[trn_indices]), CustomTensorDataset(*dset[tst_indices])


def _get_rare_class_inds_split(dset_labels, split_ratio, rare_class):
    # get indices of rare class
    rare_class_inds = (dset_labels[:, rare_class].squeeze() == 1).nonzero(as_tuple=False).squeeze()

    # shuffle
    rare_class_inds = rare_class_inds[torch.randperm(len(rare_class_inds))]

    # calculate number to split based on given split ratio
    rare_split_len = int(split_ratio * len(rare_class_inds))

    # return inds for split for this rare class
    return list(rare_class_inds[rare_split_len:].numpy()), list(rare_class_inds[:rare_split_len].numpy())


def celeba_tensor_dset(img_size, drop_infreq, cache_fn, *dset_args, transform=None, **dset_kwargs):
    """
    Convert folder dataset to tensor dataset.
    """
    cache_fn = CelebAIDImageFolder.get_cache_name(cache_fn, img_size, drop_infreq)
    try:
        with open(cache_fn, 'rb') as f:
            dset_samples, dset_labels, dset_label_info = pickle.load(f)
    except FileNotFoundError:
        img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                        torchvision.transforms.ToTensor()])
        dset = CelebAIDImageFolder(*dset_args, img_size=img_size, transform=img_transform, drop_infreq=drop_infreq,
                                   **dset_kwargs)
        dset_examples = [dset[ind] for ind in range(len(dset))]
        dset_samples, dset_labels = map(torch.stack, zip(*dset_examples))
        dset_label_info = dset._label_info
        with open(cache_fn, 'wb') as handle:
            pickle.dump((dset_samples, dset_labels, dset_label_info), handle, protocol=4)

    # tpl_labels = [tuple(lbl.numpy()) for lbl in dset_labels]
    # unique_labels = set(tpl_labels)
    #
    # # remove 15, 24
    # pruned_labels = [lbl[:15] + lbl[16:24] + lbl[25:] for lbl in tpl_labels]
    # unique_pruned_labels = set(pruned_labels)
    #
    # # remove 5, 15, 20, 24
    # more_pruned_labels = [lbl[:5] + lbl[6:15] + lbl[16:20] + lbl[21:24] + lbl[25:] for lbl in tpl_labels]
    # unique_more_pruned_labels = set(more_pruned_labels)
    #
    # # remove 0, 15, 24
    # some_pruned_labels = [lbl[1:15] + lbl[16:24] + lbl[25:] for lbl in tpl_labels]
    # unique_some_pruned_labels = set(some_pruned_labels)

    return CustomTensorDataset(dset_samples, dset_labels, transform=transform), dset_label_info, cache_fn


def _check_combos_in_test_set(tst_indices, freq_and_combos, test_combos, dset_label_info, dset_labels):
    for combo_freq, combo in freq_and_combos:
        combo_inds = _get_combo_inds(dset_labels, dset_label_info, combo)
        combo_in_test_combos = combo in test_combos
        subset_of_tst = tst_indices.issuperset(combo_inds)
        if combo_in_test_combos and not subset_of_tst:
            print(f"Missing from test: {combo}")
        elif not combo_in_test_combos and subset_of_tst:
            print(f"Extra combo added: {combo}")


def find_combos_in_tst_set(trn_labels, tst_labels, dset_label_info, combos):
    """
    Interface to find combinations used in test set.
    """
    tst_combos = []
    for combo in combos:
        if _get_combo_freq(trn_labels, dset_label_info, combo) == 0 and \
                _get_combo_freq(tst_labels, dset_label_info, combo) > 0:
            tst_combos.append(combo)
    return tst_combos


def lbl_in_combos(tst_labels, dset_label_info, combos):
    """
    Return mask
    """
    onehot_labels = torch.zeros(tst_labels.shape[0], len(combos), dtype=tst_labels.dtype, device=tst_labels.device)
    for combo_ind, combo in enumerate(combos):
        onehot_labels[:, combo_ind] = _filter_to_combo(tst_labels, dset_label_info, combo)
    return onehot_labels


def split_celeba(dset, split_len, root, cache_fn,
                 split_type="random",
                 balanced_freq_thres=0.02,
                 balanced_split_ratio=-1,
                 zero_shot_split_version=1,
                 zero_shot_split_trn_prob=1.,
                 zero_shot_split_trn_hard_thres=.65,
                 zero_shot_split_tst_hard_thres=.005,
                 zero_shot_split_thres_no_split=.05,
                 zero_shot_split_unsplit_ratio=0,
                 dset_label_info=None):
    """
    Split UTZappos into train and test sets.
    """
    dataset_size = len(dset)
    if split_type == "random":
        indices = torch.randperm(dataset_size)
        trn_indices, tst_indices = indices[split_len:], indices[:split_len]
    elif split_type == "balanced":
        if balanced_split_ratio <= 0:
            balanced_split_ratio = split_len / dataset_size

        cache_fn = os.path.join(root, f"balanced_split_{balanced_split_ratio}_{balanced_freq_thres:.4f}_"
                                      f"{os.path.basename(cache_fn)}")
        try:
            with open(cache_fn, 'rb') as f:
                trn_indices, tst_indices = pickle.load(f)
        except FileNotFoundError:
            if balanced_split_ratio <= 0:
                balanced_split_ratio = split_len / dataset_size
            dset_samples, dset_labels = dset[:]
            rare_classes = dset_labels.float().mean(0) < balanced_freq_thres
            if rare_classes.any():
                trn_rare_indices, tst_rare_indices = zip(*map(partial(_get_rare_class_inds_split,
                                                                      dset_labels, balanced_split_ratio),
                                                              rare_classes.nonzero(as_tuple=False)))

                # aggregate the splits across attributes
                trn_rare_indices = list(chain.from_iterable(trn_rare_indices))
                tst_rare_indices = list(chain.from_iterable(tst_rare_indices))

                # remove duplicate indices (different attributes may place the same index in the same split)
                trn_rare_indices, tst_rare_indices = set(trn_rare_indices), set(tst_rare_indices)

                # remove conflicting indices (different attributes may place the same index in opposite splits)
                conflicting_rare_indices = trn_rare_indices.intersection(tst_rare_indices)
                trn_rare_indices = trn_rare_indices.difference(conflicting_rare_indices)
                tst_rare_indices = tst_rare_indices.difference(trn_rare_indices)

                # add conflicting indices to the test set (we want to make sure we have enough positive examples)
                tst_rare_indices = tst_rare_indices.union(conflicting_rare_indices)

                # aggregated split should be disjoint now
                assert len(trn_rare_indices.intersection(tst_rare_indices)) == 0

                # change types
                trn_rare_indices = torch.tensor(list(trn_rare_indices))
                tst_rare_indices = torch.tensor(list(tst_rare_indices))

                # track all indices split for rare classes
                rare_indices = torch.cat([trn_rare_indices, tst_rare_indices])
            else:
                trn_rare_indices = torch.tensor([], dtype=torch.int64)
                tst_rare_indices = torch.tensor([], dtype=torch.int64)
                rare_indices = torch.tensor([], dtype=torch.int64)

            # get indices not split already by rare classes
            nonrare_indices = torch.tensor([ind for ind in range(dataset_size) if ind not in rare_indices])

            assert len(nonrare_indices) + len(rare_indices) == dataset_size

            # shuffle
            nonrare_indices = nonrare_indices[torch.randperm(len(nonrare_indices))]

            # calculate remaining indices needed for the test set
            split_len_remaining = split_len - len(tst_rare_indices)
            assert split_len_remaining > 0, "Too many even rare class splits"

            trn_nonrare_indices = nonrare_indices[split_len_remaining:]
            tst_nonrare_indices = nonrare_indices[:split_len_remaining]

            # combine indices for rare and non-rare
            trn_indices = torch.cat([trn_rare_indices, trn_nonrare_indices])
            tst_indices = torch.cat([tst_rare_indices, tst_nonrare_indices])

            assert len(tst_indices) == split_len
            assert len(trn_indices) + len(tst_indices) == dataset_size

            with open(cache_fn, 'wb') as handle:
                pickle.dump((trn_indices, tst_indices), handle, protocol=4)
    elif split_type == "zero_shot":

        if dset_label_info is None:
            raise ValueError(f"dset_label_info is required for zero_shot split, got {dset_label_info}.")

        # TODO: update fname with config
        cache_fn = os.path.join(root, f"zero_shot_split_{zero_shot_split_version}_{zero_shot_split_trn_prob:.4f}_"
                                      f"{os.path.basename(cache_fn)}")
        try:
            with open(cache_fn, 'rb') as f:
                trn_indices, tst_indices = pickle.load(f)
        except FileNotFoundError:
            dset_samples, dset_labels = dset[:]

            freq_and_combos = list(sorted(zip(map(partial(_get_combo_freq, dset_labels, dset_label_info),
                                                  CELEBA_ZERO_SHOT_COMBOS),
                                              CELEBA_ZERO_SHOT_COMBOS), reverse=True))

            rng = np.random.RandomState(0)
            trn_indices, tst_indices = set(), set()

            tst_combos = []

            # assign each combination to train or test split
            for combo_freq, combo in freq_and_combos:
                combo_inds = _get_combo_inds(dset_labels, dset_label_info, combo)
                # noinspection PyArgumentList
                trn_tst_choice = rng.rand()  # always call this, even if we don't use it, to keep the same seed
                if combo_freq < zero_shot_split_thres_no_split * len(dset_labels):
                    if combo_freq < zero_shot_split_tst_hard_thres * len(dset_labels) and \
                            (trn_tst_choice < zero_shot_split_trn_prob or
                             combo_freq > zero_shot_split_trn_hard_thres * len(dset_labels)):
                        # put it in the train set if chosen, or if it's significant portion of entire dataset
                        trn_indices = trn_indices.union(combo_inds)
                    else:
                        tst_combos.append(combo)
                        print(f"Test: {combo} {combo_freq}")
                        tst_indices = tst_indices.union(combo_inds)

            _check_combos_in_test_set(tst_indices, freq_and_combos, tst_combos, dset_label_info, dset_labels)

            # any indices in both trn and tst are now only in tst
            # this maintains all held-out attribute combinations, at the cost of losing some (not held-out) attribute
            # combinations for training
            conflicting_indices = trn_indices.intersection(tst_indices)
            trn_indices = trn_indices.difference(conflicting_indices)

            _check_combos_in_test_set(tst_indices, freq_and_combos, tst_combos, dset_label_info, dset_labels)

            # any indices which haven't been split on yet will be placed randomly
            unsplit_indices = list(set(range(len(dset_labels))).difference(trn_indices.union(tst_indices)))
            unsplit_indices = torch.tensor(unsplit_indices, dtype=torch.int64)[torch.randperm(len(unsplit_indices))]
            unsplit_tst_len = int(zero_shot_split_unsplit_ratio * len(unsplit_indices))
            trn_unsplit_indices = unsplit_indices[unsplit_tst_len:]
            tst_unsplit_indices = unsplit_indices[:unsplit_tst_len]

            # add partitions of unsplit indices to respective partitions
            trn_indices = trn_indices.union(list(trn_unsplit_indices.numpy()))
            tst_indices = tst_indices.union(list(tst_unsplit_indices.numpy()))

            # check trn and tst is a valid partition (disjoint and covers the entire set)
            assert trn_indices.isdisjoint(tst_indices)
            # check that indices cover the entire set
            all_indices = trn_indices.union(tst_indices)
            assert len(all_indices) == len(dset_labels)
            assert min(all_indices) == 0 and max(all_indices) == len(all_indices) - 1

            _check_combos_in_test_set(tst_indices, freq_and_combos, tst_combos, dset_label_info, dset_labels)

            print(len(trn_indices), len(tst_indices), len(tst_combos))

            # cast types
            trn_indices = torch.tensor(list(trn_indices))
            tst_indices = torch.tensor(list(tst_indices))

            # shuffle the test set, since we're only going to evaluate on part of it since it's so big
            tst_indices = tst_indices[torch.randperm(len(tst_indices))]

            with open(cache_fn, 'wb') as handle:
                pickle.dump((trn_indices, tst_indices), handle, protocol=4)
    else:
        raise ValueError(f"Unrecognized split type {split_type}")

    return CustomTensorDataset(*dset[trn_indices]), CustomTensorDataset(*dset[tst_indices])


class CUBIDImageFolder(datasets.ImageFolder):
    def __init__(self, attr_fn, attr_name_fn, img_id_fn, bb_fn, pkl_fn, img_size, drop_infreq=None, log=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        pkl_fn = CUBIDImageFolder.get_cache_name(pkl_fn, img_size, drop_infreq)

        try:
            with open(pkl_fn, 'rb') as f:
                self.img_ids, self._label_info, self._img_id_nums = pickle.load(f)
        except FileNotFoundError:
            self.img_ids, self._label_info, self._img_id_nums = CUBIDImageFolder._read_images(attr_fn,
                                                                                              attr_name_fn,
                                                                                              img_id_fn,
                                                                                              drop_infreq=drop_infreq,
                                                                                              log=log)
            with open(pkl_fn, 'wb') as handle:
                # use protocol 4 to be backwards compatible up to Python 3.4, but still allow large pickle files
                pickle.dump((self.img_ids, self._label_info, self._img_id_nums), handle, protocol=4)

        assert len(self.samples) == len(self.img_ids)

        # throw out the class parsed by the parent class
        self.samples = [path for path, _cls in self.samples]

        # load bounding boxes
        self.bounding_boxes = CUBIDImageFolder._read_bounding_boxes(bb_fn)

    @staticmethod
    def _update_cache_name(pkl_fn, name):
        """
        Add <name> to <pkl_fn>.
        """
        pkl_fn, _ = pkl_fn.split(".pickle")
        return f"{pkl_fn}_{name}.pickle"

    @staticmethod
    def get_cache_name(pkl_fn, img_size, drop_infreq):
        assert pkl_fn.endswith(".pickle")

        pkl_fn = CUBIDImageFolder._update_cache_name(pkl_fn, f"img{img_size}")

        if drop_infreq is not None:
            pkl_fn = CUBIDImageFolder._update_cache_name(pkl_fn, f"drop_infreq{drop_infreq:.4f}")

        return pkl_fn

    @staticmethod
    def _read_images(attr_fn, attr_name_fn, img_id_fn, drop_infreq=False, log=False):
        """
        Get a dictionary mapping paths to attributes.
        """
        attr_num_col, attr_name_col = "attr_num", "attr_name"
        attr_names = pd.read_csv(attr_name_fn, delim_whitespace=True, names=[attr_num_col, attr_name_col])

        img_id_col, img_pth_col = "img_id", "img_pth"
        img_id_names = pd.read_csv(img_id_fn, delim_whitespace=True, names=[img_id_col, img_pth_col])
        img_id_names[img_pth_col] = img_id_names[img_pth_col].apply(CUBIDImageFolder.get_img_id)

        attr_col = "attr"
        img_ids = pd.read_csv(attr_fn, delim_whitespace=True, usecols=(0, 1, 2),
                              names=[img_id_col, attr_num_col, attr_col])

        assert len(img_ids) == 3677856
        assert len(img_id_names) == 11788
        assert len(attr_names) == 312

        assert len(attr_names) * len(img_id_names) == len(img_ids)

        # now there is a column for each attribute saying whether it's there or not
        img_ids = img_ids.pivot_table(attr_col, [img_id_col], attr_num_col)

        assert img_ids.values.shape == (len(img_id_names), len(attr_names))

        # rename columns with attribute names, rows with images paths
        img_ids = img_ids.rename(columns=dict(zip(attr_names[attr_num_col], attr_names[attr_name_col])),
                                 index=dict(zip(img_id_names[img_id_col], img_id_names[img_pth_col])))

        # collect label frequencies
        class_freqs = [img_ids[col_name].values.mean() * 100 for col_name in img_ids.columns]

        for column, freq in zip(img_ids.columns, class_freqs):
            if drop_infreq is not None:
                if freq < drop_infreq:
                    img_ids.drop(column, axis=1, inplace=True)

        class_freqs = [img_ids[col_name].values.mean() * 100 for col_name in img_ids.columns]
        _label_info = dict(zip(img_ids.columns, enumerate(class_freqs)))

        if log:
            # log the frequencies in sorted order
            argsort_freqs = np.argsort(class_freqs)
            for ind in argsort_freqs:
                print(f"{ind} {img_ids.columns[ind]}\t{class_freqs[ind]}")

            print(len(class_freqs))

        # transpose to get index as column, create dict mapping {path_name: list}
        img_ids = img_ids.T.to_dict(orient="list")

        # transform type of values in img_ids dict
        img_ids = {k: tuple(v) for k, v in img_ids.items()}

        _img_id_nums = dict(zip(img_id_names[img_pth_col], img_id_names[img_id_col]))
        return img_ids, _label_info, _img_id_nums

    @staticmethod
    def _read_bounding_boxes(fname):
        bb = pd.read_csv(fname, delim_whitespace=True, names=['id', 'x', 'y', 'w', 'h'])
        assert len(bb) == 11788
        bb = bb.set_index('id').T.to_dict(orient="list")
        return bb

    def _crop_bounding_box(self, im_id, sample):
        x, y, width, height = self.bounding_boxes[im_id]

        left = x
        right = x + width
        top = y
        bottom = y + height

        sample = sample.crop((left, top, right, bottom))

        return sample

    @staticmethod
    def get_img_id(path):
        # define the ID of an image
        return os.path.basename(path)

    def _get_img_id_num(self, path):
        return self._img_id_nums[CUBIDImageFolder.get_img_id(path)]

    def __getitem__(self, index):
        """
        Given an index for the dataset, return the element at that index
        """
        # load in image path and target
        path = self.samples[index]

        # load the image sample
        sample = self.loader(path)

        # crop the image to the bounding box
        sample = self._crop_bounding_box(self._get_img_id_num(path), sample)

        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(self.img_ids[CUBIDImageFolder.get_img_id(path)], dtype=torch.int64)

        return sample, target


def cub_tensor_dset(img_size, drop_infreq, cache_fn, *dset_args, transform=None, **dset_kwargs):
    """
    Convert folder dataset to tensor dataset.
    """
    cache_fn = CUBIDImageFolder.get_cache_name(cache_fn, img_size, drop_infreq)
    try:
        with open(cache_fn, 'rb') as f:
            dset_samples, dset_labels, dset_label_info = pickle.load(f)
    except FileNotFoundError:
        img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                        torchvision.transforms.ToTensor()])
        dset = CUBIDImageFolder(*dset_args, img_size=img_size, transform=img_transform, drop_infreq=drop_infreq,
                                     **dset_kwargs)
        dset_examples = [dset[ind] for ind in range(len(dset))]
        dset_samples, dset_labels = map(torch.stack, zip(*dset_examples))
        dset_label_info = dset._label_info
        with open(cache_fn, 'wb') as handle:
            pickle.dump((dset_samples, dset_labels, dset_label_info), handle, protocol=4)

    return CustomTensorDataset(dset_samples, dset_labels, transform=transform), dset_label_info, cache_fn


def split_cub(dset, split_len, root, cache_fn,
              split_type="random",
              balanced_freq_thres=0.02,
              balanced_split_ratio=-1):
    """
    Split CUB into train and test sets.
    """
    dataset_size = len(dset)
    if split_type == "random":
        indices = torch.randperm(dataset_size)
        trn_indices, tst_indices = indices[split_len:], indices[:split_len]
    elif split_type == "balanced":
        if balanced_split_ratio <= 0:
            balanced_split_ratio = split_len / dataset_size

        cache_fn = os.path.join(root, f"balanced_split_{balanced_split_ratio}_{balanced_freq_thres:.4f}_"
                                      f"{os.path.basename(cache_fn)}")
        try:
            with open(cache_fn, 'rb') as f:
                trn_indices, tst_indices = pickle.load(f)
        except FileNotFoundError:
            if balanced_split_ratio <= 0:
                balanced_split_ratio = split_len / dataset_size
            dset_samples, dset_labels = dset[:]
            rare_classes = dset_labels.float().mean(0) < balanced_freq_thres
            if rare_classes.any():
                trn_rare_indices, tst_rare_indices = zip(*map(partial(_get_rare_class_inds_split,
                                                                      dset_labels, balanced_split_ratio),
                                                              rare_classes.nonzero(as_tuple=False)))

                # aggregate the splits across attributes
                trn_rare_indices = list(chain.from_iterable(trn_rare_indices))
                tst_rare_indices = list(chain.from_iterable(tst_rare_indices))

                # remove duplicate indices (different attributes may place the same index in the same split)
                trn_rare_indices, tst_rare_indices = set(trn_rare_indices), set(tst_rare_indices)

                # remove conflicting indices (different attributes may place the same index in opposite splits)
                conflicting_rare_indices = trn_rare_indices.intersection(tst_rare_indices)
                trn_rare_indices = trn_rare_indices.difference(conflicting_rare_indices)
                tst_rare_indices = tst_rare_indices.difference(trn_rare_indices)

                # add conflicting indices to the test set (we want to make sure we have enough positive examples)
                tst_rare_indices = tst_rare_indices.union(conflicting_rare_indices)

                # aggregated split should be disjoint now
                assert len(trn_rare_indices.intersection(tst_rare_indices)) == 0

                # change types
                trn_rare_indices = torch.tensor(list(trn_rare_indices))
                tst_rare_indices = torch.tensor(list(tst_rare_indices))

                # track all indices split for rare classes
                rare_indices = torch.cat([trn_rare_indices, tst_rare_indices])
            else:
                trn_rare_indices = torch.tensor([], dtype=torch.int64)
                tst_rare_indices = torch.tensor([], dtype=torch.int64)
                rare_indices = torch.tensor([], dtype=torch.int64)

            # get indices not split already by rare classes
            nonrare_indices = torch.tensor([ind for ind in range(dataset_size) if ind not in rare_indices])

            assert len(nonrare_indices) + len(rare_indices) == dataset_size

            # shuffle
            nonrare_indices = nonrare_indices[torch.randperm(len(nonrare_indices))]

            # calculate remaining indices needed for the test set
            split_len_remaining = split_len - len(tst_rare_indices)
            assert split_len_remaining > 0, "Too many even rare class splits"

            trn_nonrare_indices = nonrare_indices[split_len_remaining:]
            tst_nonrare_indices = nonrare_indices[:split_len_remaining]

            # combine indices for rare and non-rare
            trn_indices = torch.cat([trn_rare_indices, trn_nonrare_indices])
            tst_indices = torch.cat([tst_rare_indices, tst_nonrare_indices])

            assert len(tst_indices) == split_len
            assert len(trn_indices) + len(tst_indices) == dataset_size

            with open(cache_fn, 'wb') as handle:
                pickle.dump((trn_indices, tst_indices), handle, protocol=4)
    else:
        raise ValueError(f"Unrecognized split type {split_type}")

    return CustomTensorDataset(*dset[trn_indices]), CustomTensorDataset(*dset[tst_indices])


def parse_split(root, split):
    def parse_pairs(pair_list):
        with open(pair_list, 'r') as f:
            pairs = f.read().strip().split('\n')
            pairs = [t.split() for t in pairs]
            pairs = list(map(tuple, pairs))
        attrs, objs = zip(*pairs)
        return attrs, objs, pairs

    tr_attrs, tr_objs, tr_pairs = parse_pairs(os.path.join(root, split, "train_pairs.txt"))
    vl_attrs, vl_objs, vl_pairs = parse_pairs(os.path.join(root, split, "val_pairs.txt"))
    ts_attrs, ts_objs, ts_pairs = parse_pairs(os.path.join(root, split, "test_pairs.txt"))

    all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
    all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
    all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

    return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs


def get_split_info(root, split, pairs):
    data = torch.load(os.path.join(root, f'metadata_{split}.t7'))
    train_data, val_data, test_data = [], [], []
    for instance in data:
        image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

        if attr == 'NA' or (attr, obj) not in pairs or settype == 'NA':
            # ignore instances with unlabeled attributes or that are not in the current split
            continue

        data_i = [image, attr, obj]
        if settype == 'train':
            train_data.append(data_i)
        elif settype == 'val':
            val_data.append(data_i)
        else:
            test_data.append(data_i)

    return train_data, val_data, test_data


def convert_split_info_to_df(split_info):
    df = pd.DataFrame.from_dict(split_info)
    df.columns = ["image", "attr", "obj"]
    return df


def split_obj(obj_name):
    if "." in obj_name:
        cat, *subcat = obj_name.split(".")
        return cat, ".".join(subcat)
    return obj_name, None  # if subcategory, set it to None


def join_obj(cat, subcat):
    if subcat is None:
        return cat
    return ".".join((cat, subcat))


def reindex_utzappos_zero_shot_df(split_data):
    split_data["image"] = split_data["image"].apply(UTZapposIDImageFolderZeroShot.get_img_id)

    split_data = split_data.set_index('image')

    return split_data


def convert_df_to_dict(df):
    # transpose to get index as column, create dict mapping {path_name: list}
    df = df.T.to_dict(orient="list")

    # transform type of values in img_ids dict
    df = {k: tuple(v) for k, v in df.items()}

    return df


def get_index_and_range_of_label(split_data, lbl_col, lbl_val):
    label_index = None if lbl_val is None else list(split_data.columns).index("_".join((lbl_col, lbl_val)))
    indices_for_category = [i for i, col in enumerate(split_data.columns) if col.startswith(lbl_col + "_")]
    label_range = (min(indices_for_category), max(indices_for_category))
    return label_index, label_range


def process_split_utzappos_zero_shot(split_data, trn_df=None):
    split_data = convert_split_info_to_df(split_data)

    split_data["cat"], split_data["subcat"] = zip(*split_data["obj"].apply(split_obj))
    split_data.drop("obj", axis=1, inplace=True)

    split_data_zs_cls = split_data.copy()

    for column in ["attr", "cat", "subcat"]:
        # if subcat is missing, sets all the binarized columns to 0
        binarized_columns = pd.get_dummies(split_data[column], prefix=column)
        split_data.drop(column, axis=1, inplace=True)

        if trn_df is not None:
            # add columns for labels that don't appear in non-train split
            for trn_df_col in trn_df.columns:
                if trn_df_col.startswith(column) and trn_df_col not in binarized_columns.columns:
                    binarized_columns[trn_df_col] = pd.Series(np.zeros(len(binarized_columns)), dtype=np.uint8)

        split_data = split_data.join(binarized_columns, how="outer")

    split_data = reindex_utzappos_zero_shot_df(split_data)

    if trn_df is not None:
        # sort the columns to be in the same order as the training set
        split_data = split_data[trn_df.columns]

    split_data_zs_cls = reindex_utzappos_zero_shot_df(split_data_zs_cls)

    for lbl_col in split_data_zs_cls.columns:
        split_data_zs_cls[lbl_col] = split_data_zs_cls[lbl_col].apply(partial(get_index_and_range_of_label,
                                                                              split_data, lbl_col))

    return split_data, split_data_zs_cls


class UTZapposIDImageFolderZeroShot(datasets.ImageFolder):
    def __init__(self, attr_fn, pkl_fn, img_size, split="trn", *args, **kwargs):

        super().__init__(*args, **kwargs)

        pkl_fn = UTZapposIDImageFolderZeroShot.get_cache_name(pkl_fn, img_size, split)

        try:
            with open(pkl_fn, 'rb') as f:
                self.img_ids, self._zs_cls, self._label_info = pickle.load(f)
        except FileNotFoundError:
            self.img_ids, self._zs_cls, self._label_info = UTZapposIDImageFolderZeroShot._read_images(attr_fn, split)
            with open(pkl_fn, 'wb') as handle:
                # use protocol 4 to be backwards compatible up to Python 3.4, but still allow large pickle files
                pickle.dump((self.img_ids, self._zs_cls, self._label_info), handle, protocol=4)

        # throw out the class
        _samples = [path for path, _cls in self.samples]

        # the basename of the file corresponds to its ID
        # remove duplicate IDs by constructing dict with IDs as keys
        # use an OrderedDict for reproducibility between runs, assuming self.samples is always the same order
        _samples = OrderedDict(zip(map(UTZapposIDImageFolderZeroShot.get_img_id, _samples), _samples))

        # remove any samples which were filtered out when constructing img_ids (e.g. by picking trn, val, tst split)
        # use the full path and throw out the ID for now
        self.samples = [v for k, v in _samples.items() if k in self.img_ids]

    @staticmethod
    def _update_cache_name(pkl_fn, name):
        """
        Add <name> to <pkl_fn>.
        """
        pkl_fn, _ = pkl_fn.split(".pickle")
        return f"{pkl_fn}_{name}.pickle"

    @staticmethod
    def get_cache_name(pkl_fn, img_size, split):
        assert pkl_fn.endswith(".pickle")

        pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, f"img{img_size}")
        pkl_fn = UTZapposIDImageFolder._update_cache_name(pkl_fn, f"split_{split}")

        return pkl_fn

    @staticmethod
    def _get_unique_multicol(img_ids, col):
        return order_set(list(chain.from_iterable(map(lambda x: x.split(";"), filter(lambda x: isinstance(x, str),
                                                                                     img_ids[col].values)))))

    @staticmethod
    def _get_option_col(img_ids, col_val, col_name):
        """
        Return a column with binarized version of img_ids[col_name] for attribute value col_val.
        """
        def _binarize_attribute(val):
            if isinstance(val, str):
                return 1 if col_val in val else 0
            return -1  # if it's not str, then it's missing

        return img_ids[col_name].apply(_binarize_attribute)

    @staticmethod
    def _read_images(attr_fn, split):
        """
        Get a dictionary mapping paths to attributes.
        """
        subfolder = "compositional-split-natural"
        attrs, objs, pairs, trn_pairs, val_pairs, tst_pairs = parse_split(attr_fn, subfolder)
        trn, val, tst = get_split_info(attr_fn, subfolder, pairs)

        # always need to process this to get all the label names
        trn_df, trn_zs_cls_df = process_split_utzappos_zero_shot(trn)

        if split == "trn":
            df, zs_cls_df = trn_df, trn_zs_cls_df
        elif split == "val":
            df, zs_cls_df = process_split_utzappos_zero_shot(val, trn_df)
        elif split == "tst":
            df, zs_cls_df = process_split_utzappos_zero_shot(tst, trn_df)
        else:
            raise ValueError

        def get_class_freq(col_name):
            if col_name not in df.columns:
                return 0
            # get the percentage of positive labels for a given column
            return df[col_name].values.mean() * 100

        class_freqs = list(map(get_class_freq, trn_df.columns))
        _label_info = dict(zip(df.columns, enumerate(class_freqs)))

        # transpose to get index as column, create dict mapping {path_name: list}
        df = convert_df_to_dict(df)
        zs_cls_df = convert_df_to_dict(zs_cls_df)

        return df, zs_cls_df, _label_info

    @staticmethod
    def get_img_id(path):
        # define the ID of an image
        return os.path.basename(path)

    @staticmethod
    def _flatten_zs_cls(zs_cls):
        for cat in zs_cls:
            cat_index, (lbl_min, lbl_max) = cat
            if cat_index is None:
                cat_index = np.nan
            for el in (cat_index, lbl_min, lbl_max):
                yield el

    @staticmethod
    def flatten_zs_cls_convert_to_tensor(zs_cls):
        flat_zs_cls = list(UTZapposIDImageFolderZeroShot._flatten_zs_cls(zs_cls))
        return torch.tensor(flat_zs_cls, dtype=torch.float32)

    @staticmethod
    def structure_zs_cls_lst(zs_cls_lst):
        for i in range(len(zs_cls_lst) // 3):
            cat_index, lbl_min, lbl_max = zs_cls_lst[3 * i: 3 * (i + 1)]
            yield cat_index, (lbl_min, lbl_max)

    @staticmethod
    def tensor_zs_cls_to_tuple(zs_cls_tensor):
        zs_cls_lst = list(zs_cls_tensor.numpy())

        if len(zs_cls_lst) % 3 != 0:
            raise ValueError(f"Unrecognized format for zero-shot class, got len {len(zs_cls_lst)}")

        return tuple(UTZapposIDImageFolderZeroShot.structure_zs_cls_lst(zs_cls_lst))

    def __getitem__(self, index):
        """
        Given an index for the dataset, return the element at that index
        """
        # load in image path and target
        path = self.samples[index]

        # load the image sample
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        target = torch.tensor(self.img_ids[UTZapposIDImageFolder.get_img_id(path)], dtype=torch.int64)

        zs_cls = self._zs_cls[UTZapposIDImageFolder.get_img_id(path)]
        zs_cls_tensor = UTZapposIDImageFolderZeroShot.flatten_zs_cls_convert_to_tensor(zs_cls)

        return sample, target, zs_cls_tensor


def utzappos_zero_shot_tensor_dset(img_size, cache_fn, split, transform=None, *dset_args, **dset_kwargs):
    """
    Convert folder dataset to tensor dataset.
    """
    cache_fn = UTZapposIDImageFolderZeroShot.get_cache_name(cache_fn, img_size, split)
    try:
        with open(cache_fn, 'rb') as f:
            dset_samples, dset_labels, dset_zs_cls, dset_label_info = pickle.load(f)
    except FileNotFoundError:
        img_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                        torchvision.transforms.ToTensor()])
        dset = UTZapposIDImageFolderZeroShot(*dset_args, img_size=img_size, transform=img_transform, split=split,
                                             **dset_kwargs)
        dset_examples = [dset[ind] for ind in range(len(dset))]
        dset_samples, dset_labels, dset_zs_cls = map(torch.stack, zip(*dset_examples))
        dset_label_info = dset._label_info
        with open(cache_fn, 'wb') as handle:
            pickle.dump((dset_samples, dset_labels, dset_zs_cls, dset_label_info), handle, protocol=4)

    return CustomTensorDataset(dset_samples, dset_labels, dset_zs_cls, zs=True, transform=transform), \
           dset_label_info, cache_fn


def _get_all_and_unique_labels(labels):
    all_labels = list(map(tuple, map(lambda lbl: lbl.numpy(), labels)))
    return all_labels, set(all_labels)


def _get_label_frequencies(trn_labels, tst_labels):
    counts = defaultdict(lambda: {"trn": 0, "tst": 0})

    for label in trn_labels:
        counts[label]["trn"] += 1

    for label in tst_labels:
        counts[label]["tst"] += 1

    return counts


def _print_label_frequencies(freqs, filter_freq=100):
    # sort by frequency
    for k, v in sorted(freqs.items(), key=lambda item: item[1]["trn"], reverse=True):
        if v["trn"] < filter_freq:
            break
        print(k, v["trn"], v["tst"])


def filter_to_salient_labels(labels, dset_label_info):
    # salient_labels = ["Male", "No_Beard", "Smiling", "Wearing_Lipstick", "Arched_Eyebrows", "High_Cheekbones",
    #                   "Bangs", "Young", "Wavy_Hair", "Black_Hair", "Bags_Under_Eyes", "Mouth_Slightly_Open",
    #                   "Attractive"]

    more_salient_labels = ["Male", "No_Beard", "Smiling", "Bangs", "Young", "Wavy_Hair", "Black_Hair",
                           "Mouth_Slightly_Open", "Attractive"]

    salient_label_inds = [dset_label_info[salient_label][0] for salient_label in more_salient_labels]

    return labels[:, salient_label_inds]


def index_by_combo_name(shaped_logits, dset_label_info, combo, include_marginals):
    (att1, val1), (att2, val2) = combo.items()
    att1_ind = dset_label_info[att1][0]
    att2_ind = dset_label_info[att2][0]
    pos_prob = shaped_logits[:, att1_ind, val1] + shaped_logits[:, att2_ind, val2]
    if include_marginals:
        logit_logsumexp = shaped_logits.logsumexp(-1)
        lsp_indices = torch.arange(logit_logsumexp.shape[-1])
        lsp_mask = torch.logical_and(lsp_indices != att1_ind, lsp_indices != att2_ind)
        lsp = logit_logsumexp[:, lsp_mask].sum(-1)
        pos_prob = pos_prob + lsp
    return pos_prob


def _filter_to_combo(all_labels, dset_label_info, combo):
    (att1, val1), (att2, val2) = combo.items()
    att1_ind = dset_label_info[att1][0]
    att2_ind = dset_label_info[att2][0]
    return torch.logical_and(all_labels[:, att1_ind] == val1, all_labels[:, att2_ind] == val2)


def _get_combo_inds(*args, **kwargs):
    # noinspection PyArgumentList
    return list(_filter_to_combo(*args, **kwargs).nonzero(as_tuple=False).squeeze().numpy())


def _get_combo_freq(*args, **kwargs):
    return _filter_to_combo(*args, **kwargs).sum().item()


def celeba_stats():
    root = os.path.join("../data", "celeba")  # TODO: change to data/
    dset, dset_label_info, cache_fn = celeba_tensor_dset(root=root,
                                                         img_size=32,  # TODO: change to 64
                                                         attr_fn=os.path.join(root, "list_attr_celeba.txt"),
                                                         pkl_fn=os.path.join(root, "cache.pickle"),
                                                         cache_fn=os.path.join(root, "dset.pickle"),
                                                         drop_infreq=13)

    trn_dset, tst_dset = split_celeba(dset, root=root, cache_fn=cache_fn, split_len=5000,
                                      split_type="balanced", balanced_split_ratio=0.5)

    all_labels = dset[:][1]
    trn_labels, tst_labels = trn_dset[:][1], tst_dset[:][1]

    trn_labels = filter_to_salient_labels(trn_labels, dset_label_info)
    tst_labels = filter_to_salient_labels(tst_labels, dset_label_info)

    print(f"Number of attributes: {trn_labels.shape[1]}")

    # get number of unique label combinations in each
    trn_labels, unique_trn_labels = _get_all_and_unique_labels(trn_labels)
    tst_labels, unique_tst_labels = _get_all_and_unique_labels(tst_labels)

    print(f"Number of unique attribute combinations: {len(unique_trn_labels)} {(len(unique_tst_labels))}")

    print(f"Number of unique attribute combinations only in train: "
          f"{len(unique_trn_labels.difference(unique_tst_labels))}")
    print(f"Number of unique attribute combinations only in test: "
          f"{len(unique_tst_labels.difference(unique_trn_labels))}")

    label_freqs = _get_label_frequencies(trn_labels, tst_labels)

    print("============ TRN LABEL FREQS ============")
    _print_label_frequencies(label_freqs)

    # get frequencies of all these attribute combinations
    combo_freqs = sorted(zip(map(partial(_get_combo_freq, all_labels, dset_label_info), CELEBA_ZERO_SHOT_COMBOS),
                             CELEBA_ZERO_SHOT_COMBOS), reverse=True)

    for combo_freq, combo in combo_freqs:
        print(combo_freq, combo)


if __name__ == "__main__":
    # just so we can run it on the cluster
    parser = argparse.ArgumentParser("Zero Shot Learning yeet")
    parser.add_argument("--save_dir", type=str, default="tmp")
    parser.add_argument("--ckpt_path", type=str, default="tmp")

    # data_root = "../data/utzappos/ut-zap50k-images-square"
    # dset_ = utzappos_tensor_dset(root=data_root,
    #                              img_size=64,
    #                              attr_fn=os.path.join(data_root, "meta-data.csv"),
    #                              pkl_fn=os.path.join(data_root, "cache.pickle"),
    #                              cache_fn=os.path.join(data_root, "dset.pickle"),
    #                              observed=True, binarized=True, drop_infreq=10)
    #
    # # check that indexing works
    # print(dset_[0][0])

    # data_root = "../data/celeba"
    # dset_ = celeba_tensor_dset(root=data_root,
    #                            img_size=32,
    #                            attr_fn=os.path.join(data_root, "list_attr_celeba.txt"),
    #                            pkl_fn=os.path.join(data_root, "cache.pickle"),
    #                            cache_fn=os.path.join(data_root, "dset.pickle"),
    #                            drop_infreq=10)
    #
    # print(dset_[0][0])

    # data_root = "../data/CUB"
    # dset_ = cub_tensor_dset(root=data_root,
    #                         img_size=128,
    #                         attr_fn=os.path.join(data_root, "CUB_200_2011", "attributes",
    #                                              "image_attribute_labels.txt"),
    #                         attr_name_fn=os.path.join(data_root, "attributes.txt"),
    #                         img_id_fn=os.path.join(data_root, "CUB_200_2011", "images.txt"),
    #                         bb_fn=os.path.join(data_root, "CUB_200_2011", "bounding_boxes.txt"),
    #                         pkl_fn=os.path.join(data_root, "cache.pickle"),
    #                         cache_fn=os.path.join(data_root, "dset.pickle"),
    #                         drop_infreq=10)
    #
    # print(dset_[0][0])

    # data_root = "../data/utzappos/zero-shot"
    # dset_ = UTZapposIDImageFolderZeroShot(root=os.path.join(data_root, "images"),
    #                                       img_size=32,
    #                                       attr_fn=data_root,
    #                                       pkl_fn=os.path.join(data_root, "cache.pickle"),
    #                                       split="tst")
    # print(len(dset_))
    # print(dset_[0])
    #
    # zs_cst = partial(UTZapposIDImageFolderZeroShot,
    #                  root=os.path.join(data_root, "images"),
    #                  img_size=32,
    #                  attr_fn=data_root,
    #                  pkl_fn=os.path.join(data_root, "cache.pickle"))
    # trn_dset_ = zs_cst(split="trn")
    # val_dset_ = zs_cst(split="val")
    # tst_dset_ = zs_cst(split="tst")
    #
    # assert len(trn_dset_.samples) == len(set(trn_dset_.samples))
    # assert len(val_dset_.samples) == len(set(val_dset_.samples))
    # assert len(tst_dset_.samples) == len(set(tst_dset_.samples))
    #
    # assert set(trn_dset_.samples).isdisjoint(set(val_dset_.samples))
    # assert set(tst_dset_.samples).isdisjoint(set(val_dset_.samples))
    # assert set(trn_dset_.samples).isdisjoint(set(tst_dset_.samples))

    # data_root = "../data/utzappos/zero-shot"
    # dset_ = utzappos_zero_shot_tensor_dset(root=os.path.join(data_root, "images"),
    #                                        img_size=64,
    #                                        attr_fn=data_root,
    #                                        pkl_fn=os.path.join(data_root, "cache.pickle"),
    #                                        cache_fn=os.path.join(data_root, "dset.pickle"),
    #                                        split="tst")
    # print(dset_[0][0])

    celeba_stats()
