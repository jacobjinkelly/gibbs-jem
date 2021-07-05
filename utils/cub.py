"""
Code for processing the CUB dataset.
"""

import os
import pickle
from operator import itemgetter

import numpy as np
import torch

import utils
import utils.process as im_folder


def get_attributes(root):
    """
    Get attributes.
    """
    attributes_filename = os.path.join(root, "CUB_200_2011", "CUB_200_2011", "attributes",
                                       "class_attribute_labels_continuous.txt")
    with open(attributes_filename) as f:
        attributes = np.array(list(map(lambda line: list(map(float, line.split(" "))), f.readlines())))

    classes_filename = os.path.join(root, "CUB_200_2011", "CUB_200_2011", "classes.txt")
    with open(classes_filename) as f:
        classes = list(map(lambda line: line.split(" ")[-1], f.read().splitlines()))

    return dict(zip(classes, attributes))


def get_cub_dataset(root, split, transform):
    """
    Returns a dataset object with (image, label) pairs and an
    array mapping classes to attributes.
    """
    cub_root = os.path.join(root, "CUB_200_2011", "CUB_200_2011")

    att_fn = os.path.join(cub_root, "attributes", "image_attribute_labels.txt")
    img_fn = os.path.join(cub_root, "images.txt")
    pkl_fn = os.path.join(cub_root, "cub.pkl")
    bb_fn = os.path.join(cub_root, "bounding_boxes.txt")
    if not os.path.isfile(pkl_fn):
        dset = im_folder.IDImageFolder(root=os.path.join(root, get_folder(split)),
                                       transform=transform,
                                       attributes_file=att_fn,
                                       images_file=img_fn,
                                       bounding_boxes_file=bb_fn)
        with open(pkl_fn, 'wb') as f:
            pickle.dump((dset.img_ids, dset.im_attributes, dset.im_certainties, dset.bounding_boxes), f)
    else:
        dset = im_folder.IDImageFolder(root=os.path.join(root, get_folder(split)), transform=transform, pkl_file=pkl_fn)

    attributes_dict = get_attributes(root)

    # sort this by index then get rid of the indexes since they're redundant
    _, attributes = zip(*sorted(((dset.class_to_idx[class_name], attributes_dict[class_name])
                                 for class_name in dset.class_to_idx),
                                key=itemgetter(0)))

    return dset, torch.tensor(np.array(attributes) / 100)


def get_folder(split, fold=1):
    """
    Get the path of the folder for a particular split relative to the root data.
    <split>: trn, val, or tst
    <fold>: 1, 2, 3
    """
    split_names = {
        "trn": "train",
        "val": "val",
        "tst": "test"
    }
    assert split in split_names, f"split must be a valid split, got {split}"
    assert fold in (1, 2, 3), f"fold must be a valid fold, got {fold}"

    def _get_fold_name():
        if split == "tst":
            return ""
        else:
            return fold

    return os.path.join("CUB_200_2011", f"{split_names[split]}classes{_get_fold_name()}", "images")


def get_class_splits(root):
    """
    Get the paths of all the class splits.
    """
    split_files = ["testclasses.txt"] + \
                  [f"{split}classes{fold}.txt" for fold in range(1, 4) for split in ("train", "val")]
    class_splits = {}

    for split_file in split_files:
        with open(os.path.join(root, "CUB_200_2011", "CUB_200_2011", split_file)) as f:
            class_splits[split_file] = f.read().splitlines()

    return class_splits


def make_aliases(root, class_splits):
    """
    Make aliased folders for the different splits.
    """
    for class_split in class_splits:
        class_split_name = os.path.splitext(class_split)[0]

        class_split_path = os.path.join(root, "CUB_200_2011", class_split_name)
        img_dir = os.path.join(class_split_path, "images")

        utils.makedirs(class_split_path)
        utils.makedirs(img_dir)

        for class_name in class_splits[class_split]:
            src_path = os.path.abspath(os.path.join(root, "CUB_200_2011", "CUB_200_2011", "images", class_name))
            dst_path = os.path.abspath(os.path.join(class_split_path, "images", class_name))
            os.symlink(src_path, dst_path, target_is_directory=True)


def process_data(root):
    """
    Process the data into splits.
    """
    class_splits = get_class_splits(root=root)
    make_aliases(root=root, class_splits=class_splits)


if __name__ == "__main__":
    process_data(root="../data")
