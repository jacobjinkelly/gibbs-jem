"""
UTZappos processing.
"""

import os

import pandas as pd
from collections import Counter

import torch


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


def split_obj(obj_name):
    if "." in obj_name:
        cat, *subcat = obj_name.split(".")
        return cat, ".".join(subcat)
    return obj_name, None


def main():
    root = "/Users/jacobkelly/Downloads/compositional_split_natural/ut-zap50k"
    split = "compositional-split-natural"

    _, _, pairs, *_ = parse_split(root, split)
    trn, val, tst = get_split_info(root, split, pairs)

    trn_df = pd.DataFrame.from_dict(trn)
    trn_df.columns = ["image", "attr", "obj"]
    val_df = pd.DataFrame.from_dict(val)
    val_df.columns = ["image", "attr", "obj"]
    tst_df = pd.DataFrame.from_dict(tst)
    tst_df.columns = ["image", "attr", "obj"]

    trn_attr_all = trn_df["attr"]

    trn_attr = set(trn_attr_all)
    val_attr = set(val_df["attr"])
    tst_attr = set(tst_df["attr"])

    assert val_attr.issubset(trn_attr)
    assert tst_attr.issubset(trn_attr)
    assert not val_attr.issubset(tst_attr)
    assert not val_attr.issuperset(tst_attr)

    trn_obj = set(trn_df["obj"])
    val_obj = set(val_df["obj"])
    tst_obj = set(tst_df["obj"])

    assert trn_obj == val_obj == tst_obj

    trn_cat_all, trn_subcat_all = zip(*trn_df["obj"].apply(split_obj))
    trn_cat, trn_subcat = set(trn_cat_all), set(trn_subcat_all)

    print(f"Num Attr: {len(trn_attr)} {len(trn_cat)} {len(trn_subcat)} | "
          f"{sum(map(len, (trn_attr, trn_cat, trn_subcat)))}")

    trn_attr_freq = Counter(trn_attr_all)
    trn_cat_freq = Counter(trn_cat_all)
    trn_subcat_freq = Counter(trn_subcat_all)

    print("==== ATTR ====")
    for attr, freq in trn_attr_freq.items():
        print(f"{attr} {100 * freq / len(trn_attr_all):.2f} ({freq})")

    print("==== CAT ====")
    for cat, freq in trn_cat_freq.items():
        print(f"{cat} {100 * freq / len(trn_cat_all):.2f} ({freq})")

    print("==== SUBCAT ====")
    for subcat, freq in trn_subcat_freq.items():
        print(f"{subcat} {100 * freq / len(trn_subcat_all):.2f} ({freq})")

    assert len(trn_attr_all) == len(trn_cat_all) == len(trn_subcat_all)
    print(f"Number of examples: {len(trn_attr_all)}")


if __name__ == "__main__":
    main()
