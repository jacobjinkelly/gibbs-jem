"""
Plot PR curves for different models alongside each other.
"""

import argparse
import os
import pickle
from functools import partial

import matplotlib.pyplot as plt

PREFIX = "cache_pr_"
SUFFIX = ".pickle"


def load_args(save_dir, fn):
    with open(f"{save_dir}/{PREFIX}{fn}{SUFFIX}", "rb") as f:
        d = pickle.load(f)
    return d


def count_caches_in_dir(save_dir, prefix=PREFIX, suffix=SUFFIX):
    return len([f for f in os.listdir(save_dir) if f.endswith(suffix) and f.startswith(prefix) and "micro" not in f])


def _parse_itr(prefix, suffix, f):
    if f.endswith(suffix) and f.startswith(prefix):
        _, f = f[len(prefix):-len(suffix)].split("_")
        return int(f)
    else:
        return None


def get_plot_itr(save_dir, prefix=PREFIX, suffix=SUFFIX):

    plot_itr, = set(filter(None, map(partial(_parse_itr, prefix, suffix), os.listdir(save_dir))))
    return plot_itr


def plot_pr_curve(dirs, model_names, fn_prefix):
    plt.clf()

    font = {'family': 'normal',
            'size': 20}
    plt.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    for exp_dir, model_name in zip(dirs, model_names):
        d = load_args(exp_dir, f"{fn_prefix}_{get_plot_itr(exp_dir)}")
        precision = d["precision"]
        recall = d["recall"]
        c = "#1f77b4" if model_name == "Supervised" else "#ff7f0e"
        plt.plot(recall, precision, label=f"{model_name}", color=c)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xticks([.0, .5, 1.])
    plt.yticks([.0, .5, 1.])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc=3)
    plt.gcf().subplots_adjust(right=.95, left=.14, bottom=.15)
    for exp_dir in dirs:
        plt.savefig(f"{exp_dir}/compare_pr_{fn_prefix}.png")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Zero Shot Learning yeet")
    parser.add_argument("--dirs", nargs="+", type=str, required=True)
    parser.add_argument("--model_names", nargs="+", type=str, required=True)

    args = parser.parse_args()

    assert len(args.dirs) == len(args.model_names)
    
    assert len(set(map(count_caches_in_dir, args.dirs))) == 1

    for label_dim in range(count_caches_in_dir(args.dirs[0])):
        plot_pr_curve(args.dirs, args.model_names, label_dim)

    plot_pr_curve(args.dirs, args.model_names, "micro")


if __name__ == "__main__":
    main()
