"""
Defines metrics.
"""

import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _harmonic_acc(trn_acc, tst_acc):
    return 2 * trn_acc * tst_acc / (trn_acc + tst_acc)


def initialize_metrics():
    """
    Initialize dictionary for metrics.
    """
    return defaultdict(list)


def update_metrics(metrics, new_metrics):
    """
    Update all metrics with evaluations this epoch.
    """
    for metric_name in new_metrics:
        metrics[metric_name].append(new_metrics[metric_name])

    # update metrics which are a function of other metrics
    harmonic_metrics = {"trn_acc", "val_acc"}
    if harmonic_metrics.issubset(new_metrics):
        metrics["harmonic_acc"].append(_harmonic_acc(metrics["trn_acc"][-1], metrics["val_acc"][-1]))


def print_metrics(metrics, logger):
    """
    Print metrics.
    """
    print_str = ""
    separator = " | "
    for metric_name in metrics:
        if metric_name == "eval_itr":
            print_str += f"{metric_name}: {metrics[metric_name][-1]:07}" + separator
        else:
            print_str += f"{metric_name}: {metrics[metric_name][-1]:.2f}" + separator

    print_str = print_str[:-len(separator)]  # remove final separator

    logger(print_str)


def plot_metrics(metrics, log_dir):
    """
    Plot the metrics.
    """
    # plot metrics individually
    for metric_name in metrics:
        if metric_name == "eval_itr":
            continue

        plt.clf()
        plt.plot(metrics["eval_itr"], metrics[metric_name])
        plt.savefig(os.path.join(log_dir, f"{metric_name}.png"))

    # compare metrics on the same plot
    comparison_metrics = {"trn_acc", "val_acc"}
    if comparison_metrics.issubset(metrics):
        plt.clf()
        for acc_name in comparison_metrics:
            plt.plot(metrics["eval_itr"], metrics[acc_name])
        plt.savefig(os.path.join(log_dir, f"acc.png"))


def log_metrics(metrics, logger, log_dir):
    """
    Print and plot metrics.
    """
    print_metrics(metrics, logger)
    plot_metrics(metrics, log_dir)
