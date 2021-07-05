"""
Plot ECE diagrams.
"""

import argparse
import pickle

import torch

from compare_pr import count_caches_in_dir, get_plot_itr
from train_joint import _plot_ece_hist, get_calibration, average_precision_score, _plot_pr_curve
from sklearn.metrics import precision_recall_curve

PREFIX = "cache_ece_"


def shape_logits(raw_logits):
    """
    Shape logits for logsumexp.
    """
    return raw_logits.view(raw_logits.shape[0], -1, 2)


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


def ece_plot_b(save_dir, dset_label_info, b, y, plot_itr, individual=False, micro=False):
    # convert raw unnormalized logits to probabilities
    p_mean = logit_log_prob(b, logit_logsumexp(b)).exp().detach()

    ece_info = get_calibration(y=y, p_mean=p_mean, individual=individual, micro=micro, debug=True)
    if micro:
        _plot_ece_hist(save_dir, f"micro_{plot_itr}", ece_info["reliability_diag"], ece_info["ece"],
                       y.mean(), dset_label_info)
    else:
        for label_dim_ in range(y.shape[1]):
            _plot_ece_hist(save_dir, f"{label_dim_}_{plot_itr}",
                           tuple(el[:, label_dim_] for el in ece_info["reliability_diag"]),
                           ece_info["ece"][label_dim_], y[:, label_dim_].mean(), dset_label_info)
            break  # only do first attribute

    return ece_info


def pr_curve_b(save_dir, dset_label_info, b, y, plot_itr, micro=False):
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
        _plot_pr_curve(save_dir, f"micro_{plot_itr}", precision, recall, ap, y.mean(), dset_label_info)
    else:
        y = y.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        for label_dim_ in range(y.shape[1]):
            precision, recall, thres = precision_recall_curve(y_true=y[:, label_dim_], probas_pred=b[:, label_dim_])
            ap = average_precision_score(y_true=y[:, label_dim_],
                                         y_score=b[:, label_dim_])
            _plot_pr_curve(save_dir, f"{label_dim_}_{plot_itr}",
                           precision, recall, ap, y[:, label_dim_].mean(), dset_label_info)
            break  # only do first attr

    return precision, recall, thres


def main():
    parser = argparse.ArgumentParser("Zero Shot Learning yeet")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--direct", action="store_true", default=False)

    args = parser.parse_args()

    plot_itr = get_plot_itr(args.dir, prefix=PREFIX)

    if args.direct:
        with open(f"{args.dir}/full_tst_preds.pickle", "rb") as f:
            d = pickle.load(f)
        individual_y_pred_score = torch.tensor(d["score"])
        individual_y_true = torch.tensor(d["true"])

        ece_info = ece_plot_b(args.dir, None, individual_y_pred_score, individual_y_true, plot_itr, individual=True)
        conf, sec, tau_tab, acc_tab = ece_info["conf"], ece_info["sec"], ece_info["tau_tab"], ece_info["acc_tab"]
        precision, recall, thres = pr_curve_b(args.dir, None, individual_y_pred_score, individual_y_true, plot_itr)
        print(acc_tab[tau_tab[1:].squeeze() == .65][:, 0])  # acc at 65% conf
        i = -2
        print(precision[i], recall[i], thres[i - 1])
    else:
        for label_dim in range(count_caches_in_dir(args.dir, prefix=PREFIX)):
            _plot_ece_hist(args.dir, f"{label_dim}_{plot_itr}", from_cache=True)

        _plot_ece_hist(args.dir, f"micro_{plot_itr}", from_cache=True)


if __name__ == "__main__":
    main()
