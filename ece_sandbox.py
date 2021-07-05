"""
Sandbox for ECE score.
"""

import numpy as np
import torch

from train_joint import get_calibration


def old_get_calibration(y, p_mean, num_bins=10):
    """Compute the calibration.
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
      y: one-hot encoding of the true classes, size (?, num_classes)
      p_mean: numpy array, size (batch_size, num_classes)
            containing the mean output predicted probabilities
      num_bins: number of bins
    Returns:
      cal: a dictionary
        {reliability_diag: realibility diagram
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins (11,)
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])  # (128,)
        nb_items_bin[i] = np.sum(sec)
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf_update = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        mean_conf[i] = mean_conf_update
        # compute the empirical confidence
        acc_tab_update = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
        acc_tab[i] = acc_tab_update

    # Reliability diagram (this is moved up so clean up does not change the dimension)
    reliability_diag = (mean_conf, acc_tab)

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    final_nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Expected Calibration Error
    _weights = 0.
    ece_ = np.sum(_weights * np.absolute(mean_conf - acc_tab))
    if not np.sum(final_nb_items_bin) == 0:
        _weights = final_nb_items_bin.astype(np.float32) / np.sum(final_nb_items_bin)
        ece_ = np.average(
            np.absolute(mean_conf - acc_tab),
            weights=_weights)
    # Maximum Calibration Error
    # mce = np.max(np.absolute(mean_conf - acc_tab)) # this gives np.max(empty) errors
    cal = {
        'reliability_diag': reliability_diag,
        'ece': ece_,
        'nb_items': nb_items_bin/np.sum(nb_items_bin)
    }
    return cal


rng = np.random.RandomState(0)

# basic correctness test
bs = 1000
true = rng.randint(low=0, high=2, size=(bs,))
pred = rng.rand(bs, 2)

ece_old = old_get_calibration(y=true, p_mean=pred)["ece"]
ece_vec = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred), multi_label=False)["ece"].numpy()

print(f"{ece_old:.4e} {ece_vec:.4e}")
assert np.allclose(ece_old, ece_vec)

# a test with empty bins
bs = 1000
true = rng.randint(low=0, high=2, size=(bs,))
pred = rng.rand(bs, 2) * .8 + .1

ece_old = old_get_calibration(y=true, p_mean=pred)["ece"]
ece_vec = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred), multi_label=False)["ece"].numpy()

print(f"{ece_old:.4e} {ece_vec:.4e}")
assert np.allclose(ece_old, ece_vec)

# a test with multiple attributes
bs = 1000
num_labels = 51
true = rng.randint(low=0, high=2, size=(bs, 51))
pred = rng.rand(bs, 51, 2)

ece_old_ind = np.array([old_get_calibration(y=true[:, i], p_mean=pred[:, i])["ece"] for i in range(num_labels)])
ece_vec_ind = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred), individual=True)["ece"].numpy()
ece_old = np.mean(ece_old_ind)
ece_vec = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred))["ece"].numpy()

print(f"{ece_old:.4e} {ece_vec:.4e}")
assert np.allclose(ece_old_ind, ece_vec_ind)
assert np.allclose(ece_old, ece_vec)

# a test with multiple attributes and missing bins
bs = 1000
num_labels = 51
true = rng.randint(low=0, high=2, size=(bs, 51))
pred = rng.rand(bs, 51, 2) * .8 + .1

ece_old_ind = np.array([old_get_calibration(y=true[:, i], p_mean=pred[:, i])["ece"] for i in range(num_labels)])
ece_vec_ind = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred), individual=True)["ece"].numpy()
ece_old = np.mean(ece_old_ind)
ece_vec = get_calibration(y=torch.tensor(true).long(), p_mean=torch.tensor(pred))["ece"].numpy()

print(f"{ece_old:.4e} {ece_vec:.4e}")
assert np.allclose(ece_old_ind, ece_vec_ind)
assert np.allclose(ece_old, ece_vec)
