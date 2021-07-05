"""
Can't figure out how to use the sklearn f1 score properly lol, so going to try some examples.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score

from train_joint import f1_score_missing

pred = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0]])
true = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]])

sk_f1 = f1_score(y_pred=pred, y_true=true, average="macro")

assert np.isclose(f1_score_missing(y_true=true, y_pred=pred), sk_f1)

rng = np.random.RandomState(0)
pred = rng.randint(low=0, high=2, size=(100, 111))
true = rng.randint(low=0, high=2, size=(100, 111))

sk_f1 = f1_score(y_pred=pred, y_true=true, average="macro")

assert np.isclose(f1_score_missing(y_true=true, y_pred=pred), sk_f1)

rng = np.random.RandomState(0)
pred = rng.randint(low=0, high=2, size=(100, 18))
true = rng.randint(low=-1, high=2, size=(100, 18))  # include missing values

sk_f1 = []
for ind in range(pred.shape[1]):
    pred_, true_ = pred[:, ind], true[:, ind]
    missing_mask = true_ != -1
    pred_, true_ = pred_[missing_mask], true_[missing_mask]
    sk_f1.append(f1_score(y_pred=pred_, y_true=true_))

assert np.all(np.isclose(f1_score_missing(y_true=true, y_pred=pred, individual=True), sk_f1))
assert np.isclose(f1_score_missing(y_true=true, y_pred=pred), np.mean(sk_f1))

rng = np.random.RandomState(0)
pred = rng.randint(low=0, high=2, size=(100, 111))
true = rng.randint(low=0, high=2, size=(100, 111))

sk_f1 = f1_score(y_pred=pred, y_true=true, average="micro")

assert np.isclose(f1_score_missing(y_true=torch.tensor(true).float(),
                                   y_pred=torch.tensor(pred).float(), micro=True), sk_f1)
