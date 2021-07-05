"""
Testing implementation of AP
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train_joint import ap_score

pred = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0]])
true = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]])

sk_f1 = average_precision_score(y_score=pred, y_true=true, average="macro")

assert np.isclose(ap_score(y_true=torch.tensor(true), y_pred=torch.tensor(pred)), sk_f1)

rng = np.random.RandomState(0)
pred = rng.randint(low=0, high=2, size=(100, 111))
true = rng.randint(low=0, high=2, size=(100, 111))

sk_f1 = average_precision_score(y_score=pred, y_true=true, average="macro")

assert np.isclose(ap_score(y_true=torch.tensor(true), y_pred=torch.tensor(pred)), sk_f1)

rng = np.random.RandomState(0)
pred = rng.randint(low=0, high=2, size=(100, 111))
true = rng.randint(low=0, high=2, size=(100, 111))

sk_f1 = average_precision_score(y_score=pred, y_true=true, average="micro")

assert np.isclose(ap_score(y_true=torch.tensor(true).float(),
                           y_pred=torch.tensor(pred).float(), micro=True), sk_f1)
