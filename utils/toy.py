"""
Toy data utilities.
"""
import numpy as np
import sklearn
import sklearn.datasets
import torch
from torch.utils.data import TensorDataset

TOY_DSETS = ["moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll"]
CTS_TOY_DSETS = ["cts_" + dset for dset in TOY_DSETS]
LBL_TOY_DSETS = ["lbl_" + dset for dset in TOY_DSETS]
LBL_CTS_TOY_DSETS = ["lbl_cts_" + dset for dset in TOY_DSETS]
ALL_TOY_DSETS = TOY_DSETS + CTS_TOY_DSETS + LBL_TOY_DSETS + LBL_CTS_TOY_DSETS


# noinspection PyArgumentList
def toy_data(dataset_name, batch_size):
    """
    Get toy data samples.
    """
    rng = np.random.RandomState()
    if dataset_name == "swissroll":
        x, y = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)
        x = x.astype("float32")[:, [0, 2]]
        x /= 5
        return x, y
    elif dataset_name == "circles":
        x, y = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)
        x = x.astype("float32")
        x *= 3
        return x, y
    elif dataset_name == "rings":
        obs = batch_size
        batch_size *= 20
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2
        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        Y = np.array([0] * n_samples1 + [1] * n_samples2 + [2] * n_samples3 + [3] * n_samples4)

        # Add noise
        X += rng.normal(scale=0.08, size=X.shape)

        inds = np.random.choice(list(range(batch_size)), obs)
        X = X[inds]
        Y = Y[inds]
        return X.astype("float32"), Y
    elif dataset_name == "moons":
        x, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        x = x.astype("float32")
        x = x * 2 + np.array([-1, -0.2])
        return x, y
    elif dataset_name == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, None
    elif dataset_name == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)
        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)), None
    elif dataset_name == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x, None
    elif dataset_name == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    elif dataset_name == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1), None
    elif dataset_name == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1), None
    else:
        raise ValueError


class Int2Gray:
    def __init__(self, nbits=16, nint=10**4, min_val=0., max_val=1., device=torch.device("cpu")):
        self.int2gray, self.gray2int = torch.zeros(1 << nbits, device=device), torch.zeros(1 << nbits, device=device)
        for i in range(0, 1 << nbits):
            gray = i ^ (i >> 1)
            self.int2gray[i] = gray
            self.gray2int[gray] = i

        self.mask = torch.tensor(2) ** torch.arange(nbits)

        self.min = min_val
        self.max = max_val
        self.nint = nint
        self.nbits = nbits

        self.device = device

    def int_to_binary(self, g):
        g = g.int()
        mask = self.mask.to(g.device, g.dtype)
        return g.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def binary_to_int(self, g):
        mask = self.mask.to(g.device, g.dtype)
        return (g * mask[None]).sum(-1)

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        xi = (x * self.nint).long()  # must be long() for indexing
        g = self.int2gray[xi]
        bs = self.int_to_binary(g)
        return bs

    def decode(self, g):
        g = self.binary_to_int(g)
        xi = self.gray2int[g]
        x = xi / float(self.nint)
        x = x * (self.max - self.min) + self.min
        return x

    # @wrap_type
    def encode_batch(self, x):
        xx, xy = x[:, 0], x[:, 1]
        gx, gy = self.encode(xx), self.encode(xy)
        g = torch.cat([gx, gy], 1)
        return g

    # @wrap_type
    def decode_batch(self, g):
        gx, gy = g[:, :self.nbits], g[:, self.nbits:]
        xx, xy = self.decode(gx), self.decode(gy)
        x = torch.cat([xx[:, None], xy[:, None]], 1)
        return x


def process_data(dataset, num_data, device=torch.device("cpu"), labels=False, cts=False, return_encoder=False):
    dataset = dataset.replace("cts_", "") if cts else dataset
    x, y = toy_data(dataset, num_data)

    if y is None and labels:
        raise NotImplementedError

    x = torch.from_numpy(x).float()

    min_val, max_val = x.min(), x.max()
    delta = max_val - min_val
    buffer = delta / 8.
    encoder = Int2Gray(min_val=min_val-buffer, max_val=max_val+buffer, device=device)

    if not cts:
        x = encoder.encode_batch(x)

    if y is not None and labels:
        y = torch.from_numpy(y).float()
        dataset = TensorDataset(x, y)
    else:
        dataset = TensorDataset(x)

    if return_encoder:
        return dataset, encoder
    else:
        return dataset
