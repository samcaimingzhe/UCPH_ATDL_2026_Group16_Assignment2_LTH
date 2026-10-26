"""All datasets are converted to uint8 tensors of shape N x 3 x 32 x 32, cached on
disk and kept on the GPU during training (these datasets are small, so this
removes the DataLoader bottleneck that dominates training of tiny conv nets).

No data augmentation is used, matching the Conv-2/4/6 setup of Frankle & Carbin.
"""
import os

import numpy as np
import torch

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "svhn": 10,
               "fashionmnist": 10, "mnist": 10, "fake10": 10, "fake5": 5}


def _to_uint8_nchw(x):
    x = torch.as_tensor(np.asarray(x))
    if x.dim() == 3:                      # N,H,W grayscale
        x = x.unsqueeze(1)
    elif x.shape[-1] == 3:                # N,H,W,C
        x = x.permute(0, 3, 1, 2)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[-1] == 28:                 # pad MNIST-style 28x28 -> 32x32
        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
    return x.contiguous().to(torch.uint8)


def _fake(name):
    """Synthetic, learnable data for smoke tests (no download needed)."""
    k = NUM_CLASSES[name]
    g = torch.Generator().manual_seed(1 if name == "fake10" else 2)
    protos = torch.rand(k, 3, 8, 8, generator=g)
    protos = torch.nn.functional.interpolate(protos, size=32, mode="bilinear")

    def make(n, seed):
        gg = torch.Generator().manual_seed(seed)
        y = torch.randint(0, k, (n,), generator=gg)
        x = protos[y] + 0.35 * torch.randn(n, 3, 32, 32, generator=gg)
        return (x.clamp(0, 1) * 255).to(torch.uint8), y

    xtr, ytr = make(3000, 10)
    xte, yte = make(1000, 11)
    return xtr, ytr, xte, yte


def _torchvision(name, root, download):
    import torchvision.datasets as D
    if name == "cifar10":
        tr, te = D.CIFAR10(root, True, download=download), D.CIFAR10(root, False, download=download)
        return tr.data, tr.targets, te.data, te.targets
    if name == "cifar100":
        tr, te = D.CIFAR100(root, True, download=download), D.CIFAR100(root, False, download=download)
        return tr.data, tr.targets, te.data, te.targets
    if name == "svhn":
        tr = D.SVHN(root, "train", download=download)
        te = D.SVHN(root, "test", download=download)
        return tr.data.transpose(0, 2, 3, 1), tr.labels, te.data.transpose(0, 2, 3, 1), te.labels
    if name in ("fashionmnist", "mnist"):
        cls = D.FashionMNIST if name == "fashionmnist" else D.MNIST
        tr, te = cls(root, True, download=download), cls(root, False, download=download)
        return tr.data, tr.targets, te.data, te.targets
    raise ValueError(f"unknown dataset {name}")


def load_raw(name, root, download=False):
    cache = os.path.join(root, "cache", f"{name}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    if name.startswith("fake"):
        xtr, ytr, xte, yte = _fake(name)
    else:
        a, b, c, d = _torchvision(name, root, download)
        xtr, xte = _to_uint8_nchw(a), _to_uint8_nchw(c)
        ytr, yte = torch.as_tensor(np.asarray(b)).long(), torch.as_tensor(np.asarray(d)).long()
    out = {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}
    if not name.startswith("fake"):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(out, cache)
    return out


class GPUData:
    def __init__(self, name, root, device, val_size=5000, train_subset=None,
                 download=False, split_seed=0):
        raw = load_raw(name, root, download)
        n = raw["xtr"].shape[0]
        val_size = min(val_size, n // 10) if name.startswith("fake") else val_size
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
        val_idx, tr_idx = perm[:val_size], perm[val_size:]
        if train_subset:
            tr_idx = tr_idx[:train_subset]
        self.name = name
        self.num_classes = NUM_CLASSES[name]
        self.device = device
        x_tr = raw["xtr"][tr_idx].float() / 255
        self.mean = x_tr.mean((0, 2, 3)).view(1, 3, 1, 1).to(device)
        self.std = x_tr.std((0, 2, 3)).view(1, 3, 1, 1).to(device)
        self.split = {
            "train": (raw["xtr"][tr_idx].to(device), raw["ytr"][tr_idx].to(device)),
            "val": (raw["xtr"][val_idx].to(device), raw["ytr"][val_idx].to(device)),
            "test": (raw["xte"].to(device), raw["yte"].to(device)),
        }

    def size(self, split):
        return self.split[split][0].shape[0]

    def _norm(self, x):
        return (x.float().div_(255) - self.mean) / self.std

    def batches(self, split, batch_size, shuffle=False, seed=0):
        x, y = self.split[split]
        n = x.shape[0]
        if shuffle:
            g = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.randperm(n, generator=g).to(self.device)
        else:
            idx = torch.arange(n, device=self.device)
        stop = n - n % batch_size if shuffle else n   # drop last partial batch in training
        for i in range(0, stop, batch_size):
            j = idx[i:i + batch_size]
            yield self._norm(x[j]), y[j]
