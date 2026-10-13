# path: core/data.py
from functools import lru_cache

import numpy as np
import torch
from torchvision import datasets

from core.config import DATA_DIR
from core.registry_datasets import get_spec

_TV = {
    'torchvision:FashionMNIST': datasets.FashionMNIST,
    'torchvision:MNIST': datasets.MNIST,
    'torchvision:KMNIST': datasets.KMNIST,
}


@lru_cache(maxsize=6)
def _load_raw(loader, train):
    """Whole split as uint8 (N, C, H, W) plus labels. Downloads on first use
    (FashionMNIST about 30 MB, MNIST about 10 MB, KMNIST about 20 MB)."""
    ds = _TV[loader](DATA_DIR, train=train, download=True)
    data = ds.data
    if isinstance(data, np.ndarray):                  # colour sets: (N, H, W, 3)
        x = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
    else:                                             # MNIST family: (N, 28, 28) tensor
        x = data.unsqueeze(1).contiguous()
    y = torch.as_tensor(ds.targets, dtype=torch.long).clone()
    return x, y


def _prep(x_uint8, idx, spec):
    """Same as ToTensor + Normalize(mean, std), per channel."""
    x = x_uint8[idx].float().div(255.0)
    mean = torch.tensor(spec.mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(spec.std, dtype=torch.float32).view(1, -1, 1, 1)
    return (x - mean) / std


def get_data(subset_size=3000, test_size=1000, seed=42, dataset='fashion_mnist'):
    """Load a dataset into memory tensors (deterministic subset for a seed)."""
    spec = get_spec(dataset)
    if spec.loader not in _TV:
        raise ValueError(f"Dataset '{dataset}' has no loader '{spec.loader}'")
    xtr_raw, ytr_raw = _load_raw(spec.loader, True)
    xte_raw, yte_raw = _load_raw(spec.loader, False)
    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(xtr_raw), generator=g)[:subset_size]
    te_idx = torch.randperm(len(xte_raw), generator=g)[:test_size]
    return dict(
        xtr=_prep(xtr_raw, tr_idx, spec), ytr=ytr_raw[tr_idx].clone(),
        xte=_prep(xte_raw, te_idx, spec), yte=yte_raw[te_idx].clone(),
        in_shape=spec.in_shape, n_classes=spec.n_classes,
        dataset=dataset, class_names=list(spec.class_names))