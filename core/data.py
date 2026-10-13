# path: core/data.py
import torch
from torchvision import datasets

from core.config import DATA_DIR

FASHION_MEAN, FASHION_STD = 0.2860, 0.3530


def _prep(raw, idx):
    """uint8 (N,28,28) -> normalised float (n,1,28,28); same as ToTensor + Normalize."""
    x = raw[idx].float().div(255.0).unsqueeze(1)
    return (x - FASHION_MEAN) / FASHION_STD


def get_data(subset_size=3000, test_size=1000, seed=42):
    """Load FashionMNIST once into memory tensors (deterministic subset for a seed).
    The first call downloads about 30 MB into data/."""
    train = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(train), generator=g)[:subset_size]
    te_idx = torch.randperm(len(test), generator=g)[:test_size]
    return dict(
        xtr=_prep(train.data, tr_idx), ytr=train.targets[tr_idx].clone(),
        xte=_prep(test.data, te_idx), yte=test.targets[te_idx].clone(),
        in_shape=(1, 28, 28), n_classes=10)