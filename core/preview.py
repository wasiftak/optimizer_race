# path: core/preview.py
import numpy as np
import torch

from core.data import _load_raw
from core.registry_datasets import get_spec


def sample_images(dataset, n=8, seed=0):
    """n sample test images as (uint8 array, class name). Gray: (H, W). Colour: (H, W, 3)."""
    spec = get_spec(dataset)
    x, y = _load_raw(spec.loader, False)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:n].tolist()
    out = []
    for i in idx:
        img = x[i].numpy()                          # (C, H, W) uint8
        img = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))
        out.append((np.ascontiguousarray(img), spec.class_names[int(y[i])]))
    return out