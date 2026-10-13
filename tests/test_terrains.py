# path: tests/test_terrains.py
import torch
from core.terrains import TERRAINS


def test_minimum_equals_fmin():
    for key, t in TERRAINS.items():
        val = t.fn(torch.tensor(t.minimum, dtype=torch.float32)).item()
        assert abs(val - t.fmin) < 1e-6, f'{key}: {val} != {t.fmin}'


def test_grid_evaluation_shape():
    for key, t in TERRAINS.items():
        grid = torch.zeros(2, 5, 7)
        assert t.fn(grid).shape == (5, 7), key