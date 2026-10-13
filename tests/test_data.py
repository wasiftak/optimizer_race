# path: tests/test_data.py
import pytest
import torch
from core.data import get_data


@pytest.fixture(scope='module')
def small():
    try:
        return get_data(subset_size=500, test_size=200, seed=1)
    except Exception as e:                       # offline on the first run
        pytest.skip(f'FashionMNIST could not be loaded: {e}')


def test_shapes_and_label_range(small):
    assert small['xtr'].shape == (500, 1, 28, 28)
    assert small['xte'].shape == (200, 1, 28, 28)
    assert small['ytr'].min() >= 0 and small['ytr'].max() <= 9
    assert small['yte'].min() >= 0 and small['yte'].max() <= 9
    assert small['in_shape'] == (1, 28, 28) and small['n_classes'] == 10


def test_inputs_are_normalised(small):
    assert abs(small['xtr'].mean().item()) < 0.2
    assert 0.8 < small['xtr'].std().item() < 1.2


def test_same_seed_same_subset(small):
    again = get_data(subset_size=500, test_size=200, seed=1)
    assert torch.equal(small['xtr'], again['xtr'])
    assert torch.equal(small['ytr'], again['ytr'])


def test_different_seed_different_subset(small):
    other = get_data(subset_size=500, test_size=200, seed=2)
    assert not torch.equal(small['ytr'], other['ytr'])