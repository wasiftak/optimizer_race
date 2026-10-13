# path: tests/test_data.py
import math
import pytest
import torch

from core.data import get_data
from core.models import build_model, count_params
from core.race_engine import run_to_end
from core.registry_datasets import DATASETS, DATASET_ORDER, get_spec
from core.registry_models import MODEL_ORDER
from core.schemas import RaceConfig, RacerConfig

KEYS = list(DATASET_ORDER)
EXPECTED_PARAMS = {'fashion_mnist': 20490, 'mnist': 20490, 'kmnist': 20490}


def load(key, subset=64, test=32, seed=1):
    """Only network or disk problems skip a test; real code errors still fail it."""
    try:
        return get_data(subset, test, seed, dataset=key)
    except (OSError, RuntimeError) as e:
        pytest.skip(f'{key} could not be loaded: {e}')


def test_registry_specs_are_consistent():
    assert KEYS == ['fashion_mnist', 'mnist', 'kmnist']
    for key, s in DATASETS.items():
        assert s.key == key
        assert len(s.mean) == len(s.std) == s.in_shape[0]
        assert len(s.class_names) == s.n_classes
        assert 0.0 < s.default_target < 1.0
        assert abs(s.chance - 1.0 / s.n_classes) < 1e-12
        assert s.lesson and s.loader.startswith('torchvision:')


def test_unknown_dataset_raises_keyerror_listing_valid_names():
    with pytest.raises(KeyError, match='kmnist'):
        get_spec('nonsense')


@pytest.mark.parametrize('key', KEYS)
def test_shapes_dtypes_and_label_range(key):
    spec = get_spec(key)
    d = load(key)
    assert d['xtr'].shape == (64, *spec.in_shape) and d['xte'].shape == (32, *spec.in_shape)
    assert d['xtr'].dtype == torch.float32 and d['ytr'].dtype == torch.int64
    for y in (d['ytr'], d['yte']):
        assert y.min() >= 0 and y.max() < spec.n_classes
    assert d['in_shape'] == spec.in_shape and d['n_classes'] == spec.n_classes
    assert d['class_names'] == spec.class_names


@pytest.mark.parametrize('key', KEYS)
def test_inputs_are_normalised(key):
    d = load(key, subset=500, test=100)
    assert abs(d['xtr'].mean().item()) < 0.25
    assert 0.75 < d['xtr'].std().item() < 1.25


@pytest.mark.parametrize('key', KEYS)
def test_same_seed_same_subset(key):
    a = load(key, subset=200, test=50, seed=3)
    b = get_data(200, 50, 3, dataset=key)
    assert torch.equal(a['xtr'], b['xtr']) and torch.equal(a['ytr'], b['ytr'])
    assert torch.equal(a['xte'], b['xte'])


@pytest.mark.parametrize('key', KEYS)
def test_different_seed_different_subset(key):
    a = load(key, subset=200, test=50, seed=3)
    b = get_data(200, 50, 4, dataset=key)
    assert not torch.equal(a['ytr'], b['ytr'])


def test_default_dataset_is_fashion_mnist():
    a = load('fashion_mnist', subset=64, test=32, seed=1)
    b = get_data(64, 32, 1)
    assert torch.equal(a['xtr'], b['xtr']) and b['dataset'] == 'fashion_mnist'


@pytest.mark.parametrize('key', KEYS)
def test_every_model_adapts_to_every_dataset(key):
    spec = get_spec(key)
    for name in MODEL_ORDER:
        cfg = RacerConfig(name='a', color='#000000', optimizer='SGD', model=name)
        m = build_model(cfg, spec.in_shape, spec.n_classes)
        assert m(torch.zeros(2, *spec.in_shape)).shape == (2, spec.n_classes), (key, name)
        if name == 'default':
            assert count_params(m) == EXPECTED_PARAMS[key]


@pytest.mark.parametrize('key', KEYS)
def test_engine_loads_the_requested_dataset(key):
    load(key)                                   # skips cleanly when offline
    rc = RaceConfig(mode='model', seed=1, max_steps=5, dataset=key, subset_size=64,
                    batch_size=32, target_acc=1.01, eval_every=5)
    racers = [RacerConfig(name='A', color='#000000', optimizer='Adam', lr=0.005)]
    log = run_to_end(racers, rc, device='cpu')[0]
    assert log.n_params == EXPECTED_PARAMS[key]
    assert len(log.loss) == 5 and all(math.isfinite(v) for v in log.loss)