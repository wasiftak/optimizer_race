# path: tests/test_models_race.py
import math
import pytest

from core.data import get_data
from core.race_engine import run_to_end
from core.schemas import RaceConfig, RacerConfig


@pytest.fixture(scope='module')
def small():
    try:
        return get_data(subset_size=300, test_size=200, seed=1)
    except Exception as e:
        pytest.skip(f'FashionMNIST could not be loaded: {e}')


def cfg(name, model, opt='Adam', lr=0.005):
    return RacerConfig(name=name, color='#000000', optimizer=opt, lr=lr, model=model)


def rc(**kw):
    base = dict(mode='model', seed=1, max_steps=10, batch_size=32, target_acc=1.01,
                eval_every=5)
    base.update(kw)
    return RaceConfig(**base)


def run(racers, race_cfg, small):
    return run_to_end(racers, race_cfg, data=small, device='cpu')


def test_mixed_race_logs_each_models_parameter_count(small):
    logs = run([cfg('A', 'default'), cfg('B', 'mini_vgg'), cfg('C', 'deep_narrow')], rc(), small)
    assert [l.n_params for l in logs] == [20490, 32058, 5762]


def test_same_architecture_racers_start_identical_in_a_mixed_race(small):
    # lr 0 freezes the weights, so equal loss lists prove identical starting weights
    racers = [cfg('A', 'mini_vgg', 'SGD', 0.0), cfg('B', 'default', 'SGD', 0.0),
              cfg('C', 'mini_vgg', 'Adam', 0.0)]
    logs = run(racers, rc(), small)
    assert logs[0].loss == logs[2].loss
    assert logs[0].loss != logs[1].loss


def test_mixed_race_is_reproducible(small):
    racers = [cfg('A', 'default'), cfg('B', 'tiny_resnet')]
    a = run(racers, rc(seed=3), small)
    b = run(racers, rc(seed=3), small)
    assert [l.loss for l in a] == [l.loss for l in b]


def test_all_five_presets_train_in_lockstep(small):
    names = ['default', 'mini_vgg', 'tiny_resnet', 'wide_shallow', 'deep_narrow']
    logs = run([cfg(n, n) for n in names], rc(max_steps=5), small)
    for l in logs:
        assert len(l.loss) == 5 and all(math.isfinite(v) for v in l.loss), l.name
        assert l.status == 'timeout'