# path: tests/test_optimizers.py
import pytest
import torch
from core.schemas import RacerConfig
from core.optimizers import build_optimizer, OPTIMIZER_NAMES
from core.terrains import TERRAINS


def make_cfg(name, **kw):
    return RacerConfig(name='t', color='#000000', optimizer=name, lr=0.01, **kw)


@pytest.mark.parametrize('name', OPTIMIZER_NAMES)
def test_each_optimizer_lowers_loss_on_bowl(name):
    bowl = TERRAINS['bowl']
    p = torch.tensor(bowl.default_start, dtype=torch.float32, requires_grad=True)
    opt = build_optimizer(make_cfg(name), [p])
    start = bowl.fn(p).item()
    for _ in range(100):
        opt.zero_grad()
        bowl.fn(p).backward()
        opt.step()
    assert bowl.fn(p).item() < start


def test_unknown_optimizer_raises():
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        build_optimizer(make_cfg('nonsense'), [p])


def test_nesterov_needs_momentum():
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        build_optimizer(make_cfg('Nesterov', momentum=0.0), [p])