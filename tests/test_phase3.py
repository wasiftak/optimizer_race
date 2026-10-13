# path: tests/test_phase3.py
import pytest
import torch
from core.config import LR_CHOICES
from core.optimizers import build_optimizer, build_scheduler
from core.schemas import RacerConfig
from core.terrains import TERRAINS
from core.landscape_engine import run_landscape_race


def cfg(name='r', opt='SGD', lr=0.05, **kw):
    return RacerConfig(name=name, color='#000000', optimizer=opt, lr=lr, **kw)


def test_himmelblau_minimum():
    t = TERRAINS['himmelblau']
    assert abs(t.fn(torch.tensor([3.0, 2.0])).item()) < 1e-4


def test_saddle_minimum():
    t = TERRAINS['saddle']
    assert abs(t.fn(torch.tensor([0.0, 1.4142])).item() - (-1.0)) < 1e-4


def test_terrain_defaults_are_valid():
    assert len(TERRAINS) == 7
    for key, t in TERRAINS.items():
        xmin, xmax, ymin, ymax = t.bounds
        assert xmin <= t.default_start[0] <= xmax, key
        assert ymin <= t.default_start[1] <= ymax, key
        assert t.default_lr in LR_CHOICES, key
        assert t.adaptive_lr in LR_CHOICES, key


def test_step_scheduler_halves_lr():
    p = torch.zeros(2, requires_grad=True)
    opt = build_optimizer(cfg(lr=0.1), [p])
    sch = build_scheduler(cfg(scheduler='step'), opt, 30)    # step_size 10, gamma 0.5
    for _ in range(10):
        opt.step()
        sch.step()
    assert abs(opt.param_groups[0]['lr'] - 0.05) < 1e-9


def test_cosine_scheduler_decays_to_zero():
    p = torch.zeros(2, requires_grad=True)
    opt = build_optimizer(cfg(lr=0.1), [p])
    sch = build_scheduler(cfg(scheduler='cosine'), opt, 50)
    for _ in range(50):
        opt.step()
        sch.step()
    assert opt.param_groups[0]['lr'] < 1e-6


def test_unknown_scheduler_raises():
    p = torch.zeros(2, requires_grad=True)
    opt = build_optimizer(cfg(), [p])
    with pytest.raises(ValueError):
        build_scheduler(cfg(scheduler='nonsense'), opt, 50)


def test_scheduler_changes_engine_path():
    t = TERRAINS['canyon']
    a = run_landscape_race([cfg(opt='SGD', lr=0.05)], t, t.default_start, max_steps=30)[0]
    b = run_landscape_race([cfg(opt='SGD', lr=0.05, scheduler='exponential')], t,
                           t.default_start, max_steps=30)[0]
    assert a.loss != b.loss


def test_noise_is_reproducible_and_seed_dependent():
    t = TERRAINS['bowl']
    hz = {'noise_sigma': 0.5}
    a = run_landscape_race([cfg()], t, t.default_start, max_steps=50, hazards=hz, seed=7)[0]
    b = run_landscape_race([cfg()], t, t.default_start, max_steps=50, hazards=hz, seed=7)[0]
    c = run_landscape_race([cfg()], t, t.default_start, max_steps=50, hazards=hz, seed=8)[0]
    assert a.x == b.x and a.y == b.y
    assert a.x != c.x


def test_zero_noise_equals_no_hazard():
    t = TERRAINS['bowl']
    a = run_landscape_race([cfg()], t, t.default_start, max_steps=50)[0]
    b = run_landscape_race([cfg()], t, t.default_start, max_steps=50,
                           hazards={'noise_sigma': 0.0})[0]
    assert a.x == b.x


def test_noise_changes_path():
    t = TERRAINS['bowl']
    a = run_landscape_race([cfg()], t, t.default_start, max_steps=50)[0]
    b = run_landscape_race([cfg()], t, t.default_start, max_steps=50,
                           hazards={'noise_sigma': 0.5})[0]
    assert a.x != b.x


def test_identical_racers_get_identical_noise():
    t = TERRAINS['bowl']
    logs = run_landscape_race([cfg(name='a'), cfg(name='b')], t, t.default_start,
                              max_steps=50, hazards={'noise_sigma': 0.5}, seed=3)
    assert logs[0].x == logs[1].x and logs[0].loss == logs[1].loss


def test_danger_zone_kicks_racer():
    t = TERRAINS['bowl']
    hz = {'zone': (-1.0, 1.0, -1.0, 1.0), 'kick': 0.5}
    a = run_landscape_race([cfg()], t, (0.5, 0.5), max_steps=40)[0]
    b = run_landscape_race([cfg()], t, (0.5, 0.5), max_steps=40, hazards=hz, seed=1)[0]
    assert a.x != b.x


def test_far_away_zone_has_no_effect():
    t = TERRAINS['bowl']
    hz = {'zone': (2.5, 3.0, -3.0, -2.5), 'kick': 0.5}
    a = run_landscape_race([cfg()], t, t.default_start, max_steps=40)[0]
    b = run_landscape_race([cfg()], t, t.default_start, max_steps=40, hazards=hz)[0]
    assert a.x == b.x