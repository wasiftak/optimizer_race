# path: tests/test_phase2.py
import pytest
import torch
from core.config import OPTIMIZER_PARAMS
from core.optimizers import build_optimizer, OPTIMIZER_NAMES
from core.schemas import RacerConfig, RacerLog
from core.terrains import TERRAINS
from core.landscape_engine import run_landscape_race, rank_landscape
from core.viz import animated_race_figure


def cfg(name, lr=0.01, **kw):
    return RacerConfig(name=name, color='#000000', optimizer=name, lr=lr, **kw)


def hand_log(name, status, loss, steps=10, finish=None):
    return RacerLog(name=name, steps=list(range(steps)), loss=[loss] * steps,
                    x=[0.0] * steps, y=[0.0] * steps, status=status, finish_step=finish)


def test_rank_finished_before_crashed():
    logs = [hand_log('crash', 'crashed', float('nan')),
            hand_log('slow', 'finished', 0.0, finish=80),
            hand_log('fast', 'finished', 0.0, finish=20),
            hand_log('time', 'timeout', 0.5)]
    order = [r['name'] for r in rank_landscape(logs)]
    assert order == ['fast', 'slow', 'time', 'crash']


def test_rank_timeouts_by_final_loss():
    logs = [hand_log('a', 'timeout', 2.0), hand_log('b', 'timeout', 0.1)]
    assert [r['name'] for r in rank_landscape(logs)] == ['b', 'a']


EXPECTED = {'SGD': torch.optim.SGD, 'Momentum': torch.optim.SGD, 'Nesterov': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad, 'RMSprop': torch.optim.RMSprop,
            'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}


@pytest.mark.parametrize('name', OPTIMIZER_NAMES)
def test_build_optimizer_class(name):
    opt = build_optimizer(cfg(name), [torch.zeros(2, requires_grad=True)])
    assert type(opt) is EXPECTED[name]


def test_optimizer_params_cover_all_names():
    assert set(OPTIMIZER_PARAMS) == set(OPTIMIZER_NAMES)


def test_sgd_high_rate_crashes_and_ranks_last():
    t = TERRAINS['canyon']
    logs = run_landscape_race([cfg('SGD', 0.3), cfg('Adam', 0.08)], t, t.default_start)
    ranking = rank_landscape(logs)
    assert ranking[-1]['name'] == 'SGD' and ranking[-1]['status'] == 'crashed'
    assert ranking[0]['status'] != 'crashed'


def test_animated_figure_has_frames_and_slider():
    t = TERRAINS['canyon']
    logs = run_landscape_race([cfg('SGD', 0.3), cfg('Momentum', 0.05)], t, t.default_start)
    colors = {'SGD': '#e63946', 'Momentum': '#1d7bd8'}
    fig = animated_race_figure(t, logs, colors, stride=2)
    assert len(fig.frames) > 1
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    assert len(fig.data) == 2 + 2 * len(logs)      # contour + star + trails + cars