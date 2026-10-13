# path: tests/test_model_viz.py
import pytest

from core.data import get_data
from core.race_engine import run_to_end
from core.schemas import RaceConfig, RacerConfig, RacerLog
from core.storage import replay_logs
from core.viz import (moving_average, latest_accuracy, lanes_figure, loss_figure,
                      acc_figure, CRASH_GREY)

COLORS = {'A': '#e63946', 'B': '#1d7bd8', 'C': '#2a9d8f'}


def hand(name, acc, status='running', loss=None):
    n = len(acc)
    return RacerLog(name=name, steps=list(range(1, n + 1)),
                    loss=loss or [1.0] * n, acc=list(acc), status=status)


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
    assert moving_average([1, 2, 3], 1) == [1, 2, 3]


def test_latest_accuracy_forward_fills():
    assert latest_accuracy(hand('A', [None, 0.5, None]), 0.1) == 0.5
    assert latest_accuracy(hand('A', [None, None]), 0.1) == 0.1


def test_lanes_values_and_finish_line():
    logs = [hand('A', [None, 0.5, None]), hand('B', [None, None])]
    fig = lanes_figure(logs, COLORS, target_acc=0.8, chance=0.1)
    assert list(fig.data[0].x) == [0.5, 0.1]
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == 0.8


def test_lanes_crashed_and_finished_markers():
    logs = [hand('A', [None, 0.9], status='finished'),
            hand('B', [None, 0.2], status='crashed'),
            hand('C', [None, 0.4])]
    fig = lanes_figure(logs, COLORS, target_acc=0.8)
    assert list(fig.data[1].marker.symbol) == ['star', 'x', 'diamond']
    assert fig.data[0].marker.color[1] == CRASH_GREY


def test_loss_figure_smoothing():
    logs = [hand('A', [None] * 4, loss=[1, 2, 3, 4])]
    fig = loss_figure(logs, COLORS, smooth=2)
    assert len(fig.data) == 1 and list(fig.data[0].y) == [1, 1.5, 2.5, 3.5]


def test_acc_figure_only_evaluated_points():
    logs = [hand('A', [None, 0.5, None, 0.7])]
    fig = acc_figure(logs, COLORS, target_acc=0.8)
    assert list(fig.data[0].x) == [2, 4] and list(fig.data[0].y) == [0.5, 0.7]
    assert len(fig.layout.shapes) == 1


@pytest.fixture(scope='module')
def small():
    try:
        return get_data(subset_size=300, test_size=200, seed=1)
    except Exception as e:
        pytest.skip(f'FashionMNIST could not be loaded: {e}')


def test_figures_work_on_real_and_replayed_logs(small):
    racers = [RacerConfig(name='A', color=COLORS['A'], optimizer='Adam', lr=0.005),
              RacerConfig(name='B', color=COLORS['B'], optimizer='SGD', lr=0.02)]
    rc = RaceConfig(mode='model', seed=1, max_steps=10, batch_size=32,
                    target_acc=1.01, eval_every=5)
    logs = run_to_end(racers, rc, data=small, device='cpu')
    for view in (logs, replay_logs(logs, 5)):
        assert len(lanes_figure(view, COLORS, rc.target_acc).data) == 2
        assert len(loss_figure(view, COLORS).data) == 2
        assert len(acc_figure(view, COLORS, rc.target_acc).data) == 2