# path: tests/test_viz.py
import numpy as np
import plotly.graph_objects as go

from core.landscape_engine import run_landscape_race
from core.schemas import RacerConfig
from core.terrains import TERRAINS
from core.viz import animated_race_figure, surface_figure
from ui.garage import lr_for_optimizer


def cfg(name, opt, lr):
    return RacerConfig(name=name, color='#000000', optimizer=opt, lr=lr)


COLORS = {'SGD': '#e63946', 'Adam': '#2a9d8f'}


def _logs(t, sgd_lr=None, steps=60):
    sgd_lr = t.default_lr if sgd_lr is None else sgd_lr
    return run_landscape_race([cfg('SGD', 'SGD', sgd_lr), cfg('Adam', 'Adam', t.adaptive_lr)],
                              t, t.default_start, max_steps=steps)


def test_slider_labels_are_all_set():
    t = TERRAINS['canyon']
    fig = animated_race_figure(t, _logs(t), COLORS, stride=2)
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert all(labels) and len(labels) == len(fig.frames)


def test_zone_adds_one_rectangle():
    t = TERRAINS['bowl']
    logs = _logs(t)
    assert len(animated_race_figure(t, logs, COLORS).layout.shapes) == 0
    zoned = animated_race_figure(t, logs, COLORS, zone=(-1.0, 1.0, -1.0, 1.0))
    assert len(zoned.layout.shapes) == 1


def test_surface_trace_count():
    t = TERRAINS['canyon']
    logs = _logs(t)
    fig = surface_figure(t, logs, COLORS)
    assert len(fig.data) == 2 + 3 * len(logs)     # surface + minimum + 3 per racer


def test_figures_build_for_every_terrain_even_with_a_crash():
    for key, t in TERRAINS.items():
        logs = _logs(t, sgd_lr=1.0)               # lr 1.0 crashes SGD on most terrains
        afig = animated_race_figure(t, logs, COLORS)
        assert len(afig.frames) >= 2, key
        sfig = surface_figure(t, logs, COLORS)
        for tr in sfig.data:
            if isinstance(tr, go.Scatter3d):
                assert np.all(np.isfinite(np.array(tr.z, dtype=float))), key


def test_lr_for_optimizer_uses_adaptive_rate_for_adaptive_optimizers():
    assert lr_for_optimizer('SGD', 0.002, 0.1) == 0.002
    assert lr_for_optimizer('Momentum', 0.002, 0.1) == 0.002
    assert lr_for_optimizer('Nesterov', 0.002, 0.1) == 0.002
    for name in ('Adam', 'AdamW', 'RMSprop', 'Adagrad'):
        assert lr_for_optimizer(name, 0.002, 0.1) == 0.1
    assert lr_for_optimizer('Adam', 0.002) == 0.002