# path: ui/garage.py
import math
import streamlit as st

from core.config import RACER_COLORS, OPTIMIZER_PARAMS, LR_CHOICES
from core.optimizers import OPTIMIZER_NAMES
from core.schemas import RacerConfig

_DEFAULT_OPTS = ['SGD', 'Momentum', 'Adam', 'RMSprop']


def _nearest_lr(value):
    return min(LR_CHOICES, key=lambda c: abs(math.log(c) - math.log(value)))


def render_garage(mode, default_lr):
    """Sidebar controls -> list[RacerConfig]. Knows nothing about the engine."""
    sb = st.sidebar
    sb.header('Garage')
    n = sb.slider('Number of racers', 2, 4, 3, key=f'{mode}_n')
    lr_default = _nearest_lr(default_lr)

    racers, used = [], set()
    for i in range(n):
        with sb.expander(f'Racer {i + 1}', expanded=(i == 0)):
            opt = st.selectbox('Optimizer', OPTIMIZER_NAMES,
                               index=OPTIMIZER_NAMES.index(_DEFAULT_OPTS[i]),
                               key=f'{mode}_{i}_opt')
            name = st.text_input('Name', value=opt, key=f'{mode}_{i}_name_{opt}')
            color = st.color_picker('Colour', RACER_COLORS[i], key=f'{mode}_{i}_color')
            params = OPTIMIZER_PARAMS[opt]

            kw = {}
            kw['lr'] = st.select_slider(
                'Learning rate', options=LR_CHOICES, value=lr_default,
                format_func=lambda v: f'{v:g}', key=f'{mode}_{i}_lr_{lr_default}')
            if 'momentum' in params:
                kw['momentum'] = st.slider('Momentum', 0.0, 0.99, 0.9, step=0.01,
                                           key=f'{mode}_{i}_mom')
            if 'beta1' in params:
                kw['beta1'] = st.slider('Beta 1', 0.5, 0.999, 0.9, step=0.001,
                                        format='%.3f', key=f'{mode}_{i}_b1')
                kw['beta2'] = st.slider('Beta 2', 0.9, 0.9999, 0.999, step=0.0001,
                                        format='%.4f', key=f'{mode}_{i}_b2')
            if 'weight_decay' in params:
                kw['weight_decay'] = st.number_input(
                    'Weight decay', 0.0, 1.0, 0.0, step=0.0001, format='%.4f',
                    key=f'{mode}_{i}_wd')

        name = (name.strip() or opt)
        if name in used:                      # names key the colour dict, keep unique
            name = f'{name} ({i + 1})'
        used.add(name)
        racers.append(RacerConfig(name=name, color=color, optimizer=opt, **kw))
    return racers