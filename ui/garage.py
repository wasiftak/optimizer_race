# path: ui/garage.py
import math
import streamlit as st

from core.config import RACER_COLORS, OPTIMIZER_PARAMS, LR_CHOICES
from core.optimizers import OPTIMIZER_NAMES
from core.registry_models import MODEL_ORDER, MODEL_PRESETS
from core.schemas import RacerConfig

_DEFAULT_OPTS = ['SGD', 'Momentum', 'Adam', 'RMSprop']
SGD_FAMILY = ('SGD', 'Momentum', 'Nesterov')
SCHEDULERS = ['none', 'step', 'cosine', 'exponential']


def _nearest_lr(value):
    return min(LR_CHOICES, key=lambda c: abs(math.log(c) - math.log(value)))


def lr_for_optimizer(opt, default_lr, adaptive_lr=None):
    """SGD-family racers use the terrain's default rate; adaptive ones use adaptive_lr."""
    if adaptive_lr is None or opt in SGD_FAMILY:
        return default_lr
    return adaptive_lr


def _model_label(key):
    p = MODEL_PRESETS[key]
    return f"{p['label']} ({p['params']:,} params)"


def render_garage(mode, default_lr, adaptive_lr=None):
    """Sidebar controls -> list[RacerConfig]. Knows nothing about the engine.
    In 'model' mode each racer also picks a network (or all share racer 1's choice)."""
    sb = st.sidebar
    sb.header('Garage')
    n = sb.slider('Number of racers', 2, 4, 3, key=f'{mode}_n')
    same = True
    if mode == 'model':
        same = sb.checkbox(
            'Use the same model for all racers', value=True, key='model_same',
            help='On: a pure optimizer test (identical network and starting weights). '
                 'Off: each racer picks its own network, so optimizer and architecture '
                 'both affect the result.')

    racers, used = [], set()
    shared_model = 'default'
    for i in range(n):
        with sb.expander(f'Racer {i + 1}', expanded=(i == 0)):
            opt = st.selectbox('Optimizer', OPTIMIZER_NAMES,
                               index=OPTIMIZER_NAMES.index(_DEFAULT_OPTS[i]),
                               key=f'{mode}_{i}_opt')
            name = st.text_input('Name', value=opt, key=f'{mode}_{i}_name_{opt}')
            color = st.color_picker('Colour', RACER_COLORS[i], key=f'{mode}_{i}_color')

            model = 'default'
            if mode == 'model':
                if same and i > 0:
                    model = shared_model
                    st.caption(f'Model: {_model_label(model)} (same as Racer 1)')
                else:
                    model = st.selectbox(
                        'Model (shared by all racers)' if same else 'Model',
                        MODEL_ORDER, index=0, format_func=_model_label,
                        key=f'{mode}_{i}_model',
                        help='The network this racer trains.')
                    st.caption(MODEL_PRESETS[model]['lesson'])
                    if i == 0:
                        shared_model = model

            params = OPTIMIZER_PARAMS[opt]
            lr_default = _nearest_lr(lr_for_optimizer(opt, default_lr, adaptive_lr))

            kw = {}
            kw['lr'] = st.select_slider(
                'Learning rate', options=LR_CHOICES, value=lr_default,
                format_func=lambda v: f'{v:g}', key=f'{mode}_{i}_lr_{opt}_{lr_default}',
                help='How big each update step is. Too small: slow crawl. '
                     'Too large: overshoot or crash.')
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
            kw['scheduler'] = st.selectbox(
                'Scheduler', SCHEDULERS, key=f'{mode}_{i}_sched',
                help='Shrinks the learning rate over time: big steps early, small steps late.')

        name = (name.strip() or opt)
        if name in used:                      # names key the colour dict, keep unique
            name = f'{name} ({i + 1})'
        used.add(name)
        racers.append(RacerConfig(name=name, color=color, optimizer=opt, model=model, **kw))
    return racers