# path: ui/model_tab.py
import copy
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import DEFAULT_SEED, DEVICE
from core.data import get_data
from core.models import build_model, summarize
from core.preview import sample_images
from core.race_engine import run_model_race
from core.registry_datasets import DATASETS, DATASET_ORDER, get_spec
from core.registry_models import MODEL_PRESETS
from core.schemas import RaceConfig, RacerConfig
from core.storage import save_run, load_run, list_runs, replay_logs, total_steps
from core.tables import model_results_rows, badge_text
from core.viz import lanes_figure, loss_figure, acc_figure, params_figure
from ui.garage import render_garage

REDRAW_EVERY = 5        # redraw the live view every N steps (tunable)
EVAL_EVERY = 10         # steps between test-accuracy evaluations
SUBSET = 3000           # training images
SGD_LR = 0.03           # default rate for SGD, Momentum, Nesterov
ADAPTIVE_LR = 0.003     # default rate for Adam, AdamW, RMSprop, Adagrad
BATCH_SIZES = [32, 64, 128]


@st.cache_resource(show_spinner='Loading the dataset (the first use of a dataset downloads it)...')
def cached_data(dataset, subset_size, seed):
    return get_data(subset_size, 1000, seed, dataset=dataset)


@st.cache_data(show_spinner=False)
def _summary_rows(model_name, in_shape, n_classes):
    cfg = RacerConfig(name='x', color='#000000', optimizer='SGD', model=model_name)
    return summarize(build_model(cfg, in_shape, n_classes), in_shape)


@st.cache_data(show_spinner=False)
def _cached_preview(dataset):
    return sample_images(dataset, 8, 0)


def _preview(dataset):
    spec = get_spec(dataset)
    c, h, w = spec.in_shape
    with st.expander(f'About {spec.name}', expanded=False):
        st.caption(f'{spec.lesson} {c} channel(s), {h}x{w} pixels, {spec.n_classes} classes.')
        try:
            imgs = _cached_preview(dataset)
        except Exception as e:
            st.caption(f'No preview available: {e}')
            return
        cols = st.columns(len(imgs))
        for col, (img, name) in zip(cols, imgs):
            col.image(img, caption=name, width=84)


def _make_slots():
    """Placeholders, created after the controls so the visuals sit below them."""
    return dict(badge=st.empty(), progress=st.empty(), status=st.empty(), lanes=st.empty(),
                curves=st.empty(), table=st.empty(), summary=st.empty())


def _draw(slots, logs, racers, rc, tag, status_text=None):
    """One drawing routine for live races, finished races and replays.
    `tag` must be unique within one script run (it keys the charts)."""
    colors = {r.name: r.color for r in racers}
    spec = get_spec(rc.dataset)
    slots['badge'].info(f'{badge_text(racers)}  |  Dataset: {spec.name}')
    if status_text is not None:
        slots['status'].markdown(status_text)
    slots['lanes'].plotly_chart(
        lanes_figure(logs, colors, rc.target_acc, spec.chance),
        width='stretch', key=f'{tag}_lanes')
    with slots['curves'].container():
        c1, c2 = st.columns(2)
        c1.plotly_chart(loss_figure(logs, colors), width='stretch', key=f'{tag}_loss')
        c2.plotly_chart(acc_figure(logs, colors, rc.target_acc), width='stretch',
                        key=f'{tag}_acc')
    with slots['table'].container():
        t1, t2 = st.columns([3, 2])
        t1.dataframe(pd.DataFrame(model_results_rows(logs, racers)),
                     width='stretch', hide_index=True)
        t2.plotly_chart(params_figure(logs, colors), width='stretch', key=f'{tag}_params')


def _draw_summaries(slots, racers, rc):
    """Layer table per racer. Drawn once after a race, replay or restore (not while training)."""
    spec = get_spec(rc.dataset)
    with slots['summary'].container():
        st.markdown('**Model summaries**')
        for r in racers:
            preset = MODEL_PRESETS.get(r.model, {})
            with st.expander(f"{r.name}: {preset.get('label', r.model)}"):
                try:
                    rows = _summary_rows(r.model, spec.in_shape, spec.n_classes)
                except Exception as e:
                    st.caption(f'No summary available: {e}')
                    continue
                st.dataframe(pd.DataFrame([{
                    'Layer': x['layer'], 'Type': x['type'],
                    'Output shape': ' x '.join(str(d) for d in x['output_shape']),
                    'Params': x['params']} for x in rows]),
                    width='stretch', hide_index=True)
                st.caption(f"Total: {sum(x['params'] for x in rows):,} parameters. "
                           f"{preset.get('lesson', '')}")


def _run_live(racers, rc, slots):
    try:
        data = cached_data(rc.dataset, rc.subset_size, rc.seed)
        gen = run_model_race(racers, rc, data=data)       # setup errors raise here
    except ValueError as e:
        st.error(f'Cannot start the race: {e}')
        return None
    except Exception as e:
        st.error(f'Could not start the race: {e}')
        return None

    logs = None
    try:
        for step, logs in gen:
            if step % REDRAW_EVERY == 0 or step == rc.max_steps:
                slots['progress'].progress(min(1.0, step / rc.max_steps))
                _draw(slots, logs, racers, rc, f'live{step}',
                      f'**Racing on {DEVICE.upper()}...** step {step} of {rc.max_steps}')
    except Exception as e:
        st.error(f'Training stopped: {e}')
        return None
    slots['progress'].empty()
    if logs is None:
        return None
    final = copy.deepcopy(logs)
    _draw(slots, final, racers, rc, 'final', '**Race finished.**')
    _draw_summaries(slots, racers, rc)
    path = save_run(racers, rc, final)
    st.success(f'Saved as a replay: {path.name}')
    return dict(logs=final, racers=racers, rc=rc, saved=path.name)


def _replay_controls(path):
    """Slider and Play button for a saved run. Returns a dict, or None if unreadable."""
    try:
        run = load_run(path)
    except ValueError as e:
        st.error(str(e))
        return None
    T = total_steps(run['logs'])
    env = run['env']
    st.caption(f"Replaying: {run['label']} (saved {run['created']}, torch "
               f"{env.get('torch', '?')} on {env.get('device', '?')})")
    if T < 2:
        return dict(run=run, T=T, k=T, play=False)
    k = st.slider('Replay step', 1, T, T, key=f'model_rstep_{Path(path).name}')
    play = st.button('Play replay', key='model_replay_play')
    return dict(run=run, T=T, k=k, play=play)


def _draw_replay(slots, rp):
    run, T = rp['run'], rp['T']
    logs, racers, rc = run['logs'], run['racers'], run['race_cfg']
    if rp['play']:
        marks = list(range(REDRAW_EVERY, T, REDRAW_EVERY)) + [T]
        for kk in marks:
            _draw(slots, replay_logs(logs, kk), racers, rc, f'play{kk}',
                  f'**Replay** step {kk} of {T}')
            time.sleep(0.05)
    else:
        _draw(slots, replay_logs(logs, rp['k']), racers, rc, 'rstatic',
              f"**Replay** step {rp['k']} of {T}")
    _draw_summaries(slots, racers, rc)


def render():
    st.subheader('Real Model Track')
    st.caption('Every racer trains a small CNN on a 3,000-image subset of the chosen dataset, '
               'with the same seed and the same batches in the same order. Pick each racer\'s '
               'network in the Garage. The lane position is the latest test accuracy.')

    racers = render_garage('model', SGD_LR, ADAPTIVE_LR)           # sidebar

    c0, c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.9, 1.3, 0.9])
    dataset = c0.selectbox('Dataset', DATASET_ORDER, format_func=lambda k: DATASETS[k].name,
                           key='model_dataset',
                           help='The images every racer learns from. The first use of a '
                                'dataset downloads it once.')
    spec = get_spec(dataset)
    steps = c1.slider('Steps', 50, 600, 300, step=50, key='model_steps',
                      help='How many training steps (mini-batches) each racer takes.')
    batch = c2.selectbox('Batch size', BATCH_SIZES, index=1, key='model_batch',
                         help='Images per step. Larger batches give smoother, slower steps.')
    target = c3.slider('Target accuracy', 0.30, 0.99, float(spec.default_target), step=0.01,
                       format='%.2f', key=f'model_target_{dataset}',
                       help='A racer finishes the first time its test accuracy reaches this '
                            f'value (checked every {EVAL_EVERY} steps). Each dataset has its '
                            'own default.')
    seed = int(c4.number_input('Seed', 0, 9999, DEFAULT_SEED, step=1, key='model_seed',
                               help='Shared by every racer: same data subset, same batch order.'))
    st.caption(f'Training on {DEVICE.upper()}, seed {seed}. Chance level for {spec.name}: '
               f'{spec.chance:.0%}.')
    _preview(dataset)

    r1, r2 = st.columns([1, 3])
    run_clicked = r1.button('Run race', type='primary', key='model_run')
    runs = list_runs()
    labels = {r['path']: f"{r['label']}  ({r['created']})" for r in runs}
    choice = r2.selectbox('Use cached replay', [''] + [r['path'] for r in runs],
                          format_func=lambda p: 'None (live race)' if p == '' else labels[p],
                          key='model_replay')

    replay = None
    if not run_clicked and choice:
        replay = _replay_controls(choice)

    slots = _make_slots()
    if run_clicked:
        rc = RaceConfig(mode='model', seed=seed, max_steps=steps, dataset=dataset,
                        subset_size=SUBSET, batch_size=batch, target_acc=float(target),
                        eval_every=EVAL_EVERY)
        result = _run_live(racers, rc, slots)
        if result is not None:
            st.session_state['model_result'] = result
    elif replay is not None:
        _draw_replay(slots, replay)
    elif choice:
        pass                                   # unreadable replay: error already shown
    else:
        stored = st.session_state.get('model_result')
        if stored is None:
            st.info('Build your racers in the sidebar and press Run race, or pick a saved replay.')
        else:
            _draw(slots, stored['logs'], stored['racers'], stored['rc'], 'stored',
                  '**Last race** (press Run race to repeat it with new settings)')
            _draw_summaries(slots, stored['racers'], stored['rc'])