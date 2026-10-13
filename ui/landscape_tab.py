# path: ui/landscape_tab.py
import pandas as pd
import streamlit as st

from core.config import RACER_COLORS
from core.schemas import RacerConfig
from core.terrains import TERRAINS
from core.landscape_engine import run_landscape_race
from core.viz import paths_figure


def _racers(lr):
    return [
        RacerConfig(name='SGD', color=RACER_COLORS[0], optimizer='SGD', lr=lr),
        RacerConfig(name='Momentum', color=RACER_COLORS[1], optimizer='Momentum', lr=lr, momentum=0.9),
        RacerConfig(name='Adam', color=RACER_COLORS[2], optimizer='Adam', lr=lr),
    ]


def _results_table(logs):
    rows = []
    for log in logs:
        rows.append({
            'Racer': log.name,
            'Status': log.status,
            'Steps': log.steps[-1],
            'Final loss': f'{log.loss[-1]:.6g}',
        })
    return pd.DataFrame(rows)


def render():
    st.subheader('Landscape Track (static paths)')
    keys = list(TERRAINS.keys())
    key = st.selectbox('Terrain', keys, format_func=lambda k: TERRAINS[k].name,
                       key='land_terrain')
    terrain = TERRAINS[key]
    st.caption(terrain.lesson)

    c1, c2 = st.columns(2)
    with c1:
        lr = st.slider('Learning rate (shared by all racers)', 0.0005, 0.5,
                       float(terrain.default_lr), step=0.0005, format='%.4f',
                       key=f'land_lr_{key}')
    with c2:
        max_steps = st.slider('Max steps', 50, 1000, 300, step=50, key='land_steps')

    if st.button('Run race', type='primary', key='land_run'):
        racers = _racers(lr)
        logs = run_landscape_race(racers, terrain, terrain.default_start, max_steps=max_steps)
        st.session_state['land_result'] = dict(
            terrain_key=key, lr=lr, logs=logs,
            colors={r.name: r.color for r in racers})

    res = st.session_state.get('land_result')
    if res is None:
        st.info('Pick a terrain and press Run race.')
        return

    t = TERRAINS[res['terrain_key']]
    st.markdown(f"**Showing:** {t.name}, learning rate {res['lr']}")
    st.plotly_chart(paths_figure(t, res['logs'], res['colors']), width='stretch')
    st.dataframe(_results_table(res['logs']), width='stretch', hide_index=True)