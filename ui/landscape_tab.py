# path: ui/landscape_tab.py
import pandas as pd
import streamlit as st

from core.terrains import TERRAINS
from core.landscape_engine import run_landscape_race, rank_landscape
from core.viz import animated_race_figure
from ui.garage import render_garage

PRESETS = ['Default start', 'Far corner', 'Near the minimum', 'Custom (use sliders)']


def _clamp(v, lo, hi):
    return round(max(lo, min(hi, v)), 2)


def _preset_start(terrain, preset):
    xmin, xmax, ymin, ymax = terrain.bounds
    w, h = xmax - xmin, ymax - ymin
    if preset == 'Far corner':
        return (_clamp(xmin + 0.1 * w, xmin, xmax), _clamp(ymax - 0.1 * h, ymin, ymax))
    if preset == 'Near the minimum':
        mx, my = terrain.minimum
        return (_clamp(mx + 0.1 * w, xmin, xmax), _clamp(my + 0.1 * h, ymin, ymax))
    return tuple(_clamp(v, lo, hi) for v, lo, hi in
                 zip(terrain.default_start, (xmin, ymin), (xmax, ymax)))


def _table(ranking):
    return pd.DataFrame([{
        'Rank': r['rank'], 'Racer': r['name'], 'Status': r['status'],
        'Steps': r['steps'], 'Final loss': f"{r['final_loss']:.6g}"}
        for r in ranking])


def render():
    st.subheader('Landscape Track')
    key = st.selectbox('Terrain', list(TERRAINS.keys()),
                       format_func=lambda k: TERRAINS[k].name, key='land_terrain')
    terrain = TERRAINS[key]
    st.caption(terrain.lesson)

    racers = render_garage('land', terrain.default_lr)   # sidebar

    # Start point: shared by every racer (fairness rule 2)
    xmin, xmax, ymin, ymax = terrain.bounds
    c0, c1, c2 = st.columns([2, 1, 1])
    preset = c0.selectbox('Start point', PRESETS, key=f'land_preset_{key}')
    sx0, sy0 = _preset_start(terrain, preset)
    custom = preset.startswith('Custom')
    sx = c1.slider('Start x', float(xmin), float(xmax), float(sx0), step=0.05,
                   disabled=not custom, key=f'land_sx_{key}_{preset}')
    sy = c2.slider('Start y', float(ymin), float(ymax), float(sy0), step=0.05,
                   disabled=not custom, key=f'land_sy_{key}_{preset}')
    start = (sx, sy)

    d1, d2 = st.columns(2)
    max_steps = d1.slider('Max steps', 50, 1000, 300, step=50, key='land_steps')
    speed = d2.slider('Playback speed', 1, 5, 2, key='land_speed')

    if st.button('Run race', type='primary', key='land_run'):
        try:
            logs = run_landscape_race(racers, terrain, start, max_steps=max_steps)
        except ValueError as e:
            st.error(f'Cannot start the race: {e}')
        else:
            st.session_state['land_result'] = dict(
                terrain_key=key, start=start, logs=logs,
                colors={r.name: r.color for r in racers})

    res = st.session_state.get('land_result')
    if res is None:
        st.info('Build your racers in the sidebar, pick a start, and press Run race.')
        return

    t = TERRAINS[res['terrain_key']]
    st.markdown(f"**Showing:** {t.name}, start {res['start']}. Press Play under the map.")
    fig = animated_race_figure(t, res['logs'], res['colors'],
                               stride=speed, frame_ms=50)
    st.plotly_chart(fig, width='stretch')
    st.caption(t.lesson)
    st.dataframe(_table(rank_landscape(res['logs'])), width='stretch', hide_index=True)