# path: ui/landscape_tab.py
import pandas as pd
import streamlit as st

from core.commentary import generate_commentary, explain_racer
from core.config import DEFAULT_SEED
from core.terrains import TERRAINS
from core.landscape_engine import run_landscape_race, rank_landscape
from core.viz import animated_race_figure, surface_figure
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


def _hazard_controls(terrain, key, start):
    """Returns (hazards dict, zone tuple or None, seed)."""
    xmin, xmax, ymin, ymax = terrain.bounds
    w, h = xmax - xmin, ymax - ymin
    with st.expander('Hazards and seed', expanded=False):
        seed = int(st.number_input(
            'Random seed (shared by every racer)', min_value=0, max_value=9999,
            value=DEFAULT_SEED, step=1, key='land_seed',
            help='The same seed gives the same wind and the same kicks.'))
        wind = st.checkbox('Wind: random noise added to every gradient', key='land_wind')
        sigma = st.slider('Wind strength', 0.0, 3.0, 0.5, step=0.05,
                          disabled=not wind, key='land_sigma')
        zone_on = st.checkbox('Danger zone: random kicks inside a rectangle',
                              key='land_zone_on')
        mx = (start[0] + terrain.minimum[0]) / 2
        my = (start[1] + terrain.minimum[1]) / 2
        zx_def = (_clamp(mx - 0.08 * w, xmin, xmax), _clamp(mx + 0.08 * w, xmin, xmax))
        zy_def = (_clamp(my - 0.08 * h, ymin, ymax), _clamp(my + 0.08 * h, ymin, ymax))
        zx = st.slider('Zone x range', float(xmin), float(xmax), zx_def, step=0.05,
                       disabled=not zone_on, key=f'land_zx_{key}')
        zy = st.slider('Zone y range', float(ymin), float(ymax), zy_def, step=0.05,
                       disabled=not zone_on, key=f'land_zy_{key}')
        kick = st.slider('Kick size', 0.05, 2.0, 0.5, step=0.05,
                         disabled=not zone_on, key='land_kick')

    hazards, zone = {}, None
    if wind and sigma > 0:
        hazards['noise_sigma'] = float(sigma)
    if zone_on:
        zone = (float(zx[0]), float(zx[1]), float(zy[0]), float(zy[1]))
        hazards['zone'] = zone
        hazards['kick'] = float(kick)
    return hazards, zone, seed


def _hazard_summary(res):
    bits = []
    if res.get('wind'):
        bits.append(f"wind strength {res['wind']}")
    if res.get('zone'):
        bits.append('danger zone')
    return ', '.join(bits) if bits else 'none'


def render():
    st.subheader('Landscape Track')
    key = st.selectbox('Terrain', list(TERRAINS.keys()),
                       format_func=lambda k: TERRAINS[k].name, key='land_terrain')
    terrain = TERRAINS[key]

    racers = render_garage('land', terrain.default_lr, terrain.adaptive_lr)   # sidebar

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

    hazards, zone, seed = _hazard_controls(terrain, key, start)

    d1, d2, d3 = st.columns([2, 2, 2])
    max_steps = d1.slider('Max steps', 50, 1000, 300, step=50, key='land_steps')
    speed = d2.slider('Playback speed', 1, 5, 2, key='land_speed',
                      help='Higher = fewer frames, faster playback (2D view only).')
    view = d3.radio('View', ['2D contour', '3D surface'], horizontal=True, key='land_view')

    if st.button('Run race', type='primary', key='land_run'):
        try:
            logs = run_landscape_race(racers, terrain, start, max_steps=max_steps,
                                      hazards=hazards or None, seed=seed)
        except ValueError as e:
            st.error(f'Cannot start the race: {e}')
        else:
            st.session_state['land_result'] = dict(
                terrain_key=key, start=start, logs=logs, seed=seed,
                colors={r.name: r.color for r in racers},
                zone=zone, wind=hazards.get('noise_sigma'),
                commentary=generate_commentary(logs, terrain),
                explain=[(l.name, explain_racer(l, terrain)) for l in logs])

    res = st.session_state.get('land_result')
    if res is None:
        st.info('Build your racers in the sidebar, pick a start, and press Run race.')
        return

    t = TERRAINS[res['terrain_key']]
    st.markdown(f"**Showing:** {t.name}, start {res['start']}, seed {res['seed']}, "
                f"hazards: {_hazard_summary(res)}.")
    if view == '2D contour':
        st.caption('Press Play under the map.')
        fig = animated_race_figure(t, res['logs'], res['colors'], stride=speed,
                                   frame_ms=50, zone=res['zone'])
    else:
        fig = surface_figure(t, res['logs'], res['colors'])
        if res['zone']:
            st.caption('The 3D view shows paths only; the danger zone is drawn on the 2D map.')
    st.plotly_chart(fig, width='stretch')
    st.caption(t.lesson)

    st.markdown('**Commentary**')
    with st.container(height=240, border=True):
        if not res['commentary']:
            st.write('A quiet race: nothing notable happened.')
        for step, _, text in res['commentary']:
            st.markdown(f'**step {step}** - {text}')

    st.markdown('**Results**')
    st.dataframe(_table(rank_landscape(res['logs'])), width='stretch', hide_index=True)

    st.markdown('**Explain the race**')
    for name, text in res['explain']:
        with st.expander(name):
            st.write(text)