# path: tests/test_phase5_ui.py
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.registry_models import MODEL_ORDER
from core.schemas import RacerConfig, RacerLog
from core.tables import badge_text, model_results_rows
from core.viz import params_figure

APP = str(Path(__file__).resolve().parent.parent / 'app.py')


def racer(name, model='default'):
    return RacerConfig(name=name, color='#000000', optimizer='Adam', model=model)


def test_badge_text_same_and_different():
    assert badge_text([racer('A'), racer('B')]) == 'Same model: pure optimizer test'
    assert badge_text([racer('A'), racer('B', 'mini_vgg')]).startswith('Different models')


def test_results_rows_show_model_label():
    log = RacerLog(name='A', steps=[1], loss=[1.0], acc=[0.5], n_params=9770)
    rows = model_results_rows([log], [racer('A', 'tiny_resnet')])
    assert rows[0]['Model'] == 'Tiny ResNet' and rows[0]['Params'] == 9770


def test_params_figure_bars():
    logs = [RacerLog(name='A', n_params=20490), RacerLog(name='B', n_params=5762)]
    fig = params_figure(logs, {'A': '#e63946', 'B': '#1d7bd8'})
    assert list(fig.data[0].x) == ['A', 'B'] and list(fig.data[0].y) == [20490, 5762]


def fresh_model_page():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key='track').set_value('Real Model Track').run()
    assert not at.exception, at.exception
    return at


def test_same_model_checkbox_hides_other_racers_dropdowns():
    at = fresh_model_page()
    assert at.checkbox(key='model_same').value is True
    keys = [s.key for s in at.selectbox]
    assert 'model_0_model' in keys and 'model_1_model' not in keys
    at.checkbox(key='model_same').set_value(False).run()
    assert not at.exception, at.exception
    keys = [s.key for s in at.selectbox]
    assert all(f'model_{i}_model' in keys for i in range(3))


def _pick(selectbox, model_key):
    idx = MODEL_ORDER.index(model_key)
    if hasattr(selectbox, 'select_index'):
        selectbox.select_index(idx)
    else:
        selectbox.set_value(model_key)


def test_mixed_model_race_through_the_ui(tmp_path, monkeypatch):
    import core.storage as storage
    monkeypatch.setattr(storage, 'CACHE_DIR', tmp_path)       # keep the real cache clean
    at = fresh_model_page()
    at.checkbox(key='model_same').set_value(False).run()
    _pick(at.selectbox(key='model_1_model'), 'mini_vgg')
    at.slider(key='model_steps').set_value(50).run()
    at.button(key='model_run').click().run()
    assert not at.exception, at.exception
    logs = at.session_state['model_result']['logs']
    assert [l.n_params for l in logs] == [20490, 32058, 20490]