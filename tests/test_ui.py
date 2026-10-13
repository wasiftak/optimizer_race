# path: tests/test_ui.py
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / 'app.py')


def fresh():
    return AppTest.from_file(APP, default_timeout=180).run()


def test_landscape_page_loads():
    at = fresh()
    assert not at.exception, at.exception
    assert at.radio(key='track').value == 'Landscape Track'


def test_landscape_race_runs_and_3d_view_works():
    at = fresh()
    at.button(key='land_run').click().run()
    assert not at.exception, at.exception
    assert len(at.session_state['land_result']['logs']) == 3
    at.radio(key='land_view').set_value('3D surface').run()
    assert not at.exception, at.exception


def test_model_page_loads():
    at = fresh()
    at.radio(key='track').set_value('Real Model Track').run()
    assert not at.exception, at.exception


def test_model_race_runs_and_saves(tmp_path, monkeypatch):
    import core.storage as storage
    monkeypatch.setattr(storage, 'CACHE_DIR', tmp_path)       # keep the real cache clean
    at = fresh()
    at.radio(key='track').set_value('Real Model Track').run()
    at.slider(key='model_steps').set_value(50).run()
    at.button(key='model_run').click().run()
    assert not at.exception, at.exception
    res = at.session_state['model_result']
    assert len(res['logs']) == 3 and all(len(l.loss) > 0 for l in res['logs'])
    assert len(list(tmp_path.glob('run_*.json'))) == 1