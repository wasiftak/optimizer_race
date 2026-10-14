# path: tests/test_terrain_about.py
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.terrains import TERRAINS

APP = str(Path(__file__).resolve().parent.parent / 'app.py')


def test_every_terrain_has_about_fields():
    for key, t in TERRAINS.items():
        for field in ('equation', 'about', 'minima_text', 'try_this'):
            assert getattr(t, field).strip(), (key, field)


def test_equations_are_well_formed_latex():
    for key, t in TERRAINS.items():
        assert t.equation.startswith('f(x, y) ='), key
        assert t.equation.count('{') == t.equation.count('}'), key
        assert t.equation.count('(') == t.equation.count(')'), key


def test_about_expander_for_every_terrain():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    for key, t in TERRAINS.items():
        at.selectbox(key='land_terrain').set_value(key).run()
        assert not at.exception, (key, at.exception)
        labels = [e.label for e in at.expander]
        assert f'About {t.name}' in labels, (key, labels)