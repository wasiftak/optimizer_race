# path: tests/test_phase6_ui.py
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.preview import sample_images
from core.registry_datasets import DATASET_ORDER, get_spec

APP = str(Path(__file__).resolve().parent.parent / 'app.py')


def load_preview(key, n=8, seed=0):
    try:
        return sample_images(key, n, seed)
    except (OSError, RuntimeError) as e:
        pytest.skip(f'{key} could not be loaded: {e}')


@pytest.mark.parametrize('key', DATASET_ORDER)
def test_sample_images_shapes_and_names(key):
    spec = get_spec(key)
    imgs = load_preview(key)
    assert len(imgs) == 8
    for img, name in imgs:
        assert name in spec.class_names
        expected = spec.in_shape[1:] if spec.in_shape[0] == 1 else (*spec.in_shape[1:], 3)
        assert img.shape == tuple(expected) and img.dtype.name == 'uint8'


def test_sample_images_are_deterministic():
    a, b = load_preview('mnist', 8, 0), load_preview('mnist', 8, 0)
    assert [n for _, n in a] == [n for _, n in b]
    assert all((x == y).all() for (x, _), (y, _) in zip(a, b))


def fresh_model_page():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key='track').set_value('Real Model Track').run()
    assert not at.exception, at.exception
    return at


def test_dataset_dropdown_lists_all_datasets():
    at = fresh_model_page()
    assert list(at.selectbox(key='model_dataset').options) == DATASET_ORDER
    assert at.selectbox(key='model_dataset').value == 'fashion_mnist'


def test_target_slider_follows_the_dataset():
    at = fresh_model_page()
    assert at.slider(key='model_target_fashion_mnist').value == pytest.approx(
        get_spec('fashion_mnist').default_target)
    at.selectbox(key='model_dataset').set_value('mnist').run()
    assert not at.exception, at.exception
    assert at.slider(key='model_target_mnist').value == pytest.approx(
        get_spec('mnist').default_target)


def test_race_on_another_dataset_through_the_ui(tmp_path, monkeypatch):
    import core.storage as storage
    monkeypatch.setattr(storage, 'CACHE_DIR', tmp_path)       # keep the real cache clean
    at = fresh_model_page()
    at.selectbox(key='model_dataset').set_value('mnist').run()
    at.slider(key='model_steps').set_value(50).run()
    at.button(key='model_run').click().run()
    assert not at.exception, at.exception
    res = at.session_state['model_result']
    assert res['rc'].dataset == 'mnist'
    assert len(res['logs']) == 3 and all(l.n_params == 20490 for l in res['logs'])