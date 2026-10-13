# path: tests/test_models.py
import copy
import pytest
import torch

from core.config import DEVICE
from core.models import (SmallCNN, MODEL_CLASSES, build_model, count_params, same_model,
                         summarize)
from core.registry_models import MODEL_PRESETS
from core.schemas import RacerConfig

EXPECTED = {'default': 20490, 'mini_vgg': 32058, 'tiny_resnet': 9770,
            'wide_shallow': 63562, 'deep_narrow': 5762}
NAMES = list(EXPECTED)


def cfg(model='default', arch=None):
    return RacerConfig(name='a', color='#000000', optimizer='SGD', model=model, arch=arch)


# ---- the original Phase 4 tests ----
def test_param_count_is_20490():
    # conv1 1*16*9+16=160, conv2 16*32*9+32=4640, linear 1568*10+10=15690
    assert count_params(SmallCNN()) == 20490


def test_output_shape_on_device():
    m = SmallCNN().to(DEVICE)
    out = m(torch.zeros(2, 1, 28, 28, device=DEVICE))
    assert out.shape == (2, 10)


def test_deepcopies_have_identical_weights():
    base = SmallCNN()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_adapts_to_colour_input():
    # 3x32x32: conv1 448 + conv2 4640 + linear 2048*10+10 = 25578
    m = SmallCNN(in_shape=(3, 32, 32), n_classes=10)
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert count_params(m) == 25578


def test_build_model_returns_smallcnn():
    m = build_model(cfg(), (1, 28, 28), 10)
    assert isinstance(m, SmallCNN) and count_params(m) == 20490


# ---- Phase 5: the five presets ----
@pytest.mark.parametrize('name', NAMES)
def test_preset_parameter_count(name):
    # Deep and narrow by hand: 80 + 3*584 = 1832 conv, 392*10+10 = 3930 linear -> 5762
    m = build_model(cfg(name), (1, 28, 28), 10)
    assert count_params(m) == EXPECTED[name]
    assert MODEL_PRESETS[name]['params'] == EXPECTED[name]


@pytest.mark.parametrize('name', NAMES)
def test_preset_output_shape(name):
    m = build_model(cfg(name), (1, 28, 28), 10)
    assert m(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize('name', NAMES)
def test_same_seed_gives_identical_initial_weights(name):
    torch.manual_seed(1)
    a = build_model(cfg(name), (1, 28, 28), 10)
    torch.manual_seed(1)
    b = build_model(cfg(name), (1, 28, 28), 10)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_unknown_model_raises_keyerror_listing_valid_names():
    with pytest.raises(KeyError, match='mini_vgg'):
        build_model(cfg('nonsense'), (1, 28, 28), 10)


def test_registry_and_classes_agree():
    assert set(MODEL_PRESETS) == set(MODEL_CLASSES) == set(EXPECTED)


def test_every_preset_adapts_to_colour_images():
    for name in NAMES:
        m = build_model(cfg(name), (3, 32, 32), 10)
        assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10), name


def test_tiny_resnet_gradient_reaches_first_conv():
    m = build_model(cfg('tiny_resnet'), (1, 28, 28), 10)
    m.train()
    m(torch.randn(4, 1, 28, 28)).sum().backward()
    g = m.stem[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0


# ---- summarize and same_model ----
def test_summary_parameters_sum_to_total():
    for name in NAMES:
        m = build_model(cfg(name), (1, 28, 28), 10)
        rows = summarize(m, (1, 28, 28))
        assert sum(r['params'] for r in rows) == count_params(m), name


def test_summary_shapes_for_default_model():
    rows = summarize(SmallCNN(), (1, 28, 28))
    assert rows[0]['output_shape'] == (16, 28, 28) and rows[0]['params'] == 160
    assert rows[-1]['layer'] == 'classifier' and rows[-1]['output_shape'] == (10,)


def test_summary_restores_training_mode():
    m = SmallCNN()
    m.train()
    summarize(m, (1, 28, 28))
    assert m.training
    m.eval()
    summarize(m, (1, 28, 28))
    assert not m.training


def test_same_model_detection():
    assert same_model([cfg('default'), cfg('default')])
    assert same_model([cfg('tiny_resnet')])
    assert not same_model([cfg('default'), cfg('mini_vgg')])
    assert not same_model([cfg('default'), cfg('default', arch={'blocks': []})])