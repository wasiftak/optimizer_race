# path: tests/test_models.py
import copy
import torch

from core.config import DEVICE
from core.models import SmallCNN, build_model, count_params
from core.schemas import RacerConfig


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
    cfg = RacerConfig(name='a', color='#000000', optimizer='SGD')
    m = build_model(cfg, (1, 28, 28), 10)
    assert isinstance(m, SmallCNN) and count_params(m) == 20490