# path: scripts/smoke_test.py
import sys, traceback
sys.path.insert(0, '.')

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _cfg(name, lr, **kw):
    from core.schemas import RacerConfig
    return RacerConfig(name=name, color='#000000', optimizer=name, lr=lr, **kw)


def _race(terrain_key, opt, lr=None, steps=300, **kw):
    from core.terrains import TERRAINS
    from core.landscape_engine import run_landscape_race
    t = TERRAINS[terrain_key]
    lr = t.default_lr if lr is None else lr
    return run_landscape_race([_cfg(opt, lr)], t, t.default_start, max_steps=steps, **kw)[0]


@check
def environment_imports():
    import torch, torchvision, plotly, streamlit, numpy, pandas  # noqa: F401


@check
def config_loads():
    from core import config
    assert config.CACHE_DIR.exists()
    assert config.DATA_DIR.exists()


@check
def terrains_minimum_is_fmin():
    import torch
    from core.terrains import TERRAINS
    for key, t in TERRAINS.items():
        v = t.fn(torch.tensor(t.minimum, dtype=torch.float32)).item()
        assert abs(v - t.fmin) < 1e-4, key


@check
def all_optimizers_build():
    import torch
    from core.optimizers import build_optimizer, OPTIMIZER_NAMES
    for n in OPTIMIZER_NAMES:
        build_optimizer(_cfg(n, 0.01), [torch.zeros(2, requires_grad=True)])


@check
def engine_canyon_sgd_finishes_at_default_rate():
    log = _race('canyon', 'SGD')
    assert log.status == 'finished', log.status
    assert log.finish_step is not None


@check
def engine_canyon_sgd_crashes_at_high_rate():
    log = _race('canyon', 'SGD', lr=0.11)
    assert log.status == 'crashed', log.status


@check
def engine_is_deterministic():
    a = _race('banana', 'Adam', lr=0.05, steps=100)
    b = _race('banana', 'Adam', lr=0.05, steps=100)
    assert a.loss == b.loss and a.x == b.x


@check
def seven_terrains_registered():
    from core.terrains import TERRAINS
    assert len(TERRAINS) == 7, list(TERRAINS)


@check
def himmelblau_sgd_finishes():
    log = _race('himmelblau', 'SGD')
    assert log.status == 'finished', log.status


@check
def beale_momentum_beats_sgd():
    mom = _race('beale', 'Momentum')
    sgd = _race('beale', 'SGD')
    assert mom.status == 'finished', mom.status
    assert sgd.status != 'finished' or sgd.finish_step > mom.finish_step


@check
def saddle_momentum_beats_sgd():
    mom = _race('saddle', 'Momentum')
    sgd = _race('saddle', 'SGD')
    assert mom.status == 'finished', mom.status
    assert sgd.status != 'finished' or sgd.finish_step > mom.finish_step


@check
def rastrigin_sgd_is_trapped():
    from core.terrains import TERRAINS
    t = TERRAINS['rastrigin']
    log = _race('rastrigin', 'SGD')
    assert log.status == 'timeout', log.status
    assert log.loss[-1] > t.fmin + 1e-3


@check
def hazards_are_reproducible():
    hz = {'noise_sigma': 0.3, 'zone': (-1, 1, -1, 1), 'kick': 0.2}
    a = _race('bowl', 'SGD', steps=60, hazards=hz, seed=5)
    b = _race('bowl', 'SGD', steps=60, hazards=hz, seed=5)
    assert a.x == b.x and a.loss == b.loss


if __name__ == '__main__':
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print('PASS', fn.__name__)
        except Exception:
            failed += 1
            print('FAIL', fn.__name__)
            traceback.print_exc()
    sys.exit(1 if failed else 0)