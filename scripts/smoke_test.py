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
        assert abs(v - t.fmin) < 1e-6, key


@check
def all_optimizers_build():
    import torch
    from core.optimizers import build_optimizer, OPTIMIZER_NAMES
    for n in OPTIMIZER_NAMES:
        build_optimizer(_cfg(n, 0.01), [torch.zeros(2, requires_grad=True)])


@check
def engine_canyon_sgd_finishes_at_default_rate():
    from core.terrains import TERRAINS
    from core.landscape_engine import run_landscape_race
    t = TERRAINS['canyon']
    log = run_landscape_race([_cfg('SGD', t.default_lr)], t, t.default_start)[0]
    assert log.status == 'finished', log.status
    assert log.finish_step is not None


@check
def engine_canyon_sgd_crashes_at_high_rate():
    from core.terrains import TERRAINS
    from core.landscape_engine import run_landscape_race
    t = TERRAINS['canyon']
    log = run_landscape_race([_cfg('SGD', 0.11)], t, t.default_start)[0]
    assert log.status == 'crashed', log.status


@check
def engine_is_deterministic():
    from core.terrains import TERRAINS
    from core.landscape_engine import run_landscape_race
    t = TERRAINS['banana']
    a = run_landscape_race([_cfg('Adam', 0.05)], t, t.default_start, max_steps=100)[0]
    b = run_landscape_race([_cfg('Adam', 0.05)], t, t.default_start, max_steps=100)[0]
    assert a.loss == b.loss and a.x == b.x


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