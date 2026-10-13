# path: scripts/smoke_test.py
import sys, traceback
sys.path.insert(0, '.')

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def environment_imports():
    import torch, torchvision, plotly, streamlit, numpy, pandas  # noqa: F401


@check
def config_loads():
    from core import config
    assert config.CACHE_DIR.exists()
    assert config.DATA_DIR.exists()


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