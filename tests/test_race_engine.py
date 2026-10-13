# path: tests/test_race_engine.py
import math
import pytest
import torch
import torch.nn as nn

from core.data import get_data
from core.race_engine import (run_model_race, run_to_end, rank_model, best_accuracy)
from core.schemas import RaceConfig, RacerConfig, RacerLog


@pytest.fixture(scope='module')
def small():
    try:
        return get_data(subset_size=300, test_size=200, seed=1)
    except Exception as e:
        pytest.skip(f'FashionMNIST could not be loaded: {e}')


def cfg(name, opt, lr, **kw):
    return RacerConfig(name=name, color='#000000', optimizer=opt, lr=lr, **kw)


def rc(**kw):
    base = dict(mode='model', seed=1, max_steps=20, batch_size=32,
                target_acc=1.01, eval_every=5)
    base.update(kw)
    return RaceConfig(**base)


def run(racers, race_cfg, small, **kw):
    return run_to_end(racers, race_cfg, data=small, device='cpu', **kw)


def test_losses_finite_and_adam_improves(small):
    logs = run([cfg('Adam', 'Adam', 0.005), cfg('Mom', 'Momentum', 0.02)], rc(), small)
    for l in logs:
        assert len(l.loss) == 20 and all(math.isfinite(v) for v in l.loss)
    adam = logs[0].loss
    assert sum(adam[:3]) / 3 > sum(adam[-3:]) / 3


def test_same_start_and_same_batches(small):
    # lr 0 means weights never change, so equal loss lists prove that the starting
    # weights and the batches are identical for every racer
    logs = run([cfg('A', 'SGD', 0.0), cfg('B', 'Adam', 0.0)], rc(), small)
    assert logs[0].loss == logs[1].loss
    assert logs[0].acc == logs[1].acc


def test_huge_rate_crashes_or_does_not_improve(small):
    log = run([cfg('Wild', 'SGD', 100.0)], rc(), small)[0]
    assert log.status == 'crashed' or log.loss[-1] >= log.loss[0]


def test_same_seed_gives_identical_losses(small):
    a = run([cfg('Adam', 'Adam', 0.005)], rc(seed=3), small)[0]
    b = run([cfg('Adam', 'Adam', 0.005)], rc(seed=3), small)[0]
    assert a.loss == b.loss


def test_different_seed_gives_different_losses(small):
    a = run([cfg('Adam', 'Adam', 0.005)], rc(seed=3), small)[0]
    b = run([cfg('Adam', 'Adam', 0.005)], rc(seed=4), small)[0]
    assert a.loss != b.loss


def test_finish_stops_racers_and_ends_race_early(small):
    seen = []
    last = None
    for step, last in run_model_race([cfg('A', 'Adam', 0.005), cfg('B', 'SGD', 0.01)],
                                     rc(target_acc=0.0), data=small, device='cpu'):
        seen.append(step)
    assert seen[-1] == 5
    for l in last:
        assert l.status == 'finished' and l.finish_step == 5 and len(l.steps) == 5


def test_timeout_when_target_is_unreachable(small):
    log = run([cfg('A', 'Adam', 0.005)], rc(), small)[0]
    assert log.status == 'timeout' and log.finish_step is None and log.steps[-1] == 20


def test_accuracy_list_is_aligned_and_includes_final_step(small):
    log = run([cfg('A', 'Adam', 0.005)], rc(max_steps=22), small)[0]
    assert len(log.acc) == len(log.steps)
    evaluated = [s for s, a in zip(log.steps, log.acc) if a is not None]
    assert evaluated == [5, 10, 15, 20, 22]
    assert best_accuracy(log) is not None


def test_param_count_is_logged(small):
    log = run([cfg('A', 'Adam', 0.005)], rc(max_steps=5), small)[0]
    assert log.n_params == 20490


def test_no_racers_raises(small):
    with pytest.raises(ValueError):
        run_model_race([], rc(), data=small, device='cpu')


def test_nesterov_without_momentum_raises_at_call_time(small):
    with pytest.raises(ValueError):
        run_model_race([cfg('N', 'Nesterov', 0.01, momentum=0.0)], rc(),
                       data=small, device='cpu')


def test_batch_larger_than_subset_raises(small):
    with pytest.raises(ValueError):
        run_model_race([cfg('A', 'Adam', 0.005)], rc(batch_size=1000),
                       data=small, device='cpu')


def hand(name, status, finish=None, accs=()):
    steps = list(range(1, len(accs) + 1)) or [1]
    return RacerLog(name=name, steps=steps, loss=[1.0] * len(steps),
                    acc=list(accs) or [None], status=status, finish_step=finish)


def test_rank_model_order():
    logs = [hand('crash', 'crashed'),
            hand('slow', 'finished', finish=80, accs=[0.9]),
            hand('fast', 'finished', finish=20, accs=[0.9]),
            hand('weak', 'timeout', accs=[0.3]),
            hand('mid', 'timeout', accs=[0.6])]
    assert [r['name'] for r in rank_model(logs)] == ['fast', 'slow', 'mid', 'weak', 'crash']


def test_model_factory_override(small):
    factory = lambda c: nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    log = run([cfg('Lin', 'Adam', 0.01)], rc(max_steps=5), small, model_factory=factory)[0]
    assert log.n_params == 7850 and len(log.loss) == 5


def test_runs_on_cuda_when_available(small):
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    log = run_to_end([cfg('A', 'Adam', 0.005)], rc(max_steps=5), data=small, device='cuda')[0]
    assert log.status == 'timeout' and all(math.isfinite(v) for v in log.loss)