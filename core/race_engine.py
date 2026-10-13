# path: core/race_engine.py
import copy
import time

import torch
import torch.nn.functional as F

from core.config import DEVICE
from core.data import get_data
from core.models import build_model, count_params, same_model
from core.optimizers import build_optimizer, build_scheduler
from core.schemas import RacerLog


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def best_accuracy(log):
    vals = [a for a in log.acc if a is not None]
    return max(vals) if vals else None


def _validate(racers, race_cfg, n_train):
    if not racers:
        raise ValueError('Add at least one racer')
    if race_cfg.batch_size < 1 or race_cfg.batch_size > n_train:
        raise ValueError(f'Batch size {race_cfg.batch_size} must be between 1 and the '
                         f'number of training images ({n_train})')
    if race_cfg.max_steps < 1:
        raise ValueError('Max steps must be at least 1')
    if race_cfg.eval_every < 1:
        raise ValueError('eval_every must be at least 1')


def run_model_race(racers, race_cfg, data=None, model_factory=None, device=None):
    """Lockstep training. Setup runs now (errors raise here); the returned generator
    yields (step, logs) after every step. The yielded logs are live objects that keep
    changing, so copy them if you need a snapshot.

    Fairness: one shared seed and one shared batch order. When every racer uses the same
    model, one base model is deep-copied so the starting weights are identical. When the
    models differ, the seed is reset before each model is built, so every architecture
    starts from reproducible weights (racers with the same architecture still start
    identical)."""
    device = device or DEVICE
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(race_cfg.seed)

    if data is None:
        data = get_data(race_cfg.subset_size, 1000, race_cfg.seed)
    xtr, ytr = data['xtr'].to(device), data['ytr'].to(device)
    xte, yte = data['xte'].to(device), data['yte'].to(device)
    _validate(racers, race_cfg, len(xtr))

    make = model_factory or (lambda cfg: build_model(cfg, data['in_shape'], data['n_classes']))
    shared = model_factory is not None or same_model(racers)
    base = make(racers[0]) if shared else None

    states = []
    for cfg in racers:
        if shared:
            m = copy.deepcopy(base).to(device)         # identical start for every racer
        else:
            torch.manual_seed(race_cfg.seed)           # reproducible start per architecture
            m = make(cfg).to(device)
        opt = build_optimizer(cfg, m.parameters())     # may raise ValueError: caught by the UI
        sch = build_scheduler(cfg, opt, race_cfg.max_steps)
        states.append(dict(cfg=cfg, m=m, opt=opt, sch=sch, alive=True,
                           log=RacerLog(name=cfg.name, n_params=count_params(m))))
    return _lockstep(states, race_cfg, xtr, ytr, xte, yte, device)


def _lockstep(states, race_cfg, xtr, ytr, xte, yte, device):
    g = torch.Generator().manual_seed(race_cfg.seed)
    n, bs = len(xtr), race_cfg.batch_size
    perm, pos = torch.randperm(n, generator=g), 0

    for step in range(1, race_cfg.max_steps + 1):
        if pos + bs > n:                                  # new epoch
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs].to(device)
        pos += bs
        xb, yb = xtr[idx], ytr[idx]                       # same batch for every racer

        for s in states:
            if not s['alive']:
                continue
            log = s['log']
            t0 = time.perf_counter()
            s['m'].train()
            s['opt'].zero_grad()
            loss = F.cross_entropy(s['m'](xb), yb)
            if not torch.isfinite(loss):
                log.status = 'crashed'
                s['alive'] = False
                continue
            loss.backward()
            s['opt'].step()
            if s['sch'] is not None:
                s['sch'].step()
            log.steps.append(step)
            log.loss.append(loss.item())
            acc = None
            if step % race_cfg.eval_every == 0 or step == race_cfg.max_steps:
                acc = evaluate(s['m'], xte, yte)
                if acc >= race_cfg.target_acc and log.finish_step is None:
                    log.status, log.finish_step, s['alive'] = 'finished', step, False
            log.acc.append(acc)
            _sync(device)
            log.seconds += time.perf_counter() - t0

        if step == race_cfg.max_steps:
            for s in states:
                if s['alive']:
                    s['log'].status = 'timeout'
                    s['alive'] = False
        yield step, [s['log'] for s in states]
        if not any(s['alive'] for s in states):
            break


def run_to_end(racers, race_cfg, **kw):
    """Run a whole race without drawing; returns the final logs."""
    logs = None
    for _, logs in run_model_race(racers, race_cfg, **kw):
        pass
    return logs


def _rank_key(log):
    if log.status == 'crashed':
        return (2, 0.0)
    if log.status == 'finished':
        return (0, float(log.finish_step))
    b = best_accuracy(log)
    return (1, -(b if b is not None else -1.0))


def rank_model(logs):
    """Finished racers by finish step, then the rest by best accuracy, crashed last."""
    ordered = sorted(logs, key=_rank_key)
    return [dict(rank=i + 1, name=l.name, status=l.status,
                 steps=l.steps[-1] if l.steps else 0,
                 best_acc=best_accuracy(l),
                 final_loss=l.loss[-1] if l.loss else None)
            for i, l in enumerate(ordered)]