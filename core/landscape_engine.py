# path: core/landscape_engine.py
import math
import torch
from core.schemas import RacerLog
from core.optimizers import build_optimizer


def run_landscape_race(racers, terrain, start, max_steps=300, tol=1e-3, bound=1e3):
    """Racers do not interact, so running them one after another is equivalent
    to a simultaneous race; the animation plays them together."""
    logs = []
    for cfg in racers:
        p = torch.tensor(start, dtype=torch.float32, requires_grad=True)
        opt = build_optimizer(cfg, [p])
        log = RacerLog(name=cfg.name)
        for step in range(max_steps + 1):
            loss = terrain.fn(p)
            lv = loss.item()
            log.steps.append(step)
            log.loss.append(lv)
            log.x.append(p[0].item())
            log.y.append(p[1].item())
            if not math.isfinite(lv) or max(abs(log.x[-1]), abs(log.y[-1])) > bound:
                log.status = 'crashed'
                break
            if lv <= terrain.fmin + tol:
                log.status = 'finished'
                log.finish_step = step
                break
            if step == max_steps:
                log.status = 'timeout'
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        logs.append(log)
    return logs


def _rank_key(log):
    if log.status == 'finished':
        step = log.finish_step if log.finish_step is not None else log.steps[-1]
        return (0, step)
    if log.status == 'crashed':
        return (2, 0)
    fl = log.loss[-1]
    return (1, fl if math.isfinite(fl) else float('inf'))


def rank_landscape(logs):
    """Finished racers by finish_step, then the rest by final loss, crashed last."""
    ordered = sorted(logs, key=_rank_key)   # sorted() is stable
    return [dict(rank=i + 1, name=l.name, status=l.status,
                 steps=l.steps[-1], final_loss=l.loss[-1])
            for i, l in enumerate(ordered)]