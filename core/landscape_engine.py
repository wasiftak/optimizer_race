# path: core/landscape_engine.py
import math
import torch
from core.schemas import RacerLog
from core.optimizers import build_optimizer, build_scheduler


def run_landscape_race(racers, terrain, start, max_steps=300, tol=1e-3, bound=1e3,
                       hazards=None, seed=42):
    """Racers do not interact, so running them one after another is equivalent
    to a simultaneous race; the animation plays them together.

    hazards (optional dict):
      noise_sigma: float, std of gradient noise added to every step ("wind")
      zone: (x0, x1, y0, y1) danger-zone rectangle
      kick: float, size of the random kick applied while inside the zone
    Generators are recreated from `seed` for every racer and always drawn once per
    step, so every racer receives the same noise sequence (fairness rule 5).
    """
    hz = hazards or {}
    sigma = float(hz.get('noise_sigma', 0.0))
    zone = hz.get('zone')
    kick = float(hz.get('kick', 0.0))

    logs = []
    for cfg in racers:
        p = torch.tensor(start, dtype=torch.float32, requires_grad=True)
        opt = build_optimizer(cfg, [p])
        sch = build_scheduler(cfg, opt, max_steps)
        g_noise = torch.Generator().manual_seed(seed)
        g_kick = torch.Generator().manual_seed(seed + 1)
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
            noise = torch.randn(2, generator=g_noise)     # always drawn: same sequence
            kicks = torch.randn(2, generator=g_kick)      # for every racer
            if sigma > 0:
                p.grad.add_(sigma * noise)
            opt.step()
            if sch is not None:
                sch.step()
            if zone is not None and kick > 0:
                x0, x1, y0, y1 = zone
                if x0 <= p[0].item() <= x1 and y0 <= p[1].item() <= y1:
                    with torch.no_grad():
                        p.add_(kick * kicks)
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