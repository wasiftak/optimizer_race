# path: core/landscape_engine.py
import math
import torch
from core.schemas import RacerLog
from core.optimizers import build_optimizer


def run_landscape_race(racers, terrain, start, max_steps=300, tol=1e-3, bound=1e3):
    """Racers do not interact, so running them one after another is equivalent
    to a simultaneous race; the animation (Phase 2) plays them together."""
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