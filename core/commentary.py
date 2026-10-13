# path: core/commentary.py
import itertools
import math

import torch

# ---- tunable thresholds -------------------------------------------------
COOLDOWN = 15          # min steps between repeats of the same rule for one racer
OSC_STEPS = 8          # look at the last 8 steps...
OSC_FLIPS = 5          # ...and need at least 5 direction flips
PLATEAU_STEPS = 10     # loss compared with the loss this many steps ago
PLATEAU_REL = 1e-4     # change below this (relative to |loss|+1) counts as flat
DIVERGE_FACTOR = 10.0  # loss gap above this many times the starting gap
TRAP_GRAD = 0.1        # slope length below this means "flat"
TRAP_GAP = 0.05        # loss still this far above the minimum means "not finished"
MOVE_FRAC = 0.002      # moves smaller than this fraction of the map are ignored


def _finite(*vals):
    return all(math.isfinite(v) for v in vals)


def _gap(log, i, terrain):
    """Loss above the minimum at step i (a stopped racer keeps its last value)."""
    v = log.loss[min(i, len(log.loss) - 1)]
    return v - terrain.fmin if math.isfinite(v) else float('inf')


def _finished_by(log, i):
    return log.finish_step is not None and i >= log.finish_step


def flip_steps(log, terrain):
    """Step indices where dx or dy changed sign (tiny moves ignored)."""
    xmin, xmax, ymin, ymax = terrain.bounds
    tx, ty = MOVE_FRAC * (xmax - xmin), MOVE_FRAC * (ymax - ymin)
    out = []
    for i in range(2, len(log.x)):
        x2, x1, x0 = log.x[i], log.x[i - 1], log.x[i - 2]
        y2, y1, y0 = log.y[i], log.y[i - 1], log.y[i - 2]
        if not _finite(x2, x1, x0, y2, y1, y0):
            continue
        dx1, dx0 = x2 - x1, x1 - x0
        dy1, dy0 = y2 - y1, y1 - y0
        fx = dx1 * dx0 < 0 and abs(dx1) > tx and abs(dx0) > tx
        fy = dy1 * dy0 < 0 and abs(dy1) > ty and abs(dy0) > ty
        if fx or fy:
            out.append(i)
    return out


def _grad_norm(terrain, x, y):
    p = torch.tensor([x, y], dtype=torch.float32, requires_grad=True)
    terrain.fn(p).backward()
    return p.grad.norm().item()


# ---- rules: each returns a line of text, or None ------------------------
def rule_oscillation(log, i, flip_set):
    n = sum(1 for j in range(i - (OSC_STEPS - 2), i + 1) if j in flip_set)
    if n >= OSC_FLIPS:
        return f'{log.name} is zigzagging; its learning rate may be too high for this surface'
    return None


def rule_plateau(log, i, terrain):
    if i < PLATEAU_STEPS or _finished_by(log, i):
        return None
    a, b = log.loss[i], log.loss[i - PLATEAU_STEPS]
    if not _finite(a, b) or a <= terrain.fmin + 0.01:
        return None
    if abs(a - b) < PLATEAU_REL * (abs(a) + 1):
        return f'{log.name} is barely moving; this looks like a plateau'
    return None


def rule_divergence(log, i, terrain):
    g0 = max(_gap(log, 0, terrain), 1e-3)
    g = _gap(log, i, terrain)
    if not math.isfinite(g):
        return f'{log.name} has blown up and left the map'
    if g > DIVERGE_FACTOR * g0:
        return f'{log.name} has blown up: its loss is more than 10 times its starting value'
    return None


def rule_finish(log, i):
    if log.status == 'finished' and i == log.finish_step:
        return f'{log.name} reaches the minimum in {log.finish_step} steps'
    return None


def rule_trapped(log, terrain):
    """Timeout, loss still above the minimum, and the slope is almost zero."""
    if log.status != 'timeout':
        return None
    gap = _gap(log, len(log.loss) - 1, terrain)
    if not math.isfinite(gap) or gap <= TRAP_GAP:
        return None
    if _grad_norm(terrain, log.x[-1], log.y[-1]) < TRAP_GRAD:
        return (f'{log.name} is stuck in a local minimum or flat spot: the slope is '
                f'almost zero but the loss is still above the minimum')
    return None


def _overtakes(logs, terrain):
    """(step, winner, loser) whenever two racers swap loss ranking."""
    events = []
    T = max(len(l.loss) for l in logs)
    for i in range(1, T):
        for a, b in itertools.combinations(logs, 2):
            ap, bp = _gap(a, i - 1, terrain), _gap(b, i - 1, terrain)
            ac, bc = _gap(a, i, terrain), _gap(b, i, terrain)
            if not _finite(ap, bp, ac, bc):
                continue
            if ap < bp and ac > bc:
                events.append((i, b.name, a.name))
            elif ap > bp and ac < bc:
                events.append((i, a.name, b.name))
    return events


def generate_commentary(logs, terrain, cooldown=COOLDOWN):
    """Returns a list of (step, racer, text), ordered by step."""
    cands = []   # (step, order, racer, rule, text, repeating)
    for log in logs:
        flip_set = set(flip_steps(log, terrain))
        diverged = False
        for i in range(len(log.loss)):
            t = rule_oscillation(log, i, flip_set)
            if t:
                cands.append((i, 1, log.name, 'osc', t, True))
            t = rule_plateau(log, i, terrain)
            if t:
                cands.append((i, 2, log.name, 'plateau', t, True))
            if not diverged:
                t = rule_divergence(log, i, terrain)
                if t:
                    diverged = True
                    cands.append((i, 0, log.name, 'diverge', t, False))
            t = rule_finish(log, i)
            if t:
                cands.append((i, 3, log.name, 'finish', t, False))
        t = rule_trapped(log, terrain)
        if t:
            cands.append((len(log.loss) - 1, 4, log.name, 'trapped', t, False))
    if len(logs) > 1:
        for step, winner, loser in _overtakes(logs, terrain):
            cands.append((step, 5, winner, 'overtake', f'{winner} just overtook {loser}', True))

    cands.sort(key=lambda c: (c[0], c[1]))
    last, lines = {}, []
    for step, _, racer, rule, text, repeating in cands:
        if repeating:
            key = (racer, rule)
            if key in last and step - last[key] < cooldown:
                continue
            last[key] = step
        lines.append((step, racer, text))
    return lines


# ---- Explain panel: built from counts, not free text ---------------------
def racer_stats(log, terrain):
    plateau = 0
    for i in range(1, len(log.loss)):
        a, b = log.loss[i], log.loss[i - 1]
        if (_finite(a, b) and not _finished_by(log, i) and a > terrain.fmin + 0.01
                and abs(a - b) < PLATEAU_REL * (abs(a) + 1)):
            plateau += 1
    return dict(name=log.name, status=log.status, steps=log.steps[-1],
                final_loss=log.loss[-1], flips=len(flip_steps(log, terrain)),
                plateau_steps=plateau)


def explain_racer(log, terrain):
    s = racer_stats(log, terrain)
    n = log.name
    if log.status == 'finished':
        head = f'{n} finished in {log.finish_step} steps'
    elif log.status == 'crashed':
        head = f'{n} crashed after {s["steps"]} steps'
    else:
        head = f'{n} ran out of steps with loss {s["final_loss"]:.4g}'
    parts = []
    if s['flips'] >= 10:
        parts.append(f'it flipped direction {s["flips"]} times, which means its '
                     f'learning rate is high for this surface')
    elif s['flips'] >= 4:
        parts.append(f'it changed direction {s["flips"]} times, so it wobbled a little')
    else:
        parts.append('it moved smoothly with almost no direction changes')
    if s['plateau_steps'] >= 20:
        parts.append(f'it spent {s["plateau_steps"]} steps on a flat stretch')
    if log.status == 'crashed':
        parts.append('its steps were too large, so the loss blew up; try a smaller learning rate')
    if rule_trapped(log, terrain):
        parts.append('it ended trapped in a local minimum or flat spot')
    return head + '; ' + '; '.join(parts) + '.'