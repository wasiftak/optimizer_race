# path: core/terrains.py
import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Terrain:
    key: str
    name: str
    fn: Callable          # p of shape (2,) or (2,H,W) -> loss
    bounds: tuple         # (xmin, xmax, ymin, ymax)
    fmin: float           # lowest loss value (finish line)
    minimum: tuple        # (x*, y*) of one global minimum
    default_start: tuple
    default_lr: float     # for SGD, Momentum, Nesterov
    adaptive_lr: float    # for Adam, AdamW, RMSprop, Adagrad (they need other rates)
    lesson: str


def bowl(p):
    return p[0] ** 2 + p[1] ** 2


def canyon(p):
    return p[0] ** 2 + 10 * p[1] ** 2


def banana(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def beale(p):
    x, y = p[0], p[1]
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)          # min (3, 0.5)


def himmelblau(p):
    x, y = p[0], p[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2   # four minima, e.g. (3, 2)


def rastrigin(p):
    x, y = p[0], p[1]
    return (20 + x ** 2 + y ** 2
            - 10 * (torch.cos(2 * math.pi * x) + torch.cos(2 * math.pi * y)))


def saddle(p):
    x, y = p[0], p[1]
    return x ** 2 - y ** 2 + 0.25 * y ** 4    # saddle at (0,0); minima (0, +-1.4142), fmin -1


TERRAINS = {
    'bowl': Terrain(
        'bowl', 'Simple bowl', bowl, (-3, 3, -3, 3), 0.0, (0, 0),
        (-2.5, 2.0), 0.1, 0.1,
        'Everyone finishes; only speed differs.'),
    'canyon': Terrain(
        'canyon', 'Narrow canyon', canyon, (-3, 3, -1.5, 1.5), 0.0, (0, 0),
        (-2.5, 1.2), 0.08, 0.08,
        'Steep walls make plain SGD zigzag.'),
    'banana': Terrain(
        'banana', 'Banana valley', banana, (-2, 2, -1, 3), 0.0, (1, 1),
        (-1.5, 2.0), 0.002, 0.1,
        'A long curved valley; slow for plain SGD. Adaptive optimizers need a bigger rate.'),
    'beale': Terrain(
        'beale', 'Flat then steep (Beale)', beale, (-4.5, 4.5, -4.5, 4.5), 0.0, (3, 0.5),
        (0.0, 0.0), 0.005, 0.1,
        'Flat regions, then a steep valley. Plain SGD crawls; momentum builds speed.'),
    'himmelblau': Terrain(
        'himmelblau', 'Four traps (Himmelblau)', himmelblau, (-5, 5, -5, 5), 0.0, (3, 2),
        (-0.3, 0.3), 0.01, 0.05,
        'Four equally good minima. The start point and the optimizer decide which one you reach.'),
    'rastrigin': Terrain(
        'rastrigin', 'Egg carton (Rastrigin)', rastrigin, (-5, 5, -5, 5), 0.0, (0, 0),
        (3.3, 3.3), 0.003, 0.1,
        'Many local minima. Most racers get trapped in the first dip they find.'),
    'saddle': Terrain(
        'saddle', 'Saddle', saddle, (-2, 2, -2.5, 2.5), -1.0, (0.0, math.sqrt(2)),
        (1.0, 0.01), 0.01, 0.02,
        'A flat-looking ridge that is not a minimum. Plain SGD crawls; momentum leaves.'),
}