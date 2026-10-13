# path: core/terrains.py
from dataclasses import dataclass
from typing import Callable


@dataclass
class Terrain:
    key: str
    name: str
    fn: Callable          # p of shape (2,) or (2,H,W) -> loss
    bounds: tuple         # (xmin, xmax, ymin, ymax)
    fmin: float           # lowest loss value (finish line)
    minimum: tuple        # (x*, y*) of the global minimum
    default_start: tuple
    default_lr: float
    lesson: str


def bowl(p):
    return p[0] ** 2 + p[1] ** 2


def canyon(p):
    return p[0] ** 2 + 10 * p[1] ** 2


def banana(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


TERRAINS = {
    'bowl': Terrain('bowl', 'Simple bowl', bowl, (-3, 3, -3, 3), 0.0, (0, 0),
                    (-2.5, 2.0), 0.1,
                    'Everyone finishes; only speed differs.'),
    'canyon': Terrain('canyon', 'Narrow canyon', canyon, (-3, 3, -1.5, 1.5), 0.0, (0, 0),
                      (-2.5, 1.2), 0.08,
                      'Steep walls make plain SGD zigzag.'),
    'banana': Terrain('banana', 'Banana valley', banana, (-2, 2, -1, 3), 0.0, (1, 1),
                      (-1.5, 2.0), 0.002,
                      'A long curved valley; slow for plain SGD.'),
}