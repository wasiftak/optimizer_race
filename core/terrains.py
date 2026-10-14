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
    equation: str = ''    # LaTeX, shown in the About panel
    about: str = ''       # plain-words description
    minima_text: str = '' # where the finish line(s) are
    try_this: str = ''    # an experiment to run


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
        'Everyone finishes; only speed differs.',
        equation=r'f(x, y) = x^2 + y^2',
        about=('A perfectly round bowl. The slope is the same in every direction, so every '
               'optimizer heads straight for the bottom. It is the baseline: any difference '
               'between racers here comes only from how each one sizes its steps.'),
        minima_text='One minimum at (0, 0), where the loss is 0.',
        try_this=('Run it first. Everyone finishes, so compare the step counts. Notice that '
                  'Momentum shoots past the minimum and swings back.')),
    'canyon': Terrain(
        'canyon', 'Narrow canyon', canyon, (-3, 3, -1.5, 1.5), 0.0, (0, 0),
        (-2.5, 1.2), 0.08, 0.08,
        'Steep walls make plain SGD zigzag.',
        equation=r'f(x, y) = x^2 + 10\,y^2',
        about=('A long, narrow valley. The walls are 10 times steeper across the canyon (y) '
               'than along its floor (x). A big step bounces between the walls while barely '
               'moving along the floor, which is how plain SGD wastes steps.'),
        minima_text='One minimum at (0, 0), where the loss is 0.',
        try_this=("Raise SGD's learning rate to 0.11 and run: it crashes, because each y step "
                  'overshoots by more than the last. Compare how Momentum and Adam cross the '
                  'canyon.')),
    'banana': Terrain(
        'banana', 'Banana valley', banana, (-2, 2, -1, 3), 0.0, (1, 1),
        (-1.5, 2.0), 0.002, 0.1,
        'A long curved valley; slow for plain SGD. Adaptive optimizers need a bigger rate.',
        equation=r'f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2',
        about=('The Rosenbrock function. The floor of a narrow, curved valley (the curve '
               'y = x squared) slopes only gently toward the minimum. Reaching the valley is '
               'easy; following it round the bend is slow, and big steps throw you off it.'),
        minima_text='One minimum at (1, 1), where the loss is 0.',
        try_this=('Plain SGD only works with a tiny rate here (the default is 0.002). Adam and '
                  'RMSprop get a larger default (0.1) and may need more than 300 steps: raise '
                  'Max steps to see how they do.')),
    'beale': Terrain(
        'beale', 'Flat then steep (Beale)', beale, (-4.5, 4.5, -4.5, 4.5), 0.0, (3, 0.5),
        (0.0, 0.0), 0.005, 0.1,
        'Flat regions, then a steep valley. Plain SGD crawls; momentum builds speed.',
        equation=(r'f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2'
                  r' + (2.625 - x + xy^3)^2'),
        about=('Large flat regions with a steep, narrow valley leading to the minimum. In the '
               'flat parts the slope is tiny, so plain gradient steps are tiny too; methods '
               'that build speed or rescale the slope cross them much faster.'),
        minima_text='One minimum at (3, 0.5), where the loss is 0.',
        try_this=('From the default start, compare SGD with Momentum: SGD crawls while '
                  'Momentum builds up speed and arrives much sooner.')),
    'himmelblau': Terrain(
        'himmelblau', 'Four traps (Himmelblau)', himmelblau, (-5, 5, -5, 5), 0.0, (3, 2),
        (-0.3, 0.3), 0.01, 0.05,
        'Four equally good minima. The start point and the optimizer decide which one you reach.',
        equation=r'f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2',
        about=('Four separate valleys of exactly the same depth. Which one a racer ends up in '
               'depends on where it starts and how it moves, so every one of the four '
               'minima counts as a finish line.'),
        minima_text=('Four minima, all with loss 0: about (3, 2), (-2.81, 3.13), '
                     '(-3.78, -3.28) and (3.58, -1.85).'),
        try_this=('Run from the default start, then change the start point and run again. '
                  'Different optimizers can end up in different valleys, and a small change '
                  'of start can switch which one.')),
    'rastrigin': Terrain(
        'rastrigin', 'Egg carton (Rastrigin)', rastrigin, (-5, 5, -5, 5), 0.0, (0, 0),
        (3.3, 3.3), 0.003, 0.1,
        'Many local minima. Most racers get trapped in the first dip they find.',
        equation=r'f(x, y) = 20 + x^2 + y^2 - 10\left(\cos 2\pi x + \cos 2\pi y\right)',
        about=('A bumpy egg-carton surface: a big bowl covered in small dips. Each dip looks '
               'like a minimum to gradient descent, which only sees the local slope, so '
               'racers settle in the first dip they reach.'),
        minima_text=('Global minimum at (0, 0), loss 0; many local minima near every '
                     'whole-number point.'),
        try_this=('Run it with the defaults: racers get trapped in a nearby dip (shown as a '
                  'timeout with loss above 0). Turn on the Wind hazard or raise the learning '
                  'rate and see whether anyone escapes.')),
    'saddle': Terrain(
        'saddle', 'Saddle', saddle, (-2, 2, -2.5, 2.5), -1.0, (0.0, math.sqrt(2)),
        (1.0, 0.01), 0.01, 0.02,
        'A flat-looking ridge that is not a minimum. Plain SGD crawls; momentum leaves.',
        equation=r'f(x, y) = x^2 - y^2 + \tfrac{1}{4}\,y^4',
        about=('The origin looks flat but is a saddle: a minimum along x and a maximum along '
               'y. Near it the slope is almost zero, so plain gradient descent barely moves, '
               'then slides off toward one of the two real minima.'),
        minima_text=('Saddle point at (0, 0), which is not a minimum. Two minima at '
                     '(0, +1.41) and (0, -1.41), where the loss is -1.'),
        try_this=('The default start is almost exactly on the ridge. Watch SGD crawl along it '
                  'while Momentum builds speed and slides off much sooner.')),
}