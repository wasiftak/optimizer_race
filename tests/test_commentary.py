# path: tests/test_commentary.py
import math
from core.commentary import generate_commentary, racer_stats, explain_racer
from core.landscape_engine import run_landscape_race
from core.schemas import RacerConfig, RacerLog
from core.terrains import TERRAINS


def mk(name, xs, ys, loss, status='timeout', finish=None):
    return RacerLog(name=name, steps=list(range(len(xs))), loss=list(loss),
                    x=list(xs), y=list(ys), status=status, finish_step=finish)


def zigzag_log(n, name='Z'):
    xs = [0.5 * (-1) ** i for i in range(n)]
    return mk(name, xs, [0.0] * n, [1.0 + 0.1 * (i % 2) for i in range(n)])


def texts(lines, word):
    return [(s, r, t) for s, r, t in lines if word in t]


def test_oscillation_detected():
    t = TERRAINS['canyon']
    lines = generate_commentary([zigzag_log(20)], t)
    hits = texts(lines, 'zigzag')
    assert hits and 'Z' in hits[0][2]


def test_smooth_path_no_oscillation():
    t = TERRAINS['canyon']
    n = 20
    log = mk('S', [0.1 * i for i in range(n)], [0.0] * n, [10 - 0.3 * i for i in range(n)])
    assert not texts(generate_commentary([log], t), 'zigzag')


def test_cooldown_limits_repeats():
    t = TERRAINS['canyon']
    steps = [s for s, _, _ in texts(generate_commentary([zigzag_log(60)], t), 'zigzag')]
    assert 2 <= len(steps) <= 4
    assert all(b - a >= 15 for a, b in zip(steps, steps[1:]))


def test_plateau_detected():
    t = TERRAINS['bowl']
    log = mk('P', [1.0] * 30, [1.0] * 30, [5.0] * 30)
    assert texts(generate_commentary([log], t), 'plateau')


def test_divergence_reported_once():
    t = TERRAINS['bowl']
    n = 5
    log = mk('D', [0.0, 0.1, 0.2, 0.3, 0.4], [0.0] * n, [1, 2, 5, 20, float('inf')],
             status='crashed')
    assert len(texts(generate_commentary([log], t), 'blown up')) == 1


def test_finish_line():
    t = TERRAINS['bowl']
    n = 75
    log = mk('F', [0.01 * i for i in range(n)], [0.0] * n,
             [max(0.0, 10 - 0.2 * i) for i in range(n)], status='finished', finish=74)
    hits = texts(generate_commentary([log], t), '74 steps')
    assert len(hits) == 1 and hits[0][0] == 74


def test_overtake_detected():
    t = TERRAINS['bowl']
    n = 20
    a = mk('A', [0.0] * n, [0.0] * n, [10 - 0.5 * i for i in range(n)])
    b = mk('B', [0.0] * n, [0.0] * n, [5.2] * n)
    lines = generate_commentary([a, b], t)
    assert texts(lines, 'A just overtook B')
    assert not texts(lines, 'B just overtook A')


def _rastrigin_trap_x():
    x = 3.0
    for _ in range(30):                      # Newton step to the local minimum near 3
        g = 2 * x + 20 * math.pi * math.sin(2 * math.pi * x)
        h = 2 + 40 * math.pi ** 2 * math.cos(2 * math.pi * x)
        x -= g / h
    return x


def test_trapped_detected():
    t = TERRAINS['rastrigin']
    x = _rastrigin_trap_x()
    log = mk('T', [x] * 20, [x] * 20, [17.9] * 20)
    assert texts(generate_commentary([log], t), 'stuck')


def test_not_trapped_when_slope_is_large():
    t = TERRAINS['bowl']
    log = mk('N', [1.0] * 20, [1.0] * 20, [2.0] * 20)
    assert not texts(generate_commentary([log], t), 'stuck')


def test_stats_and_explain_for_zigzag():
    t = TERRAINS['canyon']
    log = zigzag_log(30)
    assert racer_stats(log, t)['flips'] >= 10
    text = explain_racer(log, t)
    assert 'flipped' in text and 'high' in text


def test_stats_for_smooth_path():
    t = TERRAINS['canyon']
    n = 20
    log = mk('S', [0.1 * i for i in range(n)], [0.0] * n, [10 - 0.3 * i for i in range(n)])
    assert racer_stats(log, t)['flips'] == 0
    assert 'flipped' not in explain_racer(log, t)


def test_real_race_integration():
    t = TERRAINS['canyon']
    racers = [RacerConfig(name='SGD', color='#000000', optimizer='SGD', lr=0.08),
              RacerConfig(name='Adam', color='#000000', optimizer='Adam', lr=0.08)]
    logs = run_landscape_race(racers, t, t.default_start)
    lines = generate_commentary(logs, t)
    assert any(r == 'SGD' and 'zigzag' in text for _, r, text in lines)
    for log in logs:
        if log.status == 'finished':
            hits = [s for s, r, text in lines if r == log.name and 'reaches the minimum' in text]
            assert hits == [log.finish_step]