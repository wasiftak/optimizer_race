# path: tests/test_storage.py
import os
import pytest

from core.schemas import RaceConfig, RacerConfig, RacerLog
from core.storage import (save_run, load_run, list_runs, replay_logs, total_steps, run_hash)


def make_racers(lr=0.01):
    return [RacerConfig(name='SGD', color='#e63946', optimizer='SGD', lr=lr),
            RacerConfig(name='Adam', color='#1d7bd8', optimizer='Adam', lr=0.003)]


def make_cfg():
    return RaceConfig(mode='model', seed=5, max_steps=10, batch_size=32,
                      target_acc=0.8, eval_every=5)


def make_log(name, n=10, status='timeout', finish=None):
    steps = list(range(1, n + 1))
    return RacerLog(name=name, steps=steps, loss=[2.0 - 0.1 * s for s in steps],
                    acc=[0.5 if s % 5 == 0 else None for s in steps],
                    status=status, finish_step=finish, n_params=20490, seconds=1.5)


def test_round_trip_returns_equal_objects(tmp_path):
    racers, cfg = make_racers(), make_cfg()
    logs = [make_log('SGD'), make_log('Adam', n=6, status='finished', finish=6)]
    path = save_run(racers, cfg, logs, cache_dir=tmp_path)
    data = load_run(path)
    assert data['racers'] == racers
    assert data['race_cfg'] == cfg and isinstance(data['race_cfg'].start, tuple)
    assert data['logs'] == logs


def test_hash_depends_on_config():
    cfg = make_cfg()
    assert run_hash(make_racers(0.01), cfg) == run_hash(make_racers(0.01), cfg)
    assert run_hash(make_racers(0.01), cfg) != run_hash(make_racers(0.02), cfg)


def test_list_runs_newest_first(tmp_path):
    cfg = make_cfg()
    p1 = save_run(make_racers(0.01), cfg, [make_log('SGD')], cache_dir=tmp_path)
    p2 = save_run(make_racers(0.02), cfg, [make_log('SGD')], cache_dir=tmp_path)
    os.utime(p1, (1000, 1000))
    runs = list_runs(tmp_path)
    assert len(runs) == 2 and runs[0]['file'] == p2.name


def test_corrupted_file_raises_readable_error(tmp_path):
    bad = tmp_path / 'run_bad.json'
    bad.write_text('{not json', encoding='utf-8')
    with pytest.raises(ValueError, match='run_bad'):
        load_run(bad)


def test_corrupted_file_does_not_break_listing(tmp_path):
    (tmp_path / 'run_bad.json').write_text('{not json', encoding='utf-8')
    runs = list_runs(tmp_path)
    assert len(runs) == 1 and 'unreadable' in runs[0]['label']


def test_replay_truncates_and_leaves_original_untouched():
    log = make_log('SGD')
    out = replay_logs([log], 4)[0]
    assert out.steps == [1, 2, 3, 4] and len(out.loss) == 4 and len(out.acc) == 4
    assert out.status == 'running' and out.finish_step is None
    assert replay_logs([log], 10)[0].status == 'timeout'
    assert len(log.steps) == 10


def test_replay_of_a_finished_racer():
    log = make_log('Adam', n=6, status='finished', finish=6)
    assert replay_logs([log], 3)[0].status == 'running'
    done = replay_logs([log], 6)[0]
    assert done.status == 'finished' and done.finish_step == 6
    assert replay_logs([log], 100)[0].status == 'finished'


def test_total_steps():
    assert total_steps([make_log('A', n=10), make_log('B', n=6)]) == 10
    assert total_steps([]) == 0


def test_no_temp_file_left_after_save(tmp_path):
    save_run(make_racers(), make_cfg(), [make_log('SGD')], cache_dir=tmp_path)
    assert list(tmp_path.glob('*.tmp')) == []


def test_auto_label_names_optimizers(tmp_path):
    path = save_run(make_racers(), make_cfg(), [make_log('SGD')], cache_dir=tmp_path)
    label = load_run(path)['label']
    assert 'SGD 0.01' in label and 'Adam 0.003' in label