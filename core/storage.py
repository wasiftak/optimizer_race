# path: core/storage.py
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from core.config import CACHE_DIR, DEVICE
from core.schemas import RaceConfig, RacerConfig, RacerLog


def _dir(cache_dir):
    d = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_hash(racers, race_cfg):
    """Short sha1 of the racer configs and the race config."""
    blob = json.dumps([[asdict(r) for r in racers], asdict(race_cfg)],
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def auto_label(racers, race_cfg):
    opts = ', '.join(f'{r.optimizer} {r.lr:g}' for r in racers)
    return f'{opts} | {race_cfg.dataset} | {race_cfg.max_steps} steps'


def save_run(racers, race_cfg, logs, label=None, cache_dir=None):
    """Write cache/run_<hash>.json atomically; returns the path."""
    d = _dir(cache_dir)
    path = d / f'run_{run_hash(racers, race_cfg)}.json'
    payload = dict(
        version=1,
        label=label or auto_label(racers, race_cfg),
        created=datetime.now().isoformat(timespec='seconds'),
        env=dict(torch=torch.__version__, device=DEVICE, python=platform.python_version()),
        racers=[asdict(r) for r in racers],
        race_cfg=asdict(race_cfg),
        logs=[asdict(l) for l in logs])
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(tmp, path)
    return path


def load_run(path):
    """Rebuild the dataclasses. Raises ValueError with a readable message if the file is bad."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        rc = dict(payload['race_cfg'])
        rc['start'] = tuple(rc['start'])
        return dict(
            label=payload['label'], created=payload['created'], env=payload.get('env', {}),
            racers=[RacerConfig(**d) for d in payload['racers']],
            race_cfg=RaceConfig(**rc),
            logs=[RacerLog(**d) for d in payload['logs']])
    except Exception as e:
        raise ValueError(f'Could not read saved run {path.name}: {e}') from e


def list_runs(cache_dir=None):
    """Newest first. Unreadable files are listed with a warning label, never raised."""
    d = _dir(cache_dir)
    files = sorted(d.glob('run_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding='utf-8'))
            label, created = payload['label'], payload['created']
        except Exception:
            label, created = '(unreadable file)', ''
        out.append(dict(file=p.name, path=str(p), label=label, created=created))
    return out


def total_steps(logs):
    return max((l.steps[-1] for l in logs if l.steps), default=0)


def replay_logs(logs, k):
    """Copies of the logs truncated to step k, so a saved run is drawn by exactly the
    same code as a live one. A racer shows its final status once k reaches its last
    logged step; before that it is 'running'."""
    out = []
    for l in logs:
        n = sum(1 for s in l.steps if s <= k)
        finished_by_k = l.finish_step is not None and l.finish_step <= k
        if finished_by_k:
            status = 'finished'
        elif n < len(l.steps) or not l.steps:
            status = 'running'
        else:
            status = l.status
        out.append(RacerLog(
            name=l.name, steps=l.steps[:n], loss=l.loss[:n], acc=l.acc[:n],
            x=l.x[:n], y=l.y[:n], status=status,
            finish_step=l.finish_step if finished_by_k else None,
            n_params=l.n_params, seconds=l.seconds))
    return out