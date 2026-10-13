# path: core/tables.py
from core.models import same_model
from core.race_engine import best_accuracy, rank_model
from core.registry_models import MODEL_PRESETS


def badge_text(racers):
    """The fairness line shown above the lanes. It is part of the teaching."""
    if same_model(racers):
        return 'Same model: pure optimizer test'
    return 'Different models: optimizer and architecture both matter'


def model_results_rows(logs, racers):
    """Ranked rows for the results table. Works on live (partial), finished and replayed logs."""
    cfg = {r.name: r for r in racers}
    by_name = {l.name: l for l in logs}
    rows = []
    for r in rank_model(logs):
        l = by_name[r['name']]
        c = cfg.get(l.name)
        best = best_accuracy(l)
        model = MODEL_PRESETS.get(c.model, {}).get('label', c.model) if c else '?'
        rows.append({
            'Rank': r['rank'],
            'Racer': l.name,
            'Optimizer': c.optimizer if c else '?',
            'LR': f'{c.lr:g}' if c else '?',
            'Model': model,
            'Params': l.n_params if l.n_params is not None else 0,
            'Status': l.status,
            'Finish step': str(l.finish_step) if l.finish_step is not None else '-',
            'Best acc': f'{best:.1%}' if best is not None else '-',
            'Final loss': f'{l.loss[-1]:.4f}' if l.loss else '-',
            'Seconds': round(l.seconds, 1),
        })
    return rows