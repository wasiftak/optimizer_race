# path: tests/test_tables.py
from core.schemas import RacerConfig, RacerLog
from core.tables import model_results_rows

RACERS = [RacerConfig(name='A', color='#000000', optimizer='Adam', lr=0.003),
          RacerConfig(name='B', color='#111111', optimizer='SGD', lr=0.03),
          RacerConfig(name='C', color='#222222', optimizer='Momentum', lr=0.03)]


def mk(name, status, acc, finish=None):
    n = len(acc)
    return RacerLog(name=name, steps=list(range(1, n + 1)), loss=[0.5] * n, acc=list(acc),
                    status=status, finish_step=finish, n_params=20490, seconds=1.234)


def test_rows_rank_order_and_columns():
    logs = [mk('B', 'timeout', [None, 0.4]),
            mk('A', 'finished', [None, 0.85], finish=2),
            mk('C', 'crashed', [None])]
    rows = model_results_rows(logs, RACERS)
    assert [r['Racer'] for r in rows] == ['A', 'B', 'C']
    a, b = rows[0], rows[1]
    assert a['Finish step'] == '2' and a['Best acc'] == '85.0%'
    assert a['Optimizer'] == 'Adam' and a['LR'] == '0.003' and a['Params'] == 20490
    assert b['Finish step'] == '-' and b['Best acc'] == '40.0%'
    assert rows[2]['Status'] == 'crashed'


def test_rows_handle_running_racer_without_accuracy():
    rows = model_results_rows([mk('A', 'running', [None, None]), RacerLog(name='B')], RACERS)
    by = {r['Racer']: r for r in rows}
    assert by['A']['Best acc'] == '-'
    assert by['B']['Final loss'] == '-' and by['B']['Params'] == 0


def test_rows_survive_missing_config():
    rows = model_results_rows([mk('A', 'timeout', [0.5])], [])
    assert rows[0]['Optimizer'] == '?' and rows[0]['LR'] == '?'