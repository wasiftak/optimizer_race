# path: scripts/download_datasets.py
# Downloads every dataset once so nothing downloads during a demo.
import sys
sys.path.insert(0, '.')

from core.data import get_data
from core.registry_datasets import DATASET_ORDER

for key in DATASET_ORDER:
    print(f'Loading {key} ...', flush=True)
    d = get_data(subset_size=16, test_size=16, seed=0, dataset=key)
    print(f"  ok: train {tuple(d['xtr'].shape)}, test {tuple(d['xte'].shape)}", flush=True)
print('All datasets are on disk.')