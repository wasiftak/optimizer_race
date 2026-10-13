# path: core/config.py
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'cache'
DATA_DIR = ROOT / 'data'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SEED = 42
RACER_COLORS = ['#e63946', '#1d7bd8', '#2a9d8f', '#f4a261']

# Which hyperparameters each optimizer uses (the Garage shows only these)
OPTIMIZER_PARAMS = {
    'SGD': ['lr', 'weight_decay'],
    'Momentum': ['lr', 'momentum', 'weight_decay'],
    'Nesterov': ['lr', 'momentum', 'weight_decay'],
    'Adagrad': ['lr', 'weight_decay'],
    'RMSprop': ['lr', 'momentum', 'weight_decay'],
    'Adam': ['lr', 'beta1', 'beta2', 'weight_decay'],
    'AdamW': ['lr', 'beta1', 'beta2', 'weight_decay'],
}
LR_CHOICES = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 8e-2,
              0.1, 0.11, 0.2, 0.3, 1.0]

CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)