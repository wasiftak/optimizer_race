# path: core/config.py
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'cache'
DATA_DIR = ROOT / 'data'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SEED = 42
RACER_COLORS = ['#e63946', '#1d7bd8', '#2a9d8f', '#f4a261']

CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)