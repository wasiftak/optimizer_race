# path: core/schemas.py
from dataclasses import dataclass, field, asdict  # noqa: F401
from typing import Optional


@dataclass
class RacerConfig:
    name: str
    color: str                      # hex, e.g. '#e63946'
    optimizer: str                  # 'SGD','Momentum','Nesterov','Adagrad','RMSprop','Adam','AdamW'
    lr: float = 0.01
    momentum: float = 0.9           # Momentum, Nesterov, RMSprop
    weight_decay: float = 0.0
    beta1: float = 0.9              # Adam, AdamW
    beta2: float = 0.999
    scheduler: str = 'none'         # 'none','step','cosine','exponential' (Phase 3)
    model: str = 'default'          # preset name or 'custom' (Phase 5)
    arch: Optional[dict] = None     # custom architecture dict (Phase 7)


@dataclass
class RaceConfig:
    mode: str                       # 'landscape' or 'model'
    seed: int = 42
    max_steps: int = 300
    terrain: str = 'bowl'           # landscape only
    start: tuple = (-1.5, 1.5)      # landscape only
    dataset: str = 'fashion_mnist'  # model only (Phase 6 adds choices)
    subset_size: int = 3000
    batch_size: int = 64
    target_acc: float = 0.85
    eval_every: int = 10            # steps between test evaluations
    race_format: str = 'first_to_target'   # Phase 8
    param_budget: Optional[int] = None     # Phase 8


@dataclass
class RacerLog:
    name: str
    steps: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    acc: list = field(default_factory=list)   # float, or None where not evaluated
    x: list = field(default_factory=list)     # landscape only
    y: list = field(default_factory=list)     # landscape only
    status: str = 'running'         # 'running','finished','crashed','timeout'
    finish_step: Optional[int] = None
    n_params: Optional[int] = None
    seconds: float = 0.0