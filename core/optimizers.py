# path: core/optimizers.py
import torch

OPTIMIZER_NAMES = ['SGD', 'Momentum', 'Nesterov', 'Adagrad', 'RMSprop', 'Adam', 'AdamW']


def build_optimizer(cfg, params):
    """The only place that maps dropdown names to torch.optim."""
    name, lr, wd = cfg.optimizer, cfg.lr, cfg.weight_decay
    m, betas = cfg.momentum, (cfg.beta1, cfg.beta2)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    if name == 'Momentum':
        return torch.optim.SGD(params, lr=lr, momentum=m, weight_decay=wd)
    if name == 'Nesterov':
        if m <= 0:
            raise ValueError('Nesterov needs momentum greater than 0')
        return torch.optim.SGD(params, lr=lr, momentum=m, nesterov=True, weight_decay=wd)
    if name == 'Adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == 'RMSprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=m, weight_decay=wd)
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Valid names: {OPTIMIZER_NAMES}")


def build_scheduler(cfg, optimizer, max_steps):
    """Returns None for 'none'. The race loop calls scheduler.step() once per step."""
    s = cfg.scheduler
    if s == 'none':
        return None
    if s == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, max_steps // 3), gamma=0.5)
    if s == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    if s == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    raise ValueError(f"Unknown scheduler '{s}'. Valid: none, step, cosine, exponential")