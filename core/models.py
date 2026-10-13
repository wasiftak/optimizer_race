# path: core/models.py
import torch
import torch.nn as nn

from core.registry_models import MODEL_PRESETS


def _flat_size(features, in_shape):
    with torch.no_grad():
        return features(torch.zeros(1, *in_shape)).numel()


class SmallCNN(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, c1=16, c2=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(_flat_size(self.features, in_shape), n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class MiniVGG(nn.Module):
    """Two 3x3 convs, pool, two 3x3 convs, pool, linear."""
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        c = in_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(_flat_size(self.features, in_shape), n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class ResBlock(nn.Module):
    """conv-BN-ReLU-conv-BN, identity skip, ReLU after the addition."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + x)


class TinyResNet(nn.Module):
    """Stem conv+BN, two residual blocks (pool after each), global average pool, linear."""
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, ch=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_shape[0], ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        self.block1 = ResBlock(ch)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ResBlock(ch)
        self.pool2 = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        return self.classifier(self.gap(x).flatten(1))


class WideShallow(nn.Module):
    """One 5x5 conv with 32 filters, pool, linear."""
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(_flat_size(self.features, in_shape), n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class DeepNarrow(nn.Module):
    """Four convs with 8 filters each, pool after the 2nd and 4th, linear."""
    def __init__(self, in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        c = in_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(c, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(_flat_size(self.features, in_shape), n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


MODEL_CLASSES = {
    'default': SmallCNN,
    'mini_vgg': MiniVGG,
    'tiny_resnet': TinyResNet,
    'wide_shallow': WideShallow,
    'deep_narrow': DeepNarrow,
}


def build_model(cfg, in_shape, n_classes):
    """Dispatch on cfg.model. Phase 7 adds 'custom' (cfg.arch)."""
    name = getattr(cfg, 'model', 'default')
    if name not in MODEL_CLASSES:
        raise KeyError(f"Unknown model '{name}'. Valid names: {list(MODEL_CLASSES)}")
    return MODEL_CLASSES[name](in_shape, n_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def same_model(racers):
    """True when every racer uses the same architecture (a pure optimizer test)."""
    keys = {(getattr(r, 'model', 'default'), repr(getattr(r, 'arch', None))) for r in racers}
    return len(keys) <= 1


def summarize(model, in_shape):
    """Layer table from forward hooks: layer name, type, output shape, own parameters.
    The parameter column sums to count_params(model)."""
    rows, hooks, seen = [], [], set()

    def make_hook(name):
        def hook(mod, inputs, out):
            if name in seen or not isinstance(out, torch.Tensor):
                return
            seen.add(name)
            own = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
            rows.append(dict(layer=name, type=type(mod).__name__,
                             output_shape=tuple(out.shape[1:]), params=own))
        return hook

    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *in_shape, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return rows