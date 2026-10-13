# path: core/models.py
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_shape=(1, 28, 28), n_classes=10, c1=16, c2=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        with torch.no_grad():
            flat = self.features(torch.zeros(1, *in_shape)).numel()
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(cfg, in_shape, n_classes):
    return SmallCNN(in_shape, n_classes)       # Phase 5 dispatches on cfg.model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)