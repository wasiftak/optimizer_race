# path: core/registry_datasets.py
from dataclasses import dataclass


@dataclass
class DatasetSpec:
    key: str
    name: str
    in_shape: tuple
    n_classes: int
    class_names: list
    mean: tuple
    std: tuple
    default_target: float     # provisional: calibrate by running Adam once
    lesson: str
    loader: str               # 'torchvision:FashionMNIST', 'torchvision:MNIST', 'torchvision:KMNIST'

    @property
    def chance(self):
        return 1.0 / self.n_classes


DATASETS = {
    'fashion_mnist': DatasetSpec(
        'fashion_mnist', 'FashionMNIST', (1, 28, 28), 10,
        ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
         'Sneaker', 'Bag', 'Ankle boot'],
        (0.2860,), (0.3530,), 0.85,
        'Clothing items; harder than digits.',
        'torchvision:FashionMNIST'),
    'mnist': DatasetSpec(
        'mnist', 'MNIST', (1, 28, 28), 10,
        [str(i) for i in range(10)],
        (0.1307,), (0.3081,), 0.90,
        'Handwritten digits; small CNNs nearly solve them, so races end quickly.',
        'torchvision:MNIST'),
    'kmnist': DatasetSpec(
        'kmnist', 'KMNIST', (1, 28, 28), 10,
        ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'],
        (0.1918,), (0.3483,), 0.85,
        'Handwritten Japanese characters; same size as MNIST but noticeably harder.',
        'torchvision:KMNIST'),
}

DATASET_ORDER = list(DATASETS)


def get_spec(key):
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Valid names: {DATASET_ORDER}")
    return DATASETS[key]