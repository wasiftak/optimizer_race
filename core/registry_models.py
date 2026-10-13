# path: core/registry_models.py
# One dictionary read by the dropdown, the builder, the tests and (later) the Help tab.
# params = expected parameter count for a 1 x 28 x 28 input and 10 classes.
MODEL_PRESETS = {
    'default': dict(
        label='Small CNN', params=20490,
        lesson='A basic two-layer CNN; the reference model.'),
    'mini_vgg': dict(
        label='Mini-VGG', params=32058,
        lesson='Stacked 3x3 convs before each pool, like VGG.'),
    'tiny_resnet': dict(
        label='Tiny ResNet', params=9770,
        lesson='Skip connections and batch norm; stable at higher learning rates.'),
    'wide_shallow': dict(
        label='Wide and shallow', params=63562,
        lesson='One wide conv; most parameters sit in the linear head.'),
    'deep_narrow': dict(
        label='Deep and narrow', params=5762,
        lesson='Many thin layers; few parameters, harder to optimize.'),
}

MODEL_ORDER = list(MODEL_PRESETS)