import importlib


def torch_is_available():
    try:
        importlib.import_module('torch')
        return True
    except ImportError:
        print('torch is not installed')
        return False


if torch_is_available():
    from erutils.nn import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                            Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Ensemble, Focus,
                            GhostBottleneck, GhostConv, Segment, Proto, DFL)

    __all__ = ['C1', 'C2', 'C3', 'C3TR', 'SPP', 'SPPF', 'DFL', 'Proto',
               'Bottleneck', 'BottleneckCSP', 'C2f', 'C3Ghost', 'C3x',
               'Classify',
               'Concat', 'Conv', 'ConvTranspose', 'Detect', 'DWConv', 'DWConvTranspose2d', 'Ensemble', 'Focus',
               'GhostBottleneck', 'GhostConv', 'Segment']
