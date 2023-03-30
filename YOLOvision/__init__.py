__version__ = '8.0.58'

from YOLOvision.yolo.core.model import YOLO
from YOLOvision.yolo.utils.checks import check_yolo as checks


def prepare_segmentation(model: YOLO):
    model.task = 'segmentation'
    model.overrides['task'] = 'segmentation'
    return model


__all__ = '__version__', 'YOLO', 'checks', 'prepare_segmentation'
