__version__ = '0.0.2'

from YOLOvision.yolo.core.model import YOLO
from YOLOvision.yolo.utils.checks import check_yolo as checks


def prepare_segmentation(model: YOLO):
    model.task = 'segmentation'
    model.overrides['task'] = 'segmentation'
    model.overrides['cfg_ae'] = 'segmentation-/r'
    model.overrides['device_local'] = 'auto'
    try:

        import accelerate
        model.overrides['init_model_list'] = [accelerate.init_on_device, accelerate.init_empty_weights]
        model.has_accelerate = True
        model.accelerate_version = accelerate.__version__

    except ModuleNotFoundError as er:

        model.overrides['init_model_list'] = None
        model.has_accelerate = False
        model.accelerate_version = 'v0.0.0.0'

    model.use_bnb = False
    return model


__all__ = '__version__', 'YOLO', 'checks', 'prepare_segmentation'
