__version__ = '0.0.2'

from YOLOvision.yolo.core.model import YOLO
from YOLOvision.yolo.utils.checks import check_yolo as checks


def prepare_segmentation(model: YOLO):
    model.task = 'segmentation'
    model.overrides['task'] = 'segmentation'
    model.cfa = 'segmentation'
    model.cpa = 'segmentation-/r'
    # model.overrides['device_local'] = 'auto'
    try:

        import accelerate
        # model.overrides['init_model_list'] = [accelerate.init_on_device, accelerate.init_empty_weights]
        model.has_accelerate = True
        model.accelerate_version = accelerate.__version__

    except ModuleNotFoundError as er:

        # model.overrides['init_model_list'] = None
        model.has_accelerate = False
        model.accelerate_version = 'v0.0.0.0'

    model.use_bnb = False
    return model


def set_new_setting(model: YOLO,
                    task=None, mode=None,
                    data=None, epochs=None, patience=None, batch=None, imgsz=None, save=None, save_period=None,
                    cache=None, device=None, workers=None, project=None, name=None, exist_ok=None, pretrained=None,
                    optimizer=None, detail=None, seed=None, deterministic=None, single_cls=None, image_weights=None,
                    rect=None, cos_lr=None, close_mosaic=None, resume=None, amp=None, overlap_mask=None,
                    mask_ratio=None,
                    dropout=None, val=None, split=None, save_json=None, save_hybrid=None, conf=None, iou=None,
                    max_det=None, half=None, dnn=None, plots=True, source=None, show=None, save_txt=None,
                    save_conf=None, save_crop=None, hide_labels=None, hide_conf=None, vid_stride=None,
                    line_thickness=None, visualize=None, augment=None, agnostic_nms=None, classes=None,
                    retina_masks=None, boxes=None, format=None, optimize=None, int8=None,
                    dynamic=None, simplify=None, opset=None, workspace=None, nms=None, lr0=None, lrf=None,
                    momentum=None, weight_decay=None, warmup_epochs=None, warmup_momentum=None, warmup_bias_lr=None,
                    box=None, cls=None, dfl=None, fl_gamma=None, label_smoothing=None, nbs=None, hsv_h=None, hsv_s=None,
                    hsv_v=None, degrees=None, translate=None, scale=None, shear=None, perspective=None, flipud=None,
                    fliplr=None, mosaic=None, mixup=None, copy_paste=None, cfg=None

                    ):
    args = {task: None, mode: None,
            'data': data, 'epochs': epochs, 'patience': patience, 'batch': batch, 'imgsz': imgsz, 'save': save,
            'save_period': save_period,
            'cache': cache, 'device': device, 'workers': workers, 'project': project, 'name': name,
            'exist_ok': exist_ok,
            'pretrained': pretrained,
            'optimizer': optimizer, 'detail': detail, 'seed': seed, 'deterministic': deterministic,
            'single_cls': single_cls,
            'image_weights': image_weights,
            'rect': rect, 'cos_lr': cos_lr, 'close_mosaic': close_mosaic, 'resume': resume, 'amp': amp,
            'overlap_mask': overlap_mask,
            'mask_ratio': mask_ratio,
            'dropout': dropout, 'val': val, 'split': split, 'save_json': save_json, 'save_hybrid': save_hybrid,
            'conf': conf,
            'iou': iou,
            'max_det': max_det, 'half': half, 'dnn': dnn, 'plots': plots, 'source': source, 'show': show,
            'save_txt': save_txt,
            'save_conf': save_conf, 'save_crop': save_crop, 'hide_labels': hide_labels, 'hide_conf': hide_conf,
            'vid_stride': vid_stride,
            'line_thickness': line_thickness, 'visualize': visualize, 'augment': augment, 'agnostic_nms': agnostic_nms,
            'classes': classes,
            'retina_masks': retina_masks, 'boxes': boxes, 'format': format, 'optimize': optimize, 'int8': int8,
            'dynamic': dynamic, 'simplify': simplify, 'opset': opset, 'workspace': workspace, 'nms': nms, 'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum, 'weight_decay': weight_decay, 'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            'box': box, 'cls': cls, 'dfl': dfl, 'fl_gamma': fl_gamma, 'label_smoothing': label_smoothing, 'nbs': nbs,
            'hsv_h': hsv_h, 'hsv_s': hsv_s,
            'hsv_v': hsv_v, 'degrees': degrees, 'translate': translate, 'scale': scale, 'shear': shear,
            'perspective': perspective,
            'flipud': flipud,
            'fliplr': fliplr, 'mosaic': mosaic, 'mixup': mixup, 'copy_paste': copy_paste, 'cfg': cfg}
    for k, v in args.items():
        if v is not None:
            setattr(model, k, v)
            model.overrides[k] = v
    return model


__all__ = '__version__', 'YOLO', 'checks', 'prepare_segmentation', 'set_new_setting'
