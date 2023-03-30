import contextlib
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn

from YOLOvision.yolo.utils import LOGGER, ROOT, yaml_load
from YOLOvision.yolo.utils.checks import check_suffix, check_yaml
from YOLOvision.yolo.utils.downloads import attempt_download_asset, is_url


def check_class_names(names, *args, **kwargs):

    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):

        names = {int(k, *args, **kwargs): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(' you have invalid class indices '
                           f' defined in your dataset YAML.')
        if isinstance(names[0], str) and names[0].startswith('n0'):
            map = yaml_load(ROOT / 'datasets/ImageNet.yaml')['map']
            names = {k: map[v] for k, v in names.items()}
    return names


class SmartLoad(nn.Module):

    def __init__(self,
                 weights='YOLOvisionn.pt',
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True,
                 detail=True, *args, **kwargs):

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt, jit = self._model_type(w)
        fp16 &= pt or jit
        nhwc = False
        stride = 32  # default stride
        model, metadata = None, None
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if nn_module:
            model = weights.to(device)
            model = model.fuse(detail=detail) if fuse else model
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            stride = max(int(model.stride.max()), 32)  # model stride
            model.half() if fp16 else model.float()
            self.model = model
            pt = True
        elif pt:
            from YOLOvision.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights if isinstance(weights, list) else w,
                                         device=device,
                                         inplace=True,
                                         fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model
        else:

            raise TypeError(f"model='{w}' is not a supported model format. "
                            )

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names') and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata['stride']
            task = metadata['task']
            batch = metadata['batch']
            imgsz = metadata['imgsz']
            names = metadata['names']
        elif not (pt or nn_module):
            LOGGER.warning(f"Metadata not found for 'model={weights}'")

        # Check names
        if 'names' not in locals():
            names = self._apply_default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, *args, **kwargs):

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)

        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:
            y = self.model(im)
        else:
            y = []
            raise ValueError('only pt and jit supported')
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x, *args, **kwargs):

        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):

        warmup_types = self.pt, self.jit, self.nn_module
        if any(warmup_types) and (self.device.type != 'cpu'):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _apply_default_class_names(data, *args, **kwargs):
        with contextlib.suppress(Exception):
            return yaml_load(check_yaml(data))['names']
        return {i: f'class{i}' for i in range(999)}

    @staticmethod
    def _model_type(p='path/to/model.pt', *args, **kwargs):

        from YOLOvision.yolo.core.exporter import supported_formats
        sf = list(supported_formats().Suffix)
        if not is_url(p, check=False) and not isinstance(p, str):
            check_suffix(p, sf)
        url = urlparse(p)
        types = [s in Path(p).name for s in sf]
        types[-2] &= not types[-1]

        print(types)
        return types
