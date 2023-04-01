import contextlib
from copy import deepcopy
from pathlib import Path

import thop
import re
import torch
import torch.nn as nn

from YOLOvision.nn.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                   Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Ensemble, Focus,
                                   GhostBottleneck, GhostConv, Segment)
from YOLOvision.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from YOLOvision.yolo.utils.checks import check_suffix, check_yaml
from YOLOvision.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                               intersect_dicts, make_divisible, model_info, scale_img, time_sync)


class BaseModel(nn.Module):

    def forward(self, x, profile=False, visualize=False, *args, **kwargs):

        return self.forward_once_(x, profile, visualize)

    def forward_once_(self, x, profile=False, *args, **kwargs):

        spa, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = spa[m.f] if isinstance(m.f, int) else [x if j == -1 else spa[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            spa.append(x if m.i in self.save else None)

        return x

    def _profile_one_layer(self, m, x, dt, *args, **kwargs):

        c = m == self.model[-1]
        o = thop.profile(m, inputs=[x.clone() if c else x], detail=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10, *args, **kwargs):
            m(x.clone() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt, *args, **kwargs):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, detail=True, *args, **kwargs):

        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            self.info(detail=detail)

        return self

    def is_fused(self, thresh=10, *args, **kwargs):

        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detail=True, imgsz=640, *args, **kwargs):

        model_info(self, detail=detail, imgsz=imgsz)

    def _apply(self, fn, *args, **kwargs):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, detail=True, *args, **kwargs):

        model = weights['model'] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)


class DetectionModel(BaseModel):

    def __init__(self, cfg='YOLOvisionn.yaml', ch=3, nc=None, detail=False, *args, **kwargs):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, detail=detail)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()

            # Init downloads, biases
        initialize_weights(self)
        if detail:
            self.info()
            LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False, *args, **kwargs):
        if augment:
            return self._forward_augment(x)
        return self.forward_once_(x, profile, visualize)

    def _forward_augment(self, x, *args, **kwargs):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        spa = []
        for si, fi in zip(s, f, *args, **kwargs):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once_(xi)[0]  # forward

            yi = self._descale_pred(yi, fi, si, img_size)
            spa.append(yi)
        spa = self._clip_augmented(spa)
        return torch.cat(spa, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1, *args, **kwargs):
        # de-scale predictions following augmented inference (inverse operation)
        p[:, :4] /= scale  # de-scale
        x, spa, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            spa = img_size[0] - spa  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, spa, wh, cls), dim)

    def _clip_augmented(self, spa, *args, **kwargs):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (spa[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        spa[0] = spa[0][..., :-i]  # large
        i = (spa[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        spa[-1] = spa[-1][..., i:]  # small
        return spa


class SegmentationModel(DetectionModel):
    #
    def __init__(self, cfg='YOLOvisionn-seg.yaml', ch=3, nc=None, detail=True, *args, **kwargs):
        super().__init__(cfg, ch, nc, detail)

    def _forward_augment(self, x, *args, **kwargs):
        raise NotImplementedError


class ClassificationModel(BaseModel):
    # YOLOvision classification model
    def __init__(self,
                 cfg=None,
                 model=None,
                 ch=3,
                 nc=None,
                 cutoff=10,
                 detail=True, *args, **kwargs):
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, detail)

    def _from_detection_model(self, model, nc=1000, cutoff=10, *args, **kwargs):

        from YOLOvision.nn.autobackend import SmartLoad
        if isinstance(model, SmartLoad, *args, **kwargs):
            model = model.model
        model.model = model.model[:cutoff]
        m = model.model[-1]
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels
        c = Classify(ch, nc)
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'
        model.model[-1] = c
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, detail=False, *args, **kwargs):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        elif not nc and not self.yaml.get('nc', None, *args, **kwargs):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, detail=detail)
        self.stride = torch.Tensor([1])
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.info()

    @staticmethod
    def reshape_outputs(model, nc, *args, **kwargs):
        # Update a TorchVision classification model to class count 'n' if required
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


def torch_safe_load(weight, *args, **kwargs):
    from YOLOvision.yolo.utils.downloads import download_from_git

    check_suffix(file=weight, suffix='.pt')

    file = download_from_git(weight)

    try:
        return torch.load(file, map_location='cpu'), file
    except ModuleNotFoundError as e:
        if e.name == 'models':
            raise TypeError
        print(file)
        return torch.load(file, map_location='cpu'), file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False, *args, **kwargs):
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:

        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()

        model.args = args
        model.pt_path = w
        model.task = get_model_task_from_cfg(model)
        if not hasattr(model, 'stride', *args, **kwargs):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

            # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts: {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False, *args, **kwargs):

    ckpt, weight = torch_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **ckpt['train_args']}
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()

    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path = weight
    model.task = get_model_task_from_cfg(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    return model, ckpt


def parse_model(d, ch, detail=False, *args, **kwargs):
    import ast

    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if detail:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if detail:
        LOGGER.info(f"\n{'':>3}{'From':>20}{'num':>5}{'Params':>10}  {'Module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if detail:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path, *args, **kwargs):
    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)

        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path, *args, **kwargs):
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)
    return ''


def get_model_task_from_cfg(model, *args, **kwargs):
    def config_to_task(cfg, *args, **kwargs):

        m = cfg['head'][-1][-2].lower()
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detection':
            return 'detection'
        if m == 'segmentation':
            return 'segmentation'

    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return config_to_task(model)
    elif isinstance(model, nn.Module):
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return config_to_task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detection'
            elif isinstance(m, Segment):
                return 'segmentation'
            elif isinstance(m, Classify):
                return 'classify'
    elif isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segmentation' in model.parts:
            return 'segmentation'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif 'detection' in model.parts:
            return 'detection'
    else:
        LOGGER.warning('NO ANT SPECIFIC TASK FOUND RETURN DEFAULT DETECTION')
        return 'detection'
