import sys
from pathlib import Path
from typing import Union

from YOLOvision import yolo  # noqa
from YOLOvision.nn.tasks import (ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight,
                                 guess_model_task, nn, yaml_model_load)
from YOLOvision.yolo.cfg import get_cfg
from YOLOvision.yolo.engine.exporter import Exporter
from YOLOvision.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, callbacks,
                                   yaml_load)
from YOLOvision.yolo.utils.checks import check_file, check_imgsz, check_yaml
from YOLOvision.yolo.utils.torch_utils import smart_inference_mode

TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
        yolo.v8.classify.ClassificationPredictor],
    'detect': [
        DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
        yolo.v8.detect.DetectionPredictor],
    'segment': [
        SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
        yolo.v8.segment.SegmentationPredictor]}


class YOLO:

    def __init__(self, model: Union[str, Path] = 'YOLOvisionn.pt', task=None, session=None) -> None:

        self._reset_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.task = None
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}

        model = str(model).strip()
        print(f'MODEL  : {model}')
        suffix = Path(model).suffix

        if suffix == '.yaml':
            self._create_new_model(model, task)
        elif suffix == '.pt':
            self._load_model(model, task)
        else:
            raise NotImplementedError(f'the option {suffix} not implemented yet')

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _create_new_model(self, cfg: str, task=None, verbose=True):

        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)
        self.overrides['model'] = self.cfg

        args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
        self.model.task = self.task

    def _load_model(self, weights: str, task=None):
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    @smart_inference_mode()
    def reset_weights(self):

        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    @smart_inference_mode()
    def load(self, weights='YOLOvisionn.pt'):

        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, verbose=False):
        self.model.info(verbose=verbose)

    def fuse(self):

        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        if source is None:
            raise ValueError("source Can't be None")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('YOLOvision')) and \
                 ('predict' in sys.argv or 'mode=predict' in sys.argv)

        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, **kwargs):
        from YOLOvision.tracker import register_tracker
        register_tracker(self)
        conf = kwargs.get('conf') or 0.1
        kwargs['conf'] = conf
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):

        overrides = self.overrides.copy()
        overrides['rect'] = True
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def benchmark(self, **kwargs):

        from YOLOvision.yolo.utils.benchmarks import benchmark
        overrides = self.model.args.copy()
        overrides.update(kwargs)
        overrides = {**DEFAULT_CFG_DICT, **overrides}
        return benchmark(model=self, imgsz=overrides['imgsz'], half=overrides['half'], device=overrides['device'])

    def export(self, **kwargs):

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1
        return Exporter(overrides=args)(model=self.model)

    def train(self, **kwargs):

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path

        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    def to(self, device):

        self.model.to(device)

    @property
    def names(self):

        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):

        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):

        return self.model.transforms if hasattr(self.model, 'transforms') else None

    @staticmethod
    def add_callback(event: str, func):
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        include = {'imgsz', 'data', 'task', 'single_cls'}
        return {k: v for k, v in args.items() if k in include}

    @staticmethod
    def _reset_callbacks():
        for event in callbacks.default_callbacks.keys():
            callbacks.default_callbacks[event] = [callbacks.default_callbacks[event][0]]
