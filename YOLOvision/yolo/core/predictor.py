"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=YOLOvisionn.pt --source 0                               # webcam
                                                  img.jpg                         # image
                                                  vid.mp4                         # video
                                                  screen                          # screenshot
                                                  path/                           # directory
                                                  list.txt                        # list of images
                                                  list.streams                    # list of streams
                                                  'path/*.jpg'                    # glob
                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=YOLOvisionn.pt                 # PyTorch
                              YOLOvisionn.torchscript        # TorchScript
                              YOLOvisionn.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              YOLOvisionn_openvino_model     # OpenVINO
                              YOLOvisionn.core             # TensorRT
                              YOLOvisionn.mlmodel            # CoreML (macOS-only)
                              YOLOvisionn_saved_model        # TensorFlow SavedModel
                              YOLOvisionn.pb                 # TensorFlow GraphDef
                              YOLOvisionn.tflite             # TensorFlow Lite
                              YOLOvisionn_edgetpu.tflite     # TensorFlow Edge TPU
                              YOLOvisionn_paddle_model       # PaddlePaddle
"""
import os.path
import pathlib
import platform
from collections import defaultdict
from pathlib import Path

import cv2

from YOLOvision.nn.autobackend import SmartLoad
from YOLOvision.yolo.cfg import get_cfg
from YOLOvision.yolo.data import load_inference_source
from YOLOvision.yolo.data.augment import classify_transforms
from YOLOvision.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from YOLOvision.yolo.utils.checks import check_imgsz, check_imshow
from YOLOvision.yolo.utils.files import increment_path
from YOLOvision.yolo.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
    Opps Wait  stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segmentation masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, *args, **kwargs):

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    def preprocess(self, img, *args, **kwargs):
        pass

    def get_annotator(self, img, *args, **kwargs):
        raise NotImplementedError('get_annotator function needs to be implemented')

    def write_results(self, results, batch, print_string, *args, **kwargs):
        raise NotImplementedError('print_results function needs to be implemented')

    def postprocess(self, preds, img, orig_img, *args, **kwargs):
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None, *args, **kwargs):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source, *args, **kwargs):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        if self.args.task == 'classify':
            transforms = getattr(self.model.model, 'transforms', classify_transforms(self.imgsz[0]))
        else:  # predict, segmentation
            transforms = None
        self.dataset = load_inference_source(source=source,
                                             transforms=transforms,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             stride=self.model.stride,
                                             auto=self.model.pt)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        if self.args.detail:
            LOGGER.info('')
        if not os.path.exists(self.save_dir):
            self.save_dir: pathlib.WindowsPath = self.save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.warning('Out Folder Created')
        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            # preprocess
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # visualize, save, write results
            n = len(im)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                    else (path, im0s.copy())
                p = Path(p)

                if self.args.detail or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.detail:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.detail and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, detail=True, *args, **kwargs):
        device = select_device(self.args.device, detail=detail)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = SmartLoad(model,
                               device=device,
                               dnn=self.args.dnn,
                               data=self.args.data,
                               fp16=self.args.half,
                               detail=detail)
        self.device = device
        self.model.eval()

    def show(self, p, *args, **kwargs):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[4].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path, *args, **kwargs):
        im0 = self.annotator.result()
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter, *args, **kwargs):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
