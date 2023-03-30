import json
from time import time

from YOLOvision.yolo.utils import LOGGER
from YOLOvision.yolo.utils.torch_utils import get_flops, get_num_params
from random import random
import time

class Traces:

    def __init__(self):
        self.rate_limit = 3.0
        self.t = 0.0

        self.enabled = False

    def __call__(self, cfg, all_keys=False, traces_sample_rate=1.0):
        t = time.time()
        if self.enabled and random() < traces_sample_rate and (t - self.t) > self.rate_limit:
            self.t = t


traces = Traces()


def on_pretrain_routine_end(trainer, *args, **kwargs):
    session = getattr(trainer, 'hub_session', None)
    if session:
        session.timers = {'metrics': time(), 'ckpt': time()}


def on_fit_epoch_end(trainer, *args, **kwargs):
    session = getattr(trainer, 'hub_session', None)
    if session:
        all_plots = {**trainer.label_loss_items(trainer.tloss, prefix='train'), **trainer.metrics}
        if trainer.epoch == 0:
            model_info = {
                'model/parameters': get_num_params(trainer.model),
                'model/GFLOPs': round(get_flops(trainer.model), 3),
                'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
            all_plots = {**all_plots, **model_info}
        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)
        if time() - session.timers['metrics'] > session.rate_limits['metrics']:
            session.upload_metrics()
            session.timers['metrics'] = time()
            session.metrics_queue = {}


def on_model_save(trainer, *args, **kwargs):
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers['ckpt'] > session.rate_limits['ckpt']:
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers['ckpt'] = time()  #


def on_train_end(trainer, *args, **kwargs):
    session = getattr(trainer, 'hub_session', None)
    if session:
        LOGGER.info(f'Syncing final model...')
        session.upload_model(trainer.epoch, trainer.best, map=trainer.metrics.get('metrics/mAP50-95(B)', 0), final=True)
        session.alive = False
        LOGGER.info(f'Done ✅')


def on_train_start(trainer, *args, **kwargs):
    traces(trainer.args, traces_sample_rate=1.0)


def on_val_start(validator, *args, **kwargs):
    traces(validator.args, traces_sample_rate=1.0)


def on_predict_start(predictor, *args, **kwargs):
    traces(predictor.args, traces_sample_rate=1.0)


def on_export_start(exporter, *args, **kwargs):
    traces(exporter.args, traces_sample_rate=1.0)


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_model_save': on_model_save,
    'on_train_end': on_train_end,
    'on_train_start': on_train_start,
    'on_val_start': on_val_start,
    'on_predict_start': on_predict_start,
    'on_export_start': on_export_start}
