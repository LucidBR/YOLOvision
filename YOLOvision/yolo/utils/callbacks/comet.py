 
from YOLOvision.yolo.utils import LOGGER, TESTS_RUNNING
from YOLOvision.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import comet_ml

    assert not TESTS_RUNNING  # do not log pytest
    assert comet_ml.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    comet_ml = None


def on_pretrain_routine_start(trainer, *args, **kwargs):
    try:
        experiment = comet_ml.Experiment(project_name=trainer.args.project or 'YOLOvision')
        experiment.set_name(trainer.args.name)
        experiment.log_parameters(vars(trainer.args))
    except Exception as e:
        LOGGER.warning(f'Opps Wait  Comet installed but not initialized correctly, not logging this run. {e}')


def on_train_epoch_end(trainer, *args, **kwargs):
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1)
        if trainer.epoch == 1:
            for f in trainer.save_dir.glob('train_batch*.jpg', *args, **kwargs):
                experiment.log_image(f, name=f.stem, step=trainer.epoch + 1)


def on_fit_epoch_end(trainer, *args, **kwargs):
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.log_metrics(trainer.metrics, step=trainer.epoch + 1)
        if trainer.epoch == 0:
            model_info = {
                'model/parameters': get_num_params(trainer.model),
                'model/GFLOPs': round(get_flops(trainer.model), 3),
                'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
            experiment.log_metrics(model_info, step=trainer.epoch + 1)


def on_train_end(trainer, *args, **kwargs):
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.log_model('YOLOvision', file_or_folder=str(trainer.best), file_name='best.pt', overwrite=True)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if comet_ml else {}
