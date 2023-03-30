 

from YOLOvision.yolo.data import build_classification_dataloader
from YOLOvision.yolo.core.validator import BaseValidator
from YOLOvision.yolo.utils import DEFAULT_CFG, LOGGER
from YOLOvision.yolo.utils.metrics import ClassifyMetrics


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, **kwargs):
        super().__init__(dataloader, save_dir, pbar, args)
        self.args.task = 'classify'
        self.metrics = ClassifyMetrics()

    def get_desc(self, *args, **kwargs):
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

    def init_metrics(self, model, *args, **kwargs):
        self.pred = []
        self.targets = []

    def preprocess(self, batch, *args, **kwargs):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch, *args, **kwargs):
        n5 = min(len(self.model.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed
        # self.metrics.confusion_matrix = self.confusion_matrix  # TODO: classification ConfusionMatrix

    def get_stats(self, *args, **kwargs):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size, *args, **kwargs):
        return build_classification_dataloader(path=dataset_path,
                                               imgsz=self.args.imgsz,
                                               batch_size=batch_size,
                                               augment=False,
                                               shuffle=False,
                                               workers=self.args.workers)

    def print_results(self, *args, **kwargs):
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.metrics.top1, self.metrics.top5))


