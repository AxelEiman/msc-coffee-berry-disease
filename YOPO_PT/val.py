from ultralytics.yolo.v8.detect import DetectionValidator
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.utils.metrics import box_iou

import metrics# import box_iou

import torch
import numpy as np

class DetectionPointValidator(DetectionValidator): # TODO currently equivalent to parent. Remove and use DetectionValidator?

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.metrics = metrics.DetMetrics2(save_dir=self.save_dir, on_plot=self.on_plot)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       max_time_img=1.0) # Original max time = 0.05
    
    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
    

# Kanske ta bort det neadan TODO
def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'
    print(f'Validating {model} on {data}...')

    args = dict(model=model, data=data)
    if use_python:
        from model import YOPO_PT
        YOPO_PT(model).val(**args)
    else:
        validator = DetectionPointValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
