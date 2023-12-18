import torch
import cv2
from pathlib import Path

from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, LOGGER, colorstr, ops
from ultralytics.yolo.utils.ops import scale_boxes, Profile, non_max_suppression
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.metrics import box_iou

#from custom_nms import non_max_suppression

class DetectionPointPredictor(DetectionPredictor): # TODO currently equivalent to parent. Remove and use DetectionPredictor?
    ""
    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    
def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from model import YOPO_PT
        YOPO_PT(model)(**args)
    else:
        predictor = DetectionPointPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
