import math
import torch.nn as nn
from copy import copy
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
import torch

class PointDetect(Detect):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__(nc, ch)
        sI, sP = max(ch[0], min(self.nc, 100)), max(ch[0], min(self.nc, 100))  # channels
        self.sI = nn.ModuleList(nn.Sequential(Conv(x, sI, 3), Conv(sI, sI, 3), nn.Conv2d(sI, self.nc, 1)) for x in ch)  # objectness-I branch
        self.sP= nn.ModuleList(nn.Sequential(Conv(x, sP, 3), Conv(sP, sP, 3), nn.Conv2d(sP, 2, 1)) for x in ch)         # objectness-P branch

    def forward(self, x):
        """
        Concatenates and returns predicted bounding boxes and class probabilities.
        Also concatenates and returns the Objectness-P and objectness-I branch, seperately.
        NOTE: x2, x3 is added during CBD. As a postproduct cls_P as well.
        """
        shape = x[0].shape  # BCHW
        x2 = []
        x3 = []
        for i in range(self.nl):
            x2.append(self.sI[i](x[i]))
            x3.append(self.sP[i](x[i]))
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return [x2, x3, x]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        cls_P = torch.cat([xi.view(shape[0], 2, -1) for xi in x3], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x, cls_P.sigmoid())
    
    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, d, s in zip(m.cv2, m.cv3, m.sI, m.sP, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img) # TODO: same as image-size input? Default 640, but we used 1280.
            c[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # sI (.01 objects, 80 classes, 640 img) # TODO: take away?
            d[-1].bias.data[:] = math.log(5 / 2 / (640 / s) ** 2)  # sP (.01 objects, 2 columns, 640 img) # TODO: -||-
    