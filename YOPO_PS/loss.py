import torch
from ultralytics.yolo.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.yolo.utils.tal import bbox2dist, make_anchors

import metrics# bbox_iou
from tal import PointTaskAlignedAssigner




class BboxPointLoss(BboxLoss):
    """BBox loss class that calculates a different loss for points. TODO describe how its different"""
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = metrics.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max) # Vad gör denna?
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl



class v8DetectionPointLoss(v8DetectionLoss):
    """Detection loss utilizing points"""
    def __init__(self, model):
        super().__init__(model)
        device = next(model.parameters()).device  # get model device
        m = model.model[-1]  # Detect() module
        self.assigner = PointTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0) #NOTE: a custom TAL for CBD
        self.bbox_loss = BboxPointLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)  #NOTE: a custom BboxLoss for CBD

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # Sums over third dim? then 1/0 for greater than 0 (bs, n_max_boxes, 4)


        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        all_but_points_mask = fg_mask.float() - fg_mask.float() + 1.
        if fg_mask.sum() > 0:
            target_point_mask = (target_bboxes[fg_mask][:, 0] - target_bboxes[fg_mask][:, 2]) == 0 # True where x1 == x2
            all_but_points_mask[fg_mask] -= target_point_mask.float()  # Set all except the gt points to true.
        all_but_points_mask = all_but_points_mask.bool()
        target_scores_sum = max(target_scores[all_but_points_mask].sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores[all_but_points_mask], target_scores[all_but_points_mask].to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, all_but_points_mask)
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
