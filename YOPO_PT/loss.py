import torch
from ultralytics.yolo.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.yolo.utils.tal import bbox2dist, make_anchors, TaskAlignedAssigner
from torch.nn import functional as F
from math import log


def image_wise_loss(pred_scores, sI, fg_mask, image_level_labels):
    """
    Args:
        pred_scores: output of "Classification" branch. Tensor of shape [bs, n_anchor_points, C].
        sI: output of "Objectness-I" branch. Tensor of shape [bs, n_anchor_points, C].
        fg_mask: masks out the candidates. Tensor of shape [bs, n_anchor_points]
        image_level_labels: one-hot labels for each class. Tensor of shape [{0, 1}, C].
    """
    mil_loss_image = 0.
    for batch_idx, (batch_pred_scores, batch_sI) in enumerate(zip(pred_scores, sI)):
        batch_mil_loss_image = []
        scores = F.log_softmax(batch_pred_scores[fg_mask[batch_idx]], dim=1)
        scores_2 = F.log_softmax(batch_sI[fg_mask[batch_idx]], dim=0)
        for cls_idx, image_level_label in enumerate(image_level_labels[batch_idx]):
            classification_score = (torch.exp( scores[:, cls_idx]) * torch.exp(scores_2[:, cls_idx])).sum()
            if classification_score > 0.99999:
                batch_mil_loss_image.append(torch.tensor(0., device=classification_score.device))  # no box in bag, set loss to zero.
            else:
                class_log_prob = image_level_label * torch.log(max(classification_score, torch.tensor(0.05, device=classification_score.device))) + ( 1 - image_level_label ) * torch.log( 1 - classification_score )
                batch_mil_loss_image.append(-class_log_prob)
        mil_loss_image += torch.stack(batch_mil_loss_image).mean()
    mil_loss_image /= pred_scores.shape[0]
    return mil_loss_image


def point_wise_loss(pred_scores, sP, pred_bboxes, gt_points, gt_point_labels, fg_mask): 
    """
    Args:
        pred_scores: output of "Classification" branch. Tensor of shape [bs, n_anchor_points, C].
        sP: output of "Objectness-P" branch. Tensor of shape [bs, n_anchor_points, 2].
        pred_bboxes: proposals from RPN. Tensor of shape [bs, n_anchor_points, 4]. point xy: coordinate of groundtruth point.
        gt_points: coordinates of a ground truth points. Tensor of shape [bs, Np, 4]
        gt_point_labels: label of ground truth points. Tensor of shape [bs, Np, C]
        fg_mask: masks out the candidates. Tensor of shape [bs, n_anchor_points]
    """
    mil_loss_points = 0.
    for batch_idx, (batch_points, point_label_list) in enumerate(zip(gt_points, gt_point_labels)):
        _ , point_xy_list, _ = torch.tensor_split(batch_points, (1,3), dim=1)
        proposal_boxes = pred_bboxes[batch_idx, fg_mask[batch_idx]]
        batch_mil_loss_points = []
        scores = F.log_softmax(pred_scores[batch_idx, fg_mask[batch_idx]], dim=1)
        scores_2 = F.log_softmax(sP[batch_idx][fg_mask[batch_idx], :], dim=1)
        for point_xy, point_label in zip(point_xy_list, point_label_list): # extract proposals with point inside
            idxs_p = ((point_xy[0] >= proposal_boxes[:, 0])
                    & (point_xy[0] <= proposal_boxes[:, 2])
                    & (point_xy[1] >= proposal_boxes[:, 1])
                    & (point_xy[1] <= proposal_boxes[:, 3])).nonzero().reshape(-1)
            scores_p = scores[idxs_p]
            scores_2_p = scores_2[idxs_p]
            log_fg_prob = scores_p[:, int(point_label)].detach() + scores_2_p[:, 1] 
            log_bg_prob = scores_2_p[:, 0]
            eye = torch.eye(len(log_fg_prob), dtype=torch.float32, device=log_fg_prob.device)
            log_prob = (eye * log_fg_prob[None, :] + \
                        (1 - eye) * log_bg_prob[None, :]).sum(dim=-1)
            if log_prob.shape[0] == 0:
                batch_mil_loss_points.append(torch.tensor(0., device=log_prob.device))  # no box in bag, set loss to zero.
            else:
                max = log_prob.max()
                point_bag_log_prob = torch.log(torch.exp(log_prob - max).sum()) + max  # e^(log(x)) = x
                batch_mil_loss_points.append(-point_bag_log_prob)
        mil_loss_points += torch.stack(batch_mil_loss_points).mean()
    mil_loss_points /= gt_points.shape[0]
    return mil_loss_points


class v8DetectionPointLoss(v8DetectionLoss):
    """Detection loss utilizing points"""
    def __init__(self, model):
        super().__init__(model)
        device = next(model.parameters()).device  # get model device
        m = model.model[-1]  # Detect() module
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, pw_MIL, iw_MIL
        if isinstance(preds, tuple):
            feats = preds[1]
        elif isinstance(preds, list):
            feats = preds[-1]
            sI_preds = preds[0]
            sP_preds = preds[1]
            sI = torch.cat([xi.view(feats[0].shape[0], self.nc, -1) for xi in sI_preds], 2)
            sI = sI.permute(0, 2, 1).contiguous()
            sP = torch.cat([xi.view(feats[0].shape[0], 2, -1) for xi in sP_preds], 2)
            sP = sP.permute(0, 2, 1).contiguous()
        else:
            feats = preds

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

        target_scores_sum = max(target_scores.sum(), 1)
        
        if 'points' in batch.keys():    # training
            targets_point = torch.cat((batch['batch_idx_points'].view(-1, 1), batch['cls_points'].view(-1, 1), batch['points']), 1)
            targets_point = self.preprocess(targets_point.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_point_labels, gt_points = targets_point.split((1, 4), 2)  # cls, xyxy
            image_level_labels = torch.zeros((batch_size, self.nc), device=gt_point_labels.device)
            for batch_idx, batch_pt_labels in enumerate(gt_labels): # image level labels
                image_level_labels[batch_idx, batch_pt_labels.unique().long()] = 1.
            
            loss[3] = point_wise_loss(pred_scores.detach().sigmoid(), sP.sigmoid(), (pred_bboxes * stride_tensor).type(gt_bboxes.dtype), gt_points, gt_point_labels, fg_mask)
            loss[4] = image_wise_loss(pred_scores.detach().sigmoid(), sI.sigmoid(), fg_mask, image_level_labels)
        else:   # validation
            loss[3] = 0.0
            loss[4] = 0.0
        
        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        
        loss[0] *= self.hyp.box     # box gain
        loss[1] *= self.hyp.cls     # cls gain
        loss[2] *= self.hyp.dfl     # dfl gain
        loss[3] *= 0.05  # point-wise MIL gain TODO: change to self.hyp.pwMIL
        loss[4] *= 1.5 # image-wise MIL gain TODO: change to self.hyp.iwMIL

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, pw_MIL, iw_MIL)
