
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, select_highest_overlaps

from metrics import bbox_iou, point_dist

import torch

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(num_total_anchors, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)

    # Point candidates, finds anchor points within 16 px radius
    point_mask = (gt_bboxes[:,:,0]==gt_bboxes[:,:,2]).logical_and(gt_bboxes[:,:,1]==gt_bboxes[:,:,3]).logical_and(gt_bboxes.sum(axis=2)!=0) 
    # T where: x==x and y==y and sum(xyxy)!=0 
    radius = 20     # in px (6 should cover a grid of stride 8. TODO get from stride)
    xy_centers_within_radius = torch.sqrt(bbox_deltas[point_mask][:, :, 0]**2 + bbox_deltas[point_mask][:, :, 1]**2) < radius
    candidate_point_mask = torch.full_like(bbox_deltas[..., 0], 0.)
    candidate_point_mask[point_mask] = xy_centers_within_radius.float()

    point_mask, xy_centers_within_radius = None, None # NOTE added in attempt to find memory problem
    return torch.logical_or(bbox_deltas.amin(3).gt_(eps), candidate_point_mask) # shape(bs, n_boxes, n_anchors)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes, enclosed_points, larger_mask_gt):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w). True/False for if each pred has more than one gt.
        only_box_max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        
        only_points_max_overlaps_idx = enclosed_points.argmax(1)  # (b, h*w). For each pred, select the gt with highest iou.
        max_overlaps_idx = torch.where(torch.logical_or(only_box_max_overlaps_idx, mask_pos[:, 0, :]), only_box_max_overlaps_idx, only_points_max_overlaps_idx)
        # 1st pos in OR checks pred matches with gt boxes other than 0, 2nd checks pred matches with gt box 0. 
        # If either of these arematches keeps the box gts, else take the point gts. 
        # NOTE: idx 0 indicates either a match with gt 0 or no match.

        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)   # Each pred gets assosiated only once to a gt box with highest iou.

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w. If multi gts take T/F from is_max_overlaps, else as before(i.e. mask_pos)
        fg_mask = mask_pos.sum(-2)  # each preds summation of assosiated gts.
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

    only_box_max_overlaps_idx, only_points_max_overlaps_idx = None, None # NOTE added in attempt to fix memory problem
    return target_gt_idx, fg_mask, mask_pos

class PointTaskAlignedAssigner(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection, made compatible with points.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """

        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps, point_distances, larger_mask_gt = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes, point_distances, larger_mask_gt)    # TODO prioritize boxes over points here?

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        point_distances, larger_mask_gt = None, None # NOTE added in attempt to fix memory problem
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)

        point_align_metric, point_distances = self.get_point_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        align_metric[gt_bboxes[:, :, 0] == gt_bboxes[:, :, 2]] = point_align_metric[gt_bboxes[:, :, 0] == gt_bboxes[:, :, 2]]
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        point_align_metric = None # NOTE added in attempt to fix memory problem
        return mask_pos, align_metric, overlaps, point_distances, mask_in_gts * mask_gt

    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""

        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]     # Gets the candidate predicted boxes for each gt from the anchor grid
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]                   # Gt boxes associated
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
    
    def get_point_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth points."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        point_distances = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]     # Gets the candidate predicted boxes for each gt from the anchor grid
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]                   # Gt boxes associated
        point_distances[mask_gt] = point_dist(gt_boxes, pd_boxes, xywh=False).squeeze(-1).clamp_(0) # Gets the distances between the points and the predictions

        align_metric = bbox_scores.pow(self.alpha) * point_distances.pow(self.beta)
        return align_metric, point_distances