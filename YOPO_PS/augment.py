import numpy as np
from ultralytics.yolo.data.augment import (Albumentations, Compose, CopyPaste, Instances, LetterBox, MixUp, Mosaic, 
                                           RandomFlip, RandomHSV, RandomPerspective)
from ultralytics.yolo.data.utils import LOGGER
import torch


class PointInstances(Instances):
    def remove_compressed_boxes(self, box_indicies, pts_outside):
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height. This removes them, but 
        keeps the already existing points."""
        to_keep = np.full(len(box_indicies), True)
        good = self.bbox_areas[box_indicies] > 0    # The boxes that have area after clipping
        if not all(good):
            to_keep[box_indicies] = good # True for all points and boxes with area remaining

        # Remove points outside
        to_keep = to_keep^pts_outside       # xor excludes points outside image
        self._bboxes = self._bboxes[to_keep]
        good=None   # NOTE added in attempt to find memory problem
        return to_keep      # Boxes with area and points inside image remain
    
    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation. NOTE points remain TODO remove?""" 
        assert not self.normalized, 'you should add padding with absolute coordinates.'
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh


class PointMosaic(Mosaic):
    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped. NOTE made compatible with points by not removing zero"""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': PointInstances.concatenate(instances, axis=0),
            'mosaic_border': self.border}  # final_labels
        have_area = final_labels['instances'].bbox_areas > 0 # TODO could these numpy arrays on CPU cause memory issues somehow?
        pts_outside = np.logical_or(np.any(final_labels['instances'].bboxes < 0, axis=1), 
                                    np.any(final_labels['instances'].bboxes > imgsz, axis=1)) & ~have_area    # True where value is outside [0, imgsz] and area == 0
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_compressed_boxes(have_area, pts_outside)
        have_area, pts_outside = None, None    # NOTE added in attempt to find memory problem
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels
    
class PointRandomPerspective(RandomPerspective):

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
        labels.pop('ratio_pad', None)  # do not need ratio pad
        
        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')
        # Make sure the coord formats are right
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])
        border = labels.pop('mosaic_border', self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)

        # Clip
        #NOTE: CBD added, since pts are removed otherwise.
        pts_outside = np.logical_or(np.any(new_instances.bboxes < 0, axis=1), 
                                    np.any(new_instances.bboxes> self.size[0], axis=1)) & np.array(new_instances.bbox_areas == 0)
        
        #new_instances = new_instances[~pts_outside]
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(box1=instances.bboxes.T,
                                box2=new_instances.bboxes.T,
                                area_thr=0.01 if len(segments) else 0.10)
        
        i = i^pts_outside
        labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img
        labels['resized_shape'] = img.shape[:2]

        i, pts_outside = None, None # NOTE added in attempt to find memory problem
        return labels
    
    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute box candidates: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        points = (w1 == 0) & (h1 == 0)
        
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        box_candidates = (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
        total = box_candidates | points #NOTE: CBD added, logical OR. Keeps points and box candidates.
        return total
    
def v8_point_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([
        PointMosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        PointRandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms

