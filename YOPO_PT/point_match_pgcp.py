import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.yolo.utils.ops import xywh2xyxy

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def point_matcher(pred_boxes, gts, s_preds, sP_preds):
    gt_boxes = gts[gts[:, 3] != 0]
    points = gts[gts[:, 3] == 0]

    xyxy_boxes = xywh2xyxy(pred_boxes[:, :, 1:])

    n_points = points.shape[0]
    bs, n_boxes, _ = pred_boxes.shape

    _ , pts, _ = torch.tensor_split(points, (1,3), dim=1)

    lt, rb = torch.tensor_split(xyxy_boxes.view(-1,1,4), (2), dim=2) # Splits a single batch (N_b x 5). view(-1,1,5) probably works for batches (b, N_b, 5) or something similar
    
    # Calculate distances from points to corners to find enclosing boxes
    bbox_deltas = torch.cat((pts[None] - lt, rb - pts[None]), dim=2).view(bs, n_boxes, n_points, -1)

    class_matches = torch.zeros((points.size(0), pred_boxes.size(1)), device=points.device)
    p_enclosed_by_boxes = bbox_deltas.amin(3).gt(0)[0].T.float()  # (n_p, b_b)
    class_matches[ (points[:, 0].unsqueeze(0) == pred_boxes[:, :, 0].T).T ] = 1.    # (n_p, n_boxes)
    pred_boxes_in_bags = torch.logical_and(p_enclosed_by_boxes, class_matches)
    L_match = torch.zeros((points.size(0), pred_boxes.size(1)))

    for j, point_label in enumerate(points[:, 0]):
        L_match[j, :] = (1. - pred_boxes_in_bags[j].float()) + ( 1. - s_preds[int(point_label)] * sP_preds[1])  # point matching, spatial and classification in the respective parenthesis.
    point_ind, box_ind = linear_sum_assignment(L_match) # Hungarian point matching
    points_unmatched = tensor_delete(torch.arange(0, points.size(0)), point_ind)
    point_matching = torch.zeros((gt_boxes.size(0) + point_ind.size*2 + points_unmatched.size(0), 5))
    point_matching[:gt_boxes.size(0)] = gt_boxes
    point_matching[gt_boxes.size(0):gt_boxes.size(0) + (point_ind.size*2):2] = points[point_ind]
    point_matching[gt_boxes.size(0)+1:gt_boxes.size(0) + (point_ind.size*2)+1:2] = pred_boxes[0, box_ind]
    point_matching[gt_boxes.size(0) + (point_ind.size*2):] = points[points_unmatched]
    return point_matching

def point_guided_copy_paste(labels, gt_labels_library):
    ''' Performs the point guided label copy paste method. Returns the labels along with the newly added pglcp boxes.

    Args
        labels:             np.array, where each row is a label in the format [C, x, y, w, h]. First GT, then point annotations with
                            their match box immedietly after.
        gt_labels_library:  np.array with GT labels from all training or batch files.
    '''
    # TODO: two outputs, points and box_labels
    labels = labels.to('cpu')
    prev_is_point = False
    box_labels = labels[labels[:, 3] != 0.0]
    first_entry = True
    for label in labels:
        if label[-1]*label[-2] == 0.0 and prev_is_point:  # current and previous label are points
            same_class_gt_boxes = box_labels[box_labels[:, 0] == prev_point[0]]   # gt boxes with the same class
            if len(same_class_gt_boxes) == 0:   # no box with the same class in the current image
                same_class_gt_boxes = gt_labels_library[gt_labels_library[:, 0] == prev_point[0]].reshape(-1, 5)
                pgcp_box = same_class_gt_boxes[np.random.randint(same_class_gt_boxes.shape[0], size=1)].squeeze() # take a random box from GT-library
            else:
                dist_to_boxes = np.sqrt( ( prev_point[1]-same_class_gt_boxes[:, 1] )**2 + ( prev_point[2]-same_class_gt_boxes[:, 2] )**2 )
                pgcp_box = same_class_gt_boxes[np.argmin(dist_to_boxes)]    # take the closest box
            new_labels = torch.cat((new_labels, torch.cat((prev_point[:3], pgcp_box[3:]), 0)), 0)
            prev_point = label
        elif label[-1]*label[-2] == 0.0:    # just the current label is a point
            prev_is_point = True
            prev_point = label
        else:   # current label is a box
            prev_is_point = False
        if first_entry:
            first_entry = False
            new_labels = label
        else:
            new_labels = torch.cat((new_labels, label), 0)

    # check if the last is an annotated point
    if labels[-1][-1]*labels[-1][-2] == 0.0: # TODO: necessary?
        same_class_gt_boxes = box_labels[box_labels[:, 0] == prev_point[0]]
        if len(same_class_gt_boxes) == 0:
            same_class_gt_boxes = gt_labels_library[gt_labels_library[:, 0] == prev_point[0]]
            pgcp_box = same_class_gt_boxes[np.random.randint(same_class_gt_boxes.shape[0], size=1)].squeeze()
        else:
            dist_to_boxes = np.sqrt( ( prev_point[1]-same_class_gt_boxes[:, 1] )**2 + ( prev_point[2]-same_class_gt_boxes[:, 2] )**2 )
            pgcp_box = same_class_gt_boxes[np.argmin(dist_to_boxes)]
        new_labels = torch.cat((new_labels, torch.cat((prev_point[:3], pgcp_box[3:]), 0)), 0)

    return new_labels.view(-1,5), torch.cat((gt_labels_library, box_labels), 0)


if __name__ == '__main__':
    from model import YOPO_PT
    from shutil import rmtree

    teacher = YOPO_PT('/home/eiman/data/outputs/detect/yopo_pt_sc/weights/best.pt')
    input_train_dir_path = '/home/eiman/data/all/sc_10_YOPO_PT2/train/'
    prediction_output_path = '/home/eiman/data/outputs/detect/predict_loop/'
    labels_path_gt_library = '/home/eiman/data/all/sc_10_YOPO_PT/train/gt_labels.txt'
    gt_labels_library = torch.from_numpy(np.loadtxt(labels_path_gt_library))
    name = 'predict/teacher/'
    if os.path.exists(prediction_output_path + name):
        rmtree(prediction_output_path + name)
    teacher.predict(input_train_dir_path + 'images/', project=prediction_output_path, name=name, save=False, show_labels=False, save_txt=True, max_det=1500)
    s_preds = teacher.predictor.sIP['s_preds']
    sP_preds = teacher.predictor.sIP['sP_preds']

    prediction_list = sorted(os.listdir(prediction_output_path + name + 'labels/'))
    batch = []
    for i, label_name in enumerate(prediction_list):
        if label_name[-4:] == '.txt':
            pred_boxes = (torch.from_numpy(np.loadtxt(prediction_output_path + name + 'labels/' + label_name))).unsqueeze(0).to(device=s_preds[i].device)
            gts = torch.from_numpy(np.loadtxt(input_train_dir_path + 'labels/' + label_name)).to(device=s_preds[i].device)

            point_matched_image = point_matcher(pred_boxes, gts, s_preds, sP_preds)
            pgcp_labels, gt_labels_library = point_guided_copy_paste(point_matched_image, gt_labels_library)
            batch.append(pgcp_labels)

