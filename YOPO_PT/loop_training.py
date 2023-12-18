import os
import numpy as np
from model import YOPO_PT
from get_gtbox_library import create_bbox_library
import shutil


main_dir = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/with_points'
output_dir_path = '/home/aleksispi/Projects/master-theses/cbd_2023/runs'
data = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/coco.yaml'
labels_path = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/labels/'
gt_labels_path = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/gt_labels.txt'

splits = os.listdir(main_dir)
splits.sort()
start_idx = 1

print(f'Training from data located in {main_dir}: \n{splits}\n')
for i, split in enumerate(splits[start_idx:]):
    if split[:3] == 'pts':   # To avoid reading other files, e.g. ".DS_STORE"
        print(f'Training for split {i}/{len(splits)}: "{split}"')
        model = YOPO_PT('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/YOPO_PT/yopo.yaml')#/home/eiman/data/all/yopov8n.yaml
        model.load('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/yolov8n.pt')

        shutil.move(f'{main_dir}/{split}/labels', '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/')  # Move from labels dir to running dir
        create_bbox_library(f'{labels_path}', f'{gt_labels_path}')
        model.train(data=data, epochs=11, imgsz=640, batch=32, max_det=300, patience=100, close_mosaic=0, project=output_dir_path, name=f'pt_coco_{split[4:]}')

        os.remove('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/labels.cache')
        os.remove(f'{gt_labels_path}')
        shutil.move(f'{labels_path[:-1]}', f'{main_dir}/{split}/')  # Move labels back to the original dir