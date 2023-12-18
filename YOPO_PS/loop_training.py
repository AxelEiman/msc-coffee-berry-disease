import os
from model import YOPO_PS
import shutil

main_dir = '/home/eiman/data/all/multiclass_ss'
output_dir_path = '/home/eiman/data/outputs/detect/'

splits = os.listdir(main_dir)
splits.sort()

print(f'Training from data located in {main_dir}: \n{splits}\n')
for i, split in enumerate(splits):
    print(f'Training for split {i}/{len(splits)}: "{split}"')
    model = YOPO_PS('yolov8n.yaml')
    model.load('/home/eiman/CBD/yolov8n.pt')

    data = f'/home/eiman/data/all/multiclass_ss/{split}/multiclass.yaml'
    model.train(data=data, epochs=1500, imgsz=640, batch=2, max_det=1500, patience=100, project=output_dir_path, name=f'mc_{split}_640')

    # Move images and data.yaml file to next split directory
    if i < len(splits)-1:
        shutil.move(f'{main_dir}/{split}/train/images', f'{main_dir}/{splits[i+1]}/train/images')
        shutil.move(data, f'/home/eiman/data/all/multiclass_ss/{splits[i+1]}/multiclass.yaml')