#from model import YOPO, DetectionPointModel, DetectionPointPredictor, v8DetectionPointLoss, PointTaskAlignedAssigner, BboxPointLoss
from model import YOPO_PT#, DetectionPointModel
import os
import shutil


#model = YOPO_PS('/home/eiman/data/outputs/detect/sc_ss_50_new_mAP/weights/best.pt')#/home/eiman/data/outputs/detect/sc_ss_02_new_mAP/weights/best.pt')
#model.val(batch=4, split='test', imgsz=1280, max_det=1500)

main_dir = '/home/aleksispi/Projects/master-theses/cbd_2023/runs/sc_pt'
output_dir = '/home/aleksispi/Projects/master-theses/cbd_2023/runs/single_class_tests'

splits = os.listdir(main_dir)
splits.sort()

for i, split in enumerate(splits):
    print(f'Validating {split}')
    model = YOPO_PT(f'{main_dir}/{split}/weights/best.pt')
    
    data = f'/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/single_class_ps/sc_50/single_class.yaml'
    model.val(batch=4, data=data, split='test', imgsz=1280, max_det=1500, project=output_dir, name=split)

    #if i < len(splits)-1:
    #    shutil.move(data, f'/home/eiman/data/all/multiclass_ps/ps{splits[i+1]}/multiclass.yaml')