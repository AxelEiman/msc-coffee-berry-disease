from model import YOPO_PS#, DetectionPointModel
import os
import shutil


#model = YOPO_PS('/home/eiman/data/outputs/detect/sc_ss_50_new_mAP/weights/best.pt')#/home/eiman/data/outputs/detect/sc_ss_02_new_mAP/weights/best.pt')
#model.val(batch=4, split='test', imgsz=1280, max_det=1500)

main_dir = '/home/eiman/data/outputs/detect/multiclass_ps'
output_dir = '/home/eiman/data/outputs/detect/multiclass_tests'

splits = os.listdir(main_dir)
splits.sort()

for i, split in enumerate(splits):
    print(f'Validating {split}')
    model = YOPO_PS(f'{main_dir}/{split}/weights/best.pt')

    data = f'/home/eiman/data/all/multiclass_ps/ps_50/multiclass.yaml'
    model.val(batch=4, data=data, split='test', imgsz=1280, max_det=1500, project=output_dir, name=split)

    #if i < len(splits)-1:
    #    shutil.move(data, f'/home/eiman/data/all/multiclass_ss/{splits[i+1]}/multiclass.yaml')