from model import YOPO_PS
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Test for debugging purposes
output_dir = '/home/aleksispi/Projects/master-theses/cbd_2023/runs'

if __name__ == '__main__':
    hej = YOPO_PS('yolov8n.yaml')
    hej.load('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/yolov8n.pt')
    data = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/coco.yaml' # YAML file with locations of data
    hej.train(data=data, epochs=11, imgsz=640, batch=32, max_det=1500, patience=100, close_mosaic=0, project=output_dir, name=f'ss_coco_50')