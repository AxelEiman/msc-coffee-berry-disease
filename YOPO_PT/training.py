from model import YOPO_PT
from get_gtbox_library import create_bbox_library


output_dir = '/home/aleksispi/Projects/master-theses/cbd_2023/runs'

if __name__ == '__main__':
    model = YOPO_PT('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/YOPO_PT/yopo.yaml') # YAML file specific for MPT
    model.load('/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/yolov8n.pt')
    data = '/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/coco.yaml' # YAML file with locations of data
    create_bbox_library(f'/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/labels/', 
                        f'/home/aleksispi/Projects/master-theses/cbd_2023/CBD/data/train2017/labels/gt_labels.txt')
    
    model.train(data=data, epochs=11, imgsz=640, batch=32, max_det=1500, patience=100, close_mosaic=0, project=output_dir, name='pt_coco_50')