import os
import numpy as np
import shutil

def create_bbox_library(gt_dir_path, gt_library_path, all_gt_in_files = False):
    ''' Runs through the dir and saves bounding boxes in a separeate .txt file.

    Args
        gt_dir_path:                path to the dir with all label-files in the dir.
        gt_library_path:            disered output path.
        all_gt_in_files (bool):     if True, it saves all labels in each image to a concatenated file.
    '''
    gt_library = open(gt_library_path, 'w')
    gt_file_list = os.listdir(gt_dir_path)

    if all_gt_in_files:
        for gt_file in gt_file_list:
            with open(gt_dir_path + gt_file,'r') as fd:
                shutil.copyfileobj(fd, gt_library)
    else:   # read line by line to check if box or point annotation
        for gt_file in gt_file_list:
            gt_array = (np.loadtxt(gt_dir_path + gt_file)).reshape(-1, 5) # might be slow, prehaps change to f.readlines()
            for label in gt_array:
                if label[-1]*label[-2] > 0.0:
                    gt_library.write(str(int(label[0])) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(label[4]) +'\n')
    gt_library.close()

if __name__ == '__main__':
    input_dir_path = '/home/eiman/data/all/multiclass_ps/ps_10/train/'
    output_path = input_dir_path + 'gt_labels.txt'
    create_bbox_library(input_dir_path + 'labels/', output_path)