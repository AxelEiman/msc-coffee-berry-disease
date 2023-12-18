import os
import numpy as np

def create_points_from_boxes(boxes):
	center_points = []
	for idx, box in enumerate(boxes):
		center_points.append([ int(box[0]), box[1], box[2], 0, 0]) # NOTE i think yolo boxes are cx,cy,w,h? changed to assume cxcy and added w,h=0. OLD: box[1] + box[3] / 2, box[2] + box[4] / 2 
	# TODO add noise to the center points
	return center_points

# TODO change to a function below

input_dir_path = '/home/eiman/data/resized/train_s/labels/'
output_dir_path = '/home/eiman/data/resized/train_sp/labels/'
fraction = 0.5	# in [0,1]: the fraction of boxes remaining 
isExist = os.path.exists(output_dir_path)
if not isExist:
   # Create a new directory since it does not exist
   os.makedirs(output_dir_path)

list_dir = os.listdir(input_dir_path)

for filename in list_dir:
	if filename[-4:] == '.txt': # So that no other file than a '.txt' file is read 
		boxes = np.loadtxt(input_dir_path + filename)
		
		n_points = round(len(boxes)*(1-fraction))								# number of points to create
		pt_idxs = np.random.choice(len(boxes), size=n_points, replace=False)	# idxs of boxes to turn into points
		points = create_points_from_boxes(boxes[pt_idxs, :])
		boxes_remaining = np.delete(boxes, pt_idxs, axis=0)						# pops boxes to keep
		boxes_and_points = np.vstack((boxes_remaining, points))					# Stacks boxes and points in array

		file = open(output_dir_path + 'points_from_' + filename, "w")
		for label in boxes_and_points:
			file.write(str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(label[4]) +'\n')
		file.close()
