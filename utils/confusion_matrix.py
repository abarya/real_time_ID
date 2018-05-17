import sys
sys.path.append('/home/arya/Documents/real_time_ID')
import numpy as np
import cv2
import global_var

def create_matrix(labels,predictions):
	num_classes = len(global_var.CLASSES)
	matrix = np.zeros((num_classes,num_classes),dtype='uint8')

	for i in range(len(labels)):
		matrix[labels[i]][predictions[i]]+=1

	matrix = matrix*10
	print(matrix)
	cv2.imwrite("image.png",matrix)
	cv2.waitKey(0)
	return matrix