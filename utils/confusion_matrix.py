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

	image = np.zeros((60*num_classes,60*num_classes),dtype='uint8')

	size = image.shape[0]
	for i in range(size):
		for j in range(size):
			image[i][j] = matrix[i/60][j/60]
	print(matrix)
	cv2.imwrite("confusion_matrix.png",image)
	cv2.waitKey(0)
	return matrix