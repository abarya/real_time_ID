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
	
	num_samples = np.zeros(num_classes)
	for i in range(len(labels)):
		num_classes[labels[i]]+=1

	size = image.shape[0]
	for i in range(size):
		for j in range(size):
			image[i][j] = int((matrix[int(i/60)][int(j/60)]/float(num_classes[int(i/60)]))*255)
	print(matrix)
	cv2.imwrite("confusion_matrix.png",image)
	cv2.waitKey(0)
	return matrix

create_matrix([1,1,1,1,1,1,1,1,2,2,2,2,2],[1,1,1,1,1,1,2,2,2,2,2,2,2])