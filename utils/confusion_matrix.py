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
		num_samples[labels[i]]+=1

	size = image.shape[0]
	for i in range(size):
		for j in range(size):
			image[i][j] = int((matrix[int(i/60)][int(j/60)]/float(num_samples[int(i/60)]))*255)
	print(matrix)
	image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
	cv2.imshow("image",image)
	cv2.waitKey(0)

	for i in range(num_classes):
		for j in range(num_classes):
			cv2.putText(image,'{}'.format(matrix[i][j]/float(num_samples[i])),(j*60+20,i*60+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
	cv2.imwrite("confusion_matrix.png",image)
	return matrix

# create_matrix([1,1,1,1,1,1,1,1,2,2,2,2,2],[1,1,1,1,1,1,2,2,2,2,2,2,2])