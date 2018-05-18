import sys
sys.path.append('/home/arya/Documents/real_time_ID')
import numpy as np
import cv2
import global_var

def create_matrix(labels,predictions):
	img = cv2.imread('confusion_matrix(21x21).png')
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
	
	cv2.imwrite("confusion_matrix.png",image)
	
	image=cv2.imread("confusion_matrix.png")
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.line(image, (0,0), ((num_classes)*60,0), (0,0,0),3)
	cv2.line(image, (0,0), (0,(num_classes)*60), (0,0,0),3)
	for i in range(num_classes):
		cv2.line(image, (0,(i+1)*60), ((num_classes)*60,(i+1)*60), (0,0,0),3)
		cv2.line(image, ((i+1)*60,0), ((i+1)*60,(num_classes)*60), (0,0,0),3)
		for j in range(num_classes):
			cv2.putText(image,'{}'.format(matrix[i][j]/float(num_samples[i])),(j*60+20,i*60+30),font,0.5,(0,0,0),1,cv2.LINE_AA)
	cv2.imwrite("confusion_matrix.png",image)
	return matrix

# create_matrix([0,0,1,1],[1,1,1,0])