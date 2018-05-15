import os
import numpy as np
import cv2
import time
import csv
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from sklearn.externals import joblib
from PIL import Image
import argparse
import squeezenet
import global_var

parser = argparse.ArgumentParser(
    description='person identification in a video')

parser.add_argument(
    '--video_dir',
    help='path to the directory in which video is located',
    default='data')

parser.add_argument(
    'video',
    help='name of the video to be tested',
    default='data/test_video.webm')

parser.add_argument(
    '--detections_dir',
    help='path to the directory in which detections are saved',
    default='bounding_boxes')

parser.add_argument(
	'--model_path',
	help='path from where trained model will be loaded',
	default='model/svm.pkl')

parser.add_argument(
	'--result_dir',
	help='path from where trained model will be loaded',
	default='results')

def get_model():
	model =  squeezenet.SqueezeNet()
	layer_outputs = [layer.output for layer in model.layers]
	return Model(input=model.input, output=layer_outputs)

def _main(args):
	video_dir = os.path.expanduser(args.video_dir)
	detections_dir = os.path.expanduser(args.detections_dir)
	video_name = os.path.expanduser(args.video)
	result_dir = os.path.expanduser(args.result_dir)
	model_path = os.path.expanduser(args.model_path) #classifier

	conv_model = get_model() #squeezenet for feature extraction
	svm = joblib.load(model_path)

	detection_file = os.path.join(detections_dir,video_name+'.out')

	with open(detection_file, "rt") as csvfile:
		lines = csv.reader(csvfile)
		lines = list(lines)
		for i in range(len(lines)):
			for j in range(5):
				lines[i][j]=int(float(lines[i][j]))
	start = time.time()
	cap = cv2.VideoCapture(os.path.join(video_dir,video_name+'.webm'))
	count=0
	i=0
	while(1):
		ret, frame = cap.read()
		if ret==0:
			print("video processed completely in {}s\n".format(time.time()-start))
			break

		while(i<len(lines) and count==lines[i][0]):
			cv2.rectangle(frame,(lines[i][1],lines[i][2]),(lines[i][3],lines[i][4]),(0,255,0),3)
			img = frame[lines[i][2]:lines[i][4],lines[i][1]:lines[i][3],:]
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(image)
			image = image.resize(
					tuple((227,227)), Image.NEAREST)
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = conv_model.predict(x)
			pred = svm.predict(features[-2].ravel().reshape(1,-1))
			lines[i].append(pred)
			i+=1  
		count+=1

	if not os.path.exists(result_dir):
		os.mkdir(result_dir)

	bboxesPredictionArray = np.array(lines)
	np.savetxt(result_dir+video+'.out', bboxesPredictionArray, delimiter=',')

if __name__ == '__main__':
    _main(parser.parse_args())
