import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from sklearn.externals import joblib
import argparse
import squeezenet
import global_var

parser = argparse.ArgumentParser(
    description='test model on dataset')

parser.add_argument(
    '--test_dir',
    help='path to directory containing test images',
    default='data/test')

parser.add_argument(
	'--model_path',
	help='path from where trained model will be loaded',
	default='model/svm.pkl')

def get_model():
	model =  squeezenet.SqueezeNet()
	layer_outputs = [layer.output for layer in model.layers]
	return Model(input=model.input, output=layer_outputs)

def _main(args):
	test_dir = os.path.expanduser(args.test_dir)
	model_path = os.path.expanduser(args.model_path) #classifier
	conv_model = get_model() #squeezenet for feature extraction
	test_data = []
	labels = []

	for name in global_var.CLASSES:
		path = os.path.join(test_dir,name)
		image_list = os.listdir(path)
		for img_name in image_list:
			img = image.load_img(os.path.join(path,img_name), target_size=(227, 227))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			features = conv_model.predict(x)
			test_data.append(features[-2].ravel().reshape(1,-1))
			labels.append(global_var.classes_dict[name])
	labels = np.array(labels)
	test_data = np.array(test_data)

	svm = joblib.load(model_path)

	count=0 # number of correct predictions
	for i in range(len(labels)):
		pred = svm.predict(test_data[i])
		print(global_var.CLASSES[pred],global_var.CLASSES[labels[i]])
		if pred==labels[i]:
			count+=1

	accuracy = count/float(len(labels))
	print("Number of images in test set is {}.\nAccuracy of model is {}%.".format(len(labels),accuracy*100))

if __name__ == '__main__':
    _main(parser.parse_args())