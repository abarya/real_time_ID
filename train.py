import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from sklearn.externals import joblib
from sklearn import svm
import argparse
import squeezenet
import global_var

parser = argparse.ArgumentParser(
    description='train model on dataset')

parser.add_argument(
    '--train_dir',
    help='path to directory containing training images',
    default='data/train')

parser.add_argument(
    '--test_dir',
    help='path to directory containing test images',
    default='data/test')

parser.add_argument(
	'--model_dir',
	help='directory where trained model will be saved',
	default='model')

def get_model():
	model =  squeezenet.SqueezeNet()
	layer_outputs = [layer.output for layer in model.layers]
	return Model(input=model.input, output=layer_outputs)

def _main(args):
	train_dir = os.path.expanduser(args.train_dir)
	test_dir = os.path.expanduser(args.test_dir)
	model_dir = os.path.expanduser(args.model_dir) #classifier
	conv_model = get_model() #squeezenet for feature extraction

	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

	training_data = []
	labels = []

	for name in global_var.CLASSES:
		path = os.path.join(train_dir,name)
		image_list = os.listdir(path)
		for img_name in image_list:
			img = image.load_img(os.path.join(path,img_name), target_size=(227, 227))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			features = conv_model.predict(x)
			training_data.append(features[-2].ravel())
			labels.append(global_var.classes_dict[name])
	labels = np.array(labels)
	training_data = np.array(training_data)
	print("Number of training samples is {}".format(len(training_data)))
	print(labels)
	svm_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
	svm_model.fit(training_data,labels)

	joblib.dump(svm_model, os.path.join(model_dir,'svm.pkl')) 

if __name__ == '__main__':
    _main(parser.parse_args())