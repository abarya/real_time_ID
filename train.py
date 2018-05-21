import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
import argparse
import random
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
	dir_list = os.listdir(train_dir)

	for i,name in enumerate(dir_list):
		print("{:.2f} completed".format(i/float(len(dir_list))))
		path = os.path.join(train_dir,name)
		image_list = os.listdir(path)
		for img_name in image_list:
			img = image.load_img(os.path.join(path,img_name), target_size=(227, 227))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			features = conv_model.predict(x)
			training_data.append(features[-2].ravel())
			labels.append(i)

	random_perm = [x for x in range(len(labels))]
	random.shuffle(random_perm)
	
	train_data = []
	train_labels = []
	for i in range(len(labels)):
		train_data.append(training_data[random_perm[i]])
		train_labels.append(labels[random_perm[i]])
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)

	test_data = []
	labels = []
	dir_list = os.listdir(test_dir)

	for i,name in enumerate(dir_list):
		path = os.path.join(test_dir,name)
		image_list = os.listdir(path)
		for img_name in image_list:
			img = image.load_img(os.path.join(path,img_name), target_size=(227, 227))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			features = conv_model.predict(x)
			test_data.append(features[-2].ravel())
			labels.append(i)
	test_labels = np.array(labels)
	test_data = np.array(test_data)

	#dimensionality reduction
	# pca = PCA(n_components=100)
	# pca.fit(train_data)
	# train_data = pca.transform(train_data)
	# filename = 'pca_model.sav'
	# pickle.dump(pca, open(filename, 'wb')) #save pca model
	# print("variance in pca{}".format(pca.explained_variance_ratio_))  

	print("Number of training samples is {}".format(len(training_data)))
	print(train_labels)
	from sklearn.linear_model import SGDClassifier
	# svm_model = SGDClassifier(loss="hinge", penalty="l2",max_iter=1000)
	svm_model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	half = int(len(train_labels)/2)
	svm_model.partial_fit(train_data[:half][:],train_labels[:half],classes=np.unique(train_labels))
	print("accuracy is",svm_model.score(test_data,test_labels))
	svm_model.partial_fit(train_data[half:][:],train_labels[half:])
	print("accuracy is",svm_model.score(test_data,test_labels))
	joblib.dump(svm_model, os.path.join(model_dir,'svm.pkl')) 

if __name__ == '__main__':
    _main(parser.parse_args())