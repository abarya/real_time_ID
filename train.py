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
    'dataset',
    help='dataset being used')

parser.add_argument(
    'extract_features',
    help='IS it needed to extract features?, boolean')

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
	dataset = os.path.expanduser(args.dataset)
	extract_features = args.extract_features
	
	trainfeatures_file_name = 'features/'+dataset+'_train_features.out' 
	trainlabels_file_name = 'features/'+dataset+'_train_labels.out'

	testfeatures_file_name = 'features/'+dataset+'_test_features.out' 
	testlabels_file_name = 'features/'+dataset+'_test_labels.out'

	train_data = []
	train_labels = []

	test_data = []
	test_labels = []

	if extract_features==True:
		if not os.path.exists('features'):
			os.mkdir('features')

		conv_model = get_model() #squeezenet for feature extraction

		if not os.path.exists(model_dir):
			os.mkdir(model_dir)

		dir_list = os.listdir(train_dir)
		training_data = []
		labels = []	

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
		
		
		for i in range(len(labels)):
			train_data.append(training_data[random_perm[i]])
			train_labels.append(labels[random_perm[i]])

		train_data = np.array(train_data)
		train_labels = np.array(train_labels)

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
				test_labels.append(i)
		test_labels = np.array(test_labels)
		test_data = np.array(test_data)
	else:
		print("features not extracted")
		
	# np.savetxt(trainfeatures_file_name, train_data, delimiter=',')
	# np.savetxt(trainlabels_file_name, train_labels, delimiter=',')
	# np.savetxt(testfeatures_file_name, test_data, delimiter=',')
	# np.savetxt(testlabels_file_name, test_labels, delimiter=',')

	print("Number of training samples is {}".format(len(train_data)))
	print(train_labels)
	
	
	svm_model = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	
	svm_model.fit(train_data,train_labels)#,classes=np.unique(train_labels))
	
	count=0
	for i in range(len(test_labels)):
		pred = svm_model.predict(test_data[i].reshape(1,-1))
		
		if test_labels[i]==pred:
			count+=1

	print("accuracy on test set is",100*count/float(len(labels)))
	count=0
	for i in range(len(train_labels)):
		pred = svm_model.predict(train_data[i].reshape(1,-1))
		
		if train_labels[i]==pred:
			count+=1
	print("accuracy on training set is",100*count/float(len(train_labels)))
	
	
	joblib.dump(svm_model, os.path.join(model_dir,'svm.pkl')) 

if __name__ == '__main__':
    _main(parser.parse_args())