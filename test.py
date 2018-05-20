import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from sklearn.externals import joblib
import argparse
import squeezenet
import global_var
from utils import confusion_matrix 
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
			test_data.append(features[-2].ravel().reshape(1,-1))
			labels.append(i)
	labels = np.array(labels)
	test_data = np.array(test_data)

	svm = joblib.load(model_path)

	count=0 # number of correct predictions
	rank = [1,2,5,8,10,12,15,18,20,30,40,50,60,70,80,90,100]
	rank_accuracy = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	predictions = []
	for i in range(len(labels)):
		pred = svm.predict(test_data[i])
		probs = svm.predict_proba(test_data[i])
		probs=probs.ravel()
		probs = [b[0] for b in sorted(enumerate(probs),key=lambda i:i[1],reverse=True)]
		for j in range(len(rank)):
			if rank[j]<=len(probs) and (labels[i] in probs[:rank[j]]):
				rank_accuracy[j]+=1

		print(dir_list[int(pred)],dir_list[int(labels[i])])
		predictions.append(pred)
	
	rank_accuracy = [round(100*x/float(len(labels))) for x in rank_accuracy]
	print("ranks accuracy",rank_accuracy)

	confusion_matrix.create_matrix(labels,predictions)
	accuracy = rank_accuracy[0]/float(len(labels))
	print("Number of images in test set is {}.\nAccuracy of model is {}%.".format(len(labels),accuracy*100))

if __name__ == '__main__':
    _main(parser.parse_args())