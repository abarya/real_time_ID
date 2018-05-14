import sys
sys.path.append('/home/arya/Documents/real_time_ID')
import os
import cv2
import argparse
import random
import global_var

parser = argparse.ArgumentParser(
    description='Splitting dataset into training and test set')

parser.add_argument(
    '--data_dir',
    help='path to directory of of video files',
    default='data')

parser.add_argument(
	'--split_ratio',
	help='train-test split ratio, defaults to 0.2',
	type=float,
	default='0.2')

def _main(args):
	split_ratio = args.split_ratio
	data_dir = os.path.expanduser(args.data_dir)
	parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
	data_dir_path = os.path.join(parent_dir,data_dir)
	train_dir = os.path.join(data_dir_path,'train')
	test_dir = os.path.join(data_dir_path,'test')

	if not os.path.exists(train_dir):
		os.mkdir(train_dir)

	if not os.path.exists(test_dir):
		os.mkdir(test_dir)

	sub_dirs = os.listdir(data_dir_path)
	sub_dirs = list(filter(lambda x: os.path.isdir(os.path.join(data_dir_path,x)),sub_dirs))

	for directory in sub_dirs:
		if directory in global_var.CLASSES:
			if not os.path.exists(os.path.join(train_dir,directory)):
				os.mkdir(os.path.join(train_dir,directory))
			if not os.path.exists(os.path.join(test_dir,directory)):
				os.mkdir(os.path.join(test_dir,directory))
				random.seed()

			image_dir = os.path.join(data_dir_path,directory)
			image_files = os.listdir(image_dir)
			train_img_dir = os.path.join(train_dir,directory)
			test_img_dir = os.path.join(test_dir,directory)

			test_count=0
			train_count=0
			for image_name in image_files:
				img = cv2.imread(os.path.join(image_dir,image_name))
				if random.random()<split_ratio:
					img_name = "{:04d}.png".format(test_count)
					cv2.imwrite(os.path.join(test_img_dir,img_name),img)
					test_count+=1
				else:
					img_name = "{:04d}.png".format(train_count)
					cv2.imwrite(os.path.join(train_img_dir,img_name),img)
					train_count+=1
			print(train_count,test_count)
if __name__ == '__main__':
    _main(parser.parse_args())
