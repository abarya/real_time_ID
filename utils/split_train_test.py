import sys
sys.path.append('..')
import os
import cv2
import argparse
import random
import global_var

parser = argparse.ArgumentParser(
    description='Splitting dataset into training and test set')

parser.add_argument(
	'dataset',
	help='which dataset to use, MAGI or ILID')

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
	dataset = os.path.expanduser(args.dataset)
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

	if dataset=="MAGI":
		sub_dirs = os.listdir(data_dir_path)
		sub_dirs = list(filter(lambda x: os.path.isdir(os.path.join(data_dir_path,x)),sub_dirs))

		for directory in sub_dirs:
			if directory in global_var.CLASSES:
				if not os.path.exists(os.path.join(train_dir,directory)):
					os.mkdir(os.path.join(train_dir,directory))
				if not os.path.exists(os.path.join(test_dir,directory)):
					os.mkdir(os.path.join(test_dir,directory))

				image_dir = os.path.join(data_dir_path,directory)
				image_files = os.listdir(image_dir)
				train_img_dir = os.path.join(train_dir,directory)
				test_img_dir = os.path.join(test_dir,directory)

				test_count=0
				train_count=0
				random_perm = [x for x in range(len(image_files))]
				random.seed(10101)  # Fixed seed for consistent colors across runs.
				random.shuffle(random_perm)  # Shuffle colors to decorrelate adjacent classes.
				for i in range(len(random_perm)):
					img = cv2.imread(os.path.join(image_dir,image_files[random_perm[i]]))
					if i<=int(split_ratio*len(random_perm)):
						img_name = "{:04d}.png".format(test_count)
						cv2.imwrite(os.path.join(test_img_dir,img_name),img)
						test_count+=1
					else:
						img_name = "{:04d}.png".format(train_count)
						cv2.imwrite(os.path.join(train_img_dir,img_name),img)
						train_count+=1
				print(train_count,test_count)
	elif dataset=='ILID':
		data_dir = '../i-LIDS-VID/sequences/'
		cam = ['cam1','cam2']
		num_classes = 319

		for i in range(1,num_classes+1):
			person_dir = "person{:03d}".format(i)

			if not os.path.exists(os.path.join("../i-LIDS-VID/sequences/cam1",person_dir)):
				continue
			if not os.path.exists(os.path.join(train_dir,person_dir)):
				os.mkdir(os.path.join(train_dir,person_dir))
			if not os.path.exists(os.path.join(test_dir,person_dir)):
				os.mkdir(os.path.join(test_dir,person_dir))

			train_img_dir = os.path.join(train_dir,person_dir)
			test_img_dir = os.path.join(test_dir,person_dir)

			test_count=0
			train_count=0

			for dir_name in cam:
				data_path = data_dir+dir_name
				image_dir = os.path.join(data_path,person_dir)
				image_list = os.listdir(image_dir)
			
				random_perm = [x for x in range(len(image_list))]
				random.seed()  # Fixed seed for consistent colors across runs.
				random.shuffle(random_perm)  # Shuffle colors to decorrelate adjacent classes.
				for i in range(len(random_perm)):
					img_path = os.path.join(image_dir,image_list[random_perm[i]])
					if os.path.splitext(img_path)[1]!='.png':
						continue
					img = cv2.imread(img_path)
					if i<=int(split_ratio*len(random_perm)):
						img_name = "{:04d}.png".format(test_count)
						cv2.imwrite(os.path.join(test_img_dir,img_name),img)
						test_count+=1
					else:
						img_name = "{:04d}.png".format(train_count)
						cv2.imwrite(os.path.join(train_img_dir,img_name),img)
						train_count+=1
			print(train_count,test_count)
	else:
		print("dataset name is invalid. Error: {}. Valid options are MAGI or ILID".format(dataset))

if __name__ == '__main__':
    _main(parser.parse_args())
