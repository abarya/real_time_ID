import cv2
import os
import csv
import argparse

parser = argparse.ArgumentParser(
    description='Visualization of detections in test videos')

parser.add_argument(
    '--video_path',
    help='path to directory of of video files',
    default='../data')

parser.add_argument(
    '--detections_file_path',
    help='text file containing detections',
    default='bounding_boxes')


def _main(args):
    video_path = os.path.expanduser(args.video_path)
    detections_file_path = os.path.expanduser(args.detections_file_path)

    for video_file in os.listdir(video_path):
        if os.path.isdir(os.path.join(video_path,video_file)):
            continue
        print os.path.join(os.getcwd(),video_file),"arya",os.path.isdir(os.path.join(os.getcwd(),video_file))
        result_dir = os.path.join('../data',os.path.splitext(video_file)[0])
        detection_file = os.path.join(detections_file_path,os.path.splitext(video_file)[0]+'.out')
        detection_file = '../'+detection_file

        with open(detection_file, "rb") as csvfile:
        	lines = csv.reader(csvfile)
        	lines = list(lines)
        	for i in range(len(lines)):
        		for j in range(5):
        			lines[i][j]=int(float(lines[i][j]))

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        cap = cv2.VideoCapture(os.path.join(video_path,video_file))
        count=0
        i=0
        img_count = 0
        while(1):
    		ret, frame = cap.read()
    		if ret==0:
    			print("video read completely\n")
    			break

    		while(i<len(lines) and count==lines[i][0]):
    			#save detections after 20 frames
    			if count%20==0:
    				img = frame[lines[i][2]:lines[i][4],lines[i][1]:lines[i][3],:]
    				img_name = "{:04d}.png".format(img_count)
    				cv2.imwrite(os.path.join(result_dir,img_name),img)
    				img_count+=1
    			cv2.rectangle(frame,(lines[i][1],lines[i][2]),(lines[i][3],lines[i][4]),(0,255,0),3)
    			i+=1

    		frame = cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))  
    		cv2.imshow("f",frame)
    		if(cv2.waitKey(1)==27):
    			break
    		count+=1
        cap.release()


if __name__ == '__main__':
    _main(parser.parse_args())
