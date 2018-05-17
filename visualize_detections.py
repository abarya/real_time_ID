import cv2
import os
import csv
import argparse
import global_var

parser = argparse.ArgumentParser(
    description='Visualization of detections in test videos')

parser.add_argument(
    'video_file',
    help='name of the video file to process')

parser.add_argument(
    '--video_path',
    help='path to directory of of video files',
    default='data')

parser.add_argument(
    '--detections_file_path',
    help='text file containing detections',
    default='bounding_boxes')


def _main(args):
    predictions=0
    if predictions==0:
        video_path = os.path.expanduser(args.video_path)
        video_file = os.path.expanduser(args.video_file)
        detections_file_path = os.path.expanduser(args.detections_file_path)

        detection_file = os.path.join(detections_file_path,video_file+'.out')

        with open(detection_file, "rb") as csvfile:
        	lines = csv.reader(csvfile)
        	lines = list(lines)
        	for i in range(len(lines)):
        		for j in range(5):
        			lines[i][j]=int(float(lines[i][j]))

        cap = cv2.VideoCapture(os.path.join(video_path,video_file+'.webm'))
        count=0
        i=0
        while(1):
			ret, frame = cap.read()
			if ret==0:
				print("video read completely\n")
				break

			while(i<len(lines) and count==lines[i][0]):
				cv2.rectangle(frame,(lines[i][1],lines[i][2]),(lines[i][3],lines[i][4]),(0,255,0),3)
				img = frame[lines[i][2]:lines[i][4],lines[i][1]:lines[i][3],:]
				i+=1  
			frame = cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))  
			cv2.imshow("f",frame)
			if(cv2.waitKey(1)==27):
				break
			count+=1
    else:
        print "hell"
        video_path = os.path.expanduser(args.video_path)
        video_file = os.path.expanduser(args.video_file)
        detections_file_path = os.path.expanduser(args.detections_file_path)

        detection_file = video_file+'.out'#os.path.join(detections_file_path,video_file+'.out')

        with open(detection_file, "rb") as csvfile:
            lines = csv.reader(csvfile)
            lines = list(lines)
            for i in range(len(lines)):
                for j in range(6):
                    lines[i][j]=int(float(lines[i][j]))

        cap = cv2.VideoCapture(os.path.join(video_path,video_file+'.webm'))
        count=0
        i=0
        correct=0
        while(1):
            ret, frame = cap.read()
            if ret==0:
                print("video read completely\n")
                break
            while(i<len(lines) and count==lines[i][0]):
                cv2.rectangle(frame,(lines[i][1],lines[i][2]),(lines[i][3],lines[i][4]),(0,255,0),1)
                cv2.putText(frame,global_var.CLASSES[lines[i][5]],(lines[i][1],lines[i][2]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                img = frame[lines[i][2]:lines[i][4],lines[i][1]:lines[i][3],:]
                if global_var.CLASSES[lines[i][5]]==video_file:
                    correct+=1
                else:
                    print global_var.CLASSES[lines[i][5]]
                i+=1  
            frame = cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))  
            cv2.imshow("f",frame)
            img_name = "{:05d}.jpg".format(count)
            print img_name
            cv2.imwrite("images_for_video/"+img_name,frame)
            if(cv2.waitKey(1)==27):
                break
            count+=1
        print "accuracy",float(correct)/i

if __name__ == '__main__':
    _main(parser.parse_args())
