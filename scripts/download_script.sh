#!/bin/bash
wget https://pjreddie.com/media/files/yolov2.weights
./yolo.py yolov2.cfg yolov2.weights model_data/yolo.h5
