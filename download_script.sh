#!/bin/bash
fileid="1bEBwy9DA3ddSGctgWq3mhjidbL9bmDIQ"
filename="yolov2.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
