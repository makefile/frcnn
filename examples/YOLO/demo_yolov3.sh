#!/usr/bin/env sh
set -e
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

BUILD=build/examples/YOLO/demo_yolov3.bin

GLOG_logtostderr=1 $BUILD --gpu $gpu \
       --model models/YOLO/yolov3.proto \
       --weights models/YOLO/yolov3.caffemodel \
       --classes 80 \
       --image_dir examples/YOLO/images/  \
       --out_dir examples/YOLO/results/
