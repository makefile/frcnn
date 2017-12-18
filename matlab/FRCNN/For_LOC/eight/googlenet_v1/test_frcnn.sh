#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
pid=$$
BUILD=build/examples/FRCNN/loc_test_frcnn.bin

time $BUILD --gpu $gpu \
    --model matlab/FRCNN/For_LOC/eight/googlenet_v1/test.proto \
    --weights matlab/FRCNN/For_LOC/eight/googlenet_v1/googlenet_v1_faster_rcnn_final.caffemodel \
    --default_c matlab/FRCNN/For_LOC/eight/trecvid.json \
    --image_root matlab/FRCNN/For_LOC/LOC/filtered \
    --image_list matlab/FRCNN/For_LOC/dataset/test.list \
    --out_file matlab/FRCNN/For_LOC/eight/googlenet_v1/out/8_test_list_googlenet_v1_${pid}.frcnn \
    --max_per_image 100
