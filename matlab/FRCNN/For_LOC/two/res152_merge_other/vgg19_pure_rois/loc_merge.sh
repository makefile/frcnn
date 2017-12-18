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
BUILD=build/examples/FRCNN/loc_merge_frcnn.bin

time $BUILD --gpu $gpu \
    --model matlab/FRCNN/For_LOC/two/res152_merge_other/vgg19_pure_rois/test.proto \
    --weights matlab/FRCNN/For_LOC/two/vgg19/vgg19_faster_rcnn_final.caffemodel \
    --default_c matlab/FRCNN/For_LOC/two/trecvid.json \
    --image_root matlab/FRCNN/For_LOC/LOC/filtered \
    --image_list matlab/FRCNN/For_LOC/LOC/LOC_OUT/TWO_${2}_res152*.frcnn \
    --out_file matlab/FRCNN/For_LOC/two/res152_merge_other/vgg19_pure_rois/out/TWO_${2}_vgg19_$$.score
