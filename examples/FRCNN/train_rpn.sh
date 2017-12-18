#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

CAFFE=build/tools/caffe

time GLOG_log_dir=examples/FRCNN/log $CAFFE train   \
    --gpu $gpu \
    --solver models/FRCNN/zf_rpn/solver.prototxt \
    --weights models/FRCNN/ZF.v2.caffemodel 

mv ./models/FRCNN/zf_rpn/zf_rpn_iter_70000.caffemodel ./models/FRCNN/zf_rpn_final.caffemodel
