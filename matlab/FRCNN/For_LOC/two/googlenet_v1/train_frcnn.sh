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

time GLOG_log_dir=matlab/FRCNN/For_LOC/two/googlenet_v1/log $CAFFE train   \
    --gpu $gpu \
    --solver matlab/FRCNN/For_LOC/two/googlenet_v1/solver.proto \
    --weights matlab/FRCNN/For_LOC/two/googlenet_v1/bvlc_googlenet.caffemodel

time python examples/FRCNN/convert_model.py \
    --model matlab/FRCNN/For_LOC/two/googlenet_v1/test.proto \
    --weights matlab/FRCNN/For_LOC/two/googlenet_v1/snapshot/googlenet_v1_faster_rcnn_iter_90000.caffemodel \
    --config matlab/FRCNN/For_LOC/two/trecvid.json \
    --net_out matlab/FRCNN/For_LOC/two/googlenet_v1/googlenet_v1_faster_rcnn_final.caffemodel
