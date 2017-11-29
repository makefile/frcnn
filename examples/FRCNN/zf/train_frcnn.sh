#!/usr/bin/env bash
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

#CAFFE=build/tools/caffe 
CAFFE=caffe-付永康 

time GLOG_log_dir=examples/FRCNN/log $CAFFE train   \
    --gpu $gpu \
    --solver models/FRCNN/zf/solver.prototxt \
    --weights models/FRCNN/imagenet_models/ZF.v2.caffemodel 

echo 'remember to convert model'
exit 0
time python examples/FRCNN/convert_model.py \
    --model models/FRCNN/zf/test.prototxt \
    --weights models/FRCNN/snapshot/zf_frcnn_end_to_end_iter_70000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/zf_faster_rcnn_final.caffemodel

