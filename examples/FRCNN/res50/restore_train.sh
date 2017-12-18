#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "restore from snapshot: res50_faster_rcnn_iter_$1"
    iter=$1
else
    echo 'param: iter'
    exit 0
fi

CAFFE=build/tools/caffe 

time GLOG_log_dir=examples/FRCNN/log $CAFFE train   \
    --gpu $gpu \
    --solver models/FRCNN/res50/solver.proto \
    --snapshot models/FRCNN/res50/res50_faster_rcnn_iter_40000.solverstate \
#    --weights models/FRCNN/ResNet-50-model.caffemodel
echo 'remember to convert_model'
exit 0
time python examples/FRCNN/convert_model.py \
    --model models/FRCNN/res50/test.proto \
    --weights models/FRCNN/snapshot/res50_faster_rcnn_iter_180000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/res50_faster_rcnn_final.caffemodel
