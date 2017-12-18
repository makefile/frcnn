#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
if [ ! -n "$1" ] ;then
    echo "use default GPU 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
PATH=.:$PATH
#CAFFE=build/tools/caffe 
CAFFE=R-caffe
iter=60000
time GLOG_log_dir=exp/log $CAFFE train   \
    --gpu $gpu \
    --solver exp/vgg16/solver.proto \
    --weights ~/data/fyk/caffe-faster-rcnn/models/FRCNN/imagenet_models/VGG16.v2.caffemodel
    #--snapshot exp/snapshot/r-vgg16_frcnn_iter_$iter.solverstate

echo 'remember to convert_model of bbox after training'
exit 0
time python examples/FRCNN/convert_model.py \
    --model models/FRCNN/res50/test.proto \
    --weights models/FRCNN/snapshot/res50_faster_rcnn_iter_180000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/res50_faster_rcnn_final.caffemodel
