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

BUILD=build/examples/FRCNN/test_rpn.bin

time $BUILD --gpu $gpu \
    --model models/FRCNN/zf_rpn/test.prototxt \
    --weights models/FRCNN/zf_rpn_final.caffemodel \
    --default_c examples/FRCNN/config/voc_config.json \
    --image_root VOCdevkit/VOC2007/JPEGImages/ \
    --image_list examples/FRCNN/dataset/voc2007.test \
    --out_file examples/FRCNN/results/voc2007_test_${pid}.rpn

CAL_RECALL=examples/FRCNN/calculate_recall.py

time python $CAL_RECALL --gt examples/FRCNN/dataset/voc2007.test \
    --answer examples/FRCNN/results/voc2007_test_${pid}.rpn \
    --overlap 0.5
