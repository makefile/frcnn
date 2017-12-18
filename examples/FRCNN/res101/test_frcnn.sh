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
BUILD=build/examples/FRCNN/test_frcnn.bin

$BUILD --gpu $gpu \
    --model models/FRCNN/res101/test.proto \
    --weights models/FRCNN/res101_faster_rcnn_final.caffemodel \
    --default_c examples/FRCNN/config/voc_config.json \
    --image_root VOCdevkit/VOC2007/JPEGImages/ \
    --image_list examples/FRCNN/dataset/voc2007.test \
    --out_file examples/FRCNN/results/voc2007_test_res101_${pid}.frcnn \
    --max_per_image 100

CAL_AP=examples/FRCNN/calculate_voc_ap.py

python $CAL_AP --gt examples/FRCNN/dataset/voc2007.test \
    --answer examples/FRCNN/results/voc2007_test_res101_${pid}.frcnn \
    --overlap 0.5
