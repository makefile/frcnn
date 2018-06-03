#!/usr/bin/env bash
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
set -e
if [ ! -n "$1" ] ;then
    echo "use default GPU 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
PATH=`pwd`:$PATH
export OMP_NUM_THREADS=4
#export LD_LIBRARY_PATH=~/fyk/protobuf-3.1.0/lib/:$LD_LIBRARY_PATH
CAFFE=build/tools/caffe 
use_snapshot=0
if [ $use_snapshot -gt 0 ];then
	for s in `ls exp/snapshot/rfcn-res50-mix*.solverstate`;do
		ss=$s
	done
	opt_w="--snapshot $ss"
else
        opt_w="--weights /home/gpu/fyk/ResNet-50-model_merged.caffemodel"
fi
time GLOG_log_dir=exp/log \
$CAFFE train   \
    --gpu $gpu \
    --solver exp/rfcn-res50/solver.proto \
    $opt_w
