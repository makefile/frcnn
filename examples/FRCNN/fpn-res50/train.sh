#!/usr/bin/env bash
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
use_snapshot=1
if [ $use_snapshot -gt 0 ];then
	for s in `ls exp/snapshot/fpn-res50-mix*.solverstate`;do
		ss=$s
	done
	opt_w="--snapshot $ss"
else
	#opt_w="--weights /home/s04/ry/imagenet_models/VGG16.v2.caffemodel"
    #opt_w="--weights /home/s03/fyk/ResNet-50-model-merge.caffemodel"
    opt_w="--weights /home/gpu/fyk/ResNet-50-model_merged.caffemodel"
fi
time GLOG_log_dir=exp/log \
$CAFFE train   \
    --gpu $gpu \
    --solver exp/fpn-res50/solver.proto \
    $opt_w

