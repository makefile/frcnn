#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model
set -e # exist right away if statement goes wrong and return false
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "use default GPU/0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
pid=$$
export GLOG_logtostderr=1 # only log to console
#export GLOG_DEST_PREFIX='exp/log/light-rfcn-res50'
export GLOG_DEST_PREFIX='' # disable logging
# tcmalloc for s03 machine
# export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
BUILD=build/examples/FRCNN/test_frcnn.bin
user=gpu
TYPE="light-rfcn-res50"
#iter=100000
iter=80000
config=exp/$TYPE/mix_config.json
test_proto=exp/$TYPE/test_merged-light-atrous.proto
normalized=0
if [ $normalized -gt 0 ];then
	weight=exp/weights/"$TYPE"-mix_iter_"$iter"_final.caffemodel
else
	weight=exp/snapshot/light-ohem-res50-mix_iter_"$iter".caffemodel
fi
echo $weight
if [ ! -f "$weight" ];then
	echo 'convert model'
	# convert model
#        export CAFFE_LAYER_PATH='build/lib'
	python examples/FRCNN/convert_model.py \
	    --model $test_proto \
	    --weights exp/snapshot/"$TYPE"-mix_iter_$iter.caffemodel \
	    --config $config \
	    --net_out $weight
fi
fid=${pid}
time $BUILD --gpu $gpu \
    --model $test_proto \
    --weights $weight \
    --default_c $config \
    --image_root /home/$user/fyk/RSI-mix/images/ \
    --image_list /home/$user/fyk/RSI-mix/RSI-mix-3.test \
    --out_file exp/results/RSI_test_"$TYPE"_$fid.frcnn \
    --max_per_image 100

CAL_AP="examples/FRCNN/calculate_voc_ap.py examples/FRCNN/calculate_recall.py"
for cal in $CAL_AP;do
python $cal --gt /home/$user/fyk/RSI-mix/RSI-mix-3.test \
    --answer exp/results/RSI_test_"$TYPE"_$fid.frcnn \
    --overlap 0.5
done

