#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "use default GPU/0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
pid=$$
BUILD=build/examples/FRCNN/test_frcnn.bin
#TYPE="res50" #vgg16/res101
#TYPE="vgg16"
TYPE="fpn-res50"
iter=70000
config=exp/$TYPE/mix_config.json
normalized=0
#weight=exp/weights/"$TYPE"-NWPU_faster_rcnn_iter_"$iter"_final.caffemodel
#weight=exp/weights/"$TYPE"-NWPU_roialign_iter_"$iter"_final.caffemodel
if [ $normalized -gt 0 ];then
	weight=exp/weights/"$TYPE"-mix_iter_"$iter"_final.caffemodel
else
	#weight=exp/snapshot/"$TYPE"-mix_align2_iter_"$iter".caffemodel
	weight=exp/snapshot/"$TYPE"-mix_iter_"$iter".caffemodel
fi
echo $weight
if [ ! -f "$weight" ];then
	echo 'convert model'
	# convert model
	export CAFFE_LAYER_PATH=`pwd`/.build_release/lib
        python examples/FRCNN/convert_model.py \
	    --model exp/fpn/res50-fpn-test.proto \
	    --weights exp/snapshot/"$TYPE"-mix_iter_$iter.caffemodel \
	    --config $config \
	    --net_out $weight
fi
#exit 0
fid=${pid}
#fid=19493
time $BUILD --gpu $gpu \
    --model exp/$TYPE/test_merged.proto \
    --weights $weight \
    --default_c $config \
    --image_root /home/gpu/fyk/RSI-mix/images/ \
    --image_list /home/gpu/fyk/RSI-mix/RSI-mix-3.test \
    --out_file exp/results/RSI_test_"$TYPE"_$fid.frcnn \
    --max_per_image 100

CAL_AP="examples/FRCNN/calculate_voc_ap.py"
for cal in $CAL_AP;do
python $cal --gt /home/gpu/fyk/RSI-mix/RSI-mix-3.test \
    --answer exp/results/RSI_test_"$TYPE"_$fid.frcnn \
    --overlap 0.5
done

