#TYPE=$1
#if [ -z "$1" ];then
#	echo 'type: vgg16/res50/res101'
#	exit 0
#fi
iter=70000
python examples/FRCNN/convert_model.py \
    --model exp/test_merged.pt \
    --weights exp/snapshot/res50-NWPU_faster_rcnn_iter_$iter.caffemodel \
    --config exp/NWPU_config.json \
    --net_out exp/weights/res50-NWPU_faster_rcnn_iter_"$iter"_final.caffemodel
