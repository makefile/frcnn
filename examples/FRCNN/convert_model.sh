TYPE=$1
if [ -z "$1" ];then
	echo 'type: vgg16/res50/res101'
	exit 0
fi
python examples/FRCNN/convert_model.py \
    --model models/FRCNN/$TYPE/test.proto \
    --weights models/FRCNN/snapshot/"$TYPE"_faster_rcnn_iter_70000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/"$TYPE"_faster_rcnn_final.caffemodel
