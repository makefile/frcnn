time python examples/FRCNN/convert_model.py \
    --model matlab/FRCNN/For_LOC/two/googlenet_v1/test.proto \
    --weights matlab/FRCNN/For_LOC/two/googlenet_v1/snapshot/googlenet_v1_faster_rcnn_iter_90000.caffemodel \
    --config matlab/FRCNN/For_LOC/two/trecvid.json \
    --net_out matlab/FRCNN/For_LOC/two/googlenet_v1/googlenet_v1_faster_rcnn_final.caffemodel

time python examples/FRCNN/convert_model.py \
    --model matlab/FRCNN/For_LOC/two/res101/test.proto \
    --weights matlab/FRCNN/For_LOC/two/res101/snapshot/res101_faster_rcnn_iter_90000.caffemodel \
    --config matlab/FRCNN/For_LOC/two/trecvid.json \
    --net_out matlab/FRCNN/For_LOC/two/res101/res101_faster_rcnn_final.caffemodel

time python examples/FRCNN/convert_model.py \
    --model matlab/FRCNN/For_LOC/two/vgg19/test.proto \
    --weights matlab/FRCNN/For_LOC/two/vgg19/snapshot/vgg19_faster_rcnn_iter_90000.caffemodel \
    --config matlab/FRCNN/For_LOC/two/trecvid.json \
    --net_out matlab/FRCNN/For_LOC/two/vgg19/vgg19_faster_rcnn_final.caffemodel

time python examples/FRCNN/convert_model.py \
    --model matlab/FRCNN/For_LOC/two/res152/test.proto \
    --weights matlab/FRCNN/For_LOC/two/res152/snapshot/res152_faster_rcnn_iter_90000.caffemodel \
    --config matlab/FRCNN/For_LOC/two/trecvid.json \
    --net_out matlab/FRCNN/For_LOC/two/res152/res152_faster_rcnn_final.caffemodel

