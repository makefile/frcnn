python examples/FRCNN/convert_model.py \
    --model models/FRCNN/zf/test.prototxt \
    --weights models/FRCNN/snapshot/zf_frcnn_end_to_end_iter_70000.caffemodel \
    --config examples/FRCNN/config/voc_config.json \
    --net_out models/FRCNN/zf_faster_rcnn_final.caffemodel
