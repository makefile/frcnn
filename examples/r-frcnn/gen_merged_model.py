#encoding=utf-8
'''
此脚本用于合并batchnorm层,减少内存和计算时间.
训练完毕之后进行inference的时候可将全部的Conv-BN-Scale合并成一个Conv;
训练时可将被freeze的层合并,如果conv没被freeze,则仅合并BN-Scale
a tool to merge 'Conv-BN-Scale' into a single 'Conv' layer.
https://github.com/sanghoon/pva-faster-rcnn/blob/master/tools/gen_merged_model.py
也可参考这里的:https://github.com/NHZlX/Merge_bn_Caffe (注意只拷贝了卷积层和全连接层的参数,其余没拷贝,
当然像ReLU这类的层没有权重所以无需拷贝)
'''
# import _init_paths
import numpy as np
import os
import os.path as osp
from argparse import ArgumentParser
import google.protobuf as pb
import google.protobuf.text_format
import sys
caffe_path = './python/'
if caffe_path not in sys.path:
    sys.path.insert(0, caffe_path)
import caffe


def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights, only_merge_freezed, is_train_proto=True):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)

    for i, layer in enumerate(model.layer):
        if layer.type == 'Convolution': # or layer.type == 'Scale':# Add bias layer if needed
            #print(i,layer.name)
            if only_merge_freezed and layer.param[0].lr_mult == 1:  # 没有被freezed,训练时不能进行合并
                continue
            if layer.convolution_param.bias_term == False:
                layer.convolution_param.bias_term = True
                layer.convolution_param.bias_filler.type = 'constant'
                layer.convolution_param.bias_filler.value = 0.0

    with open(dst_model, 'w') as f:
        f.write(pb.text_format.MessageToString(model))

    caffe.set_mode_cpu()
    #phase = caffe.TEST
    if is_train_proto:
        phase = caffe.TRAIN
    else:
        phase = caffe.TEST
    net_src = caffe.Net(src_model, src_weights, phase)
    net_dst = caffe.Net(dst_model, phase)
    for key in net_src.params.keys():
        for i in range(len(net_src.params[key])):
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]

    if dst_weights is not None:
        # Store params
        pass

    return net_dst, model


def merge_conv_and_bn(net, i_conv, i_bn, i_scale, conv_freezed):
    # This is based on Kyeheyon's work
    assert(i_conv != None)
    assert(i_bn != None)

    def copy_double(data):
        return np.array(data, copy=True, dtype=np.double)

    key_conv = net._layer_names[i_conv]
    key_bn = net._layer_names[i_bn]
    key_scale = net._layer_names[i_scale] if i_scale else None

    # Copy
    bn_mean = copy_double(net.params[key_bn][0].data)
    bn_variance = copy_double(net.params[key_bn][1].data)
    # 在caffe实现中计算均值方差采用了滑动衰减方式,用了scale_factor代替num_bn_samples.
    num_bn_samples = copy_double(net.params[key_bn][2].data)

    # and Invalidate the BN layer
    net.params[key_bn][0].data[:] = 0
    net.params[key_bn][1].data[:] = 1
    net.params[key_bn][2].data[:] = 1
    if num_bn_samples[0] == 0:
        num_bn_samples[0] = 1

    if net.params.has_key(key_scale):
        print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)
        scale_weight = copy_double(net.params[key_scale][0].data)
        scale_bias = copy_double(net.params[key_scale][1].data)
    else:
        print 'Combine {:s} + {:s}'.format(key_conv, key_bn)
        scale_weight = 1
        scale_bias = 0

    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)
    if conv_freezed:
        weight = copy_double(net.params[key_conv][0].data)
        bias = copy_double(net.params[key_conv][1].data)
        net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)
        for i in range(len(alpha)):
            net.params[key_conv][0].data[i] = weight[i] * alpha[i]
        # and Invalidate the Scale layer
        net.params[key_scale][0].data[:] = 1
        net.params[key_scale][1].data[:] = 0
    elif net.params.has_key(key_scale): # 仅合并BN+Scale=>Scale
        net.params[key_scale][0].data[:] = alpha
        net.params[key_scale][1].data[:] -= bn_mean * alpha
    # else: # 没有scale,不合并


def merge_batchnorms_in_net(net, pb_model, is_train):
    # 对于test.proto中Input类型的输入,caffe.proto.caffe_pb2.NetParameter没有将其解析成第0层,
    # 而caffe.Net(dst_model, caffe.TEST)可以解析到,并且caffe_pb2.NetParameter解析出来的更proto中的定义不一样,新增了一下东西
    # 跟caffe.Net中层不一致,所以只能通过层名来定位.
    pb_layer_names = [l.name for l in pb_model.layer]
    # for each BN
    for i, layer in enumerate(net.layers):
        if layer.type != 'BatchNorm':
            continue

        l_name = net._layer_names[i]

        l_bottom = net.bottom_names[l_name]
        assert(len(l_bottom) == 1)
        l_bottom = l_bottom[0]
        l_top = net.top_names[l_name]
        assert(len(l_top) == 1)
        l_top = l_top[0]

        can_be_absorbed = False
        # Search all (bottom) layers
        for j in xrange(i - 1, -1, -1):
            tops_of_j = net.top_names[net._layer_names[j]]
            if l_bottom in tops_of_j:
                if net.layers[j].type in ['Convolution', 'InnerProduct']:
                    if is_train:
                        bn_pb_idx = pb_layer_names.index(l_name)
                        if not pb_model.layer[bn_pb_idx].batch_norm_param.use_global_stats: # BN需要学习
                            break
                    # freezed,不再进行学习
                    conv_ind = j
                    can_be_absorbed = True
                    # There must be only one bottom conv/FC layer
                    break

        if not can_be_absorbed:
            continue

        conv_freezed = False
        l_name = net._layer_names[conv_ind]
        bn_pb_idx = pb_layer_names.index(l_name)
        if (not is_train) or (pb_model.layer[bn_pb_idx].param[0].lr_mult == 0): # freezed,不再进行学习
            conv_freezed = True

        # find the following Scale
        scale_ind = None
        for j in range(i + 1, len(net.layers)):
            bottoms_of_j = net.bottom_names[net._layer_names[j]]
            if l_top in bottoms_of_j:
                if scale_ind:
                    # Followed by two or more layers
                    scale_ind = None
                    break

                if net.layers[j].type in ['Scale']:
                    scale_ind = j

                    top_of_j = net.top_names[net._layer_names[j]][0]
                    if top_of_j == bottoms_of_j[0]:
                        # On-the-fly => Can be merged
                        break

                else:
                    # Followed by a layer which is not 'Scale'
                    scale_ind = None
                    break


        merge_conv_and_bn(net, conv_ind, i, scale_ind, conv_freezed)

    return net

def save_model2pbfile(netParameter, out_file):
     with open(out_file, 'w') as f:
         f.write(pb.text_format.MessageToString(netParameter))

def process_model(net, src_model, func_loop, func_finally):
    with open(src_model) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), model)


    for i, layer in enumerate(model.layer):
        map(lambda x: x(layer, net, model, i), func_loop)

    map(lambda x: x(net, model), func_finally)
    return model


# Functions to remove (redundant) BN and Scale layers
to_delete_empty = []
def pick_empty_layers(layer, net, model, i):
    if layer.type not in ['BatchNorm', 'Scale']:
        return

    bottom = layer.bottom[0]
    top = layer.top[0]

    if (bottom != top):
        # Not supperted yet
        return

    if layer.type == 'BatchNorm':
        zero_mean = np.all(net.params[layer.name][0].data == 0)
        one_var = np.all(net.params[layer.name][1].data == 1)
        # 这里尚不清楚
        # length_is_1 = (net.params['conv1_1/bn'][2].data == 1) or (net.params[layer.name][2].data == 0)
        length_is_1 = (net.params[layer.name][2].data == 1) or (net.params[layer.name][2].data == 0)

        if zero_mean and one_var and length_is_1:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

    if layer.type == 'Scale':
        no_scaling = np.all(net.params[layer.name][0].data == 1)
        zero_bias = np.all(net.params[layer.name][1].data == 0)

        if no_scaling and zero_bias:
            print 'Delete layer: {}'.format(layer.name)
            to_delete_empty.append(layer)

def remove_empty_layers(net, model):
    map(model.layer.remove, to_delete_empty)


# A function to add 'engine: CAFFE' param into 1x1 convolutions
# CUDNN(library kernels + stream parallelism) 引擎通常比CAFFE(matrix multiplication)的GPU实现要快
# 这里的1x1卷积可能没办法再优化,不需要CUDNN,也可能CUDNN只用于卷积核大于1的
def set_engine_caffe(layer, net, model, i):
    if layer.type == 'Convolution':
        if layer.convolution_param.kernel_size == 1\
            or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']


def merge_pb_weights(args, model=None, train_stage=True, is_train_proto=False, save_pb_model=True, save_weights=True):
    if train_stage: # phase TRAIN
        postfix_w = '_merged-r.caffemodel'
        postfix_m = '_merged.proto'
    else:
        postfix_w = '_infer.caffemodel'
        postfix_m = '_infer.proto'
    # Set default output file names
    file_name = osp.splitext(model)[0]
    output_model = file_name + postfix_m
    file_name = osp.splitext(args.weights)[0]
    output_weights = file_name + postfix_w

    tmp_proto = model + '.temp.pt'

    net, pb_model = load_and_fill_biases(model, args.weights, tmp_proto, None, train_stage,is_train_proto)

    net = merge_batchnorms_in_net(net, pb_model, train_stage)

    dst_model = process_model(net, tmp_proto,
                  [pick_empty_layers, set_engine_caffe],
                  [remove_empty_layers])

    # Store params
    if save_pb_model: save_model2pbfile(dst_model, output_model)
    if save_weights:  net.save(output_weights)
    # remove tmp file
    os.remove(tmp_proto)
    # release memory
    pb_model = dst_model = net = tmp_proto = None
    del pb_model,dst_model
    del net
    import gc 
    gc.collect()

def main(args):
    if args.phase=='train':
        if args.train is None or args.test is None:
	    print('ERROR: --train & --test')
	    sys.exit(0)
	print('fork 1 Process from (%s) ...' % os.getpid())
	# 在一个进程中由于没有释放GPU资源,造成了第二次运行的caffe.Net的时候报错,分别使用一个进程没有影响.
	pid = os.fork() # only for UNIX,try multiprocessing instead
# 如果双进程有问题，可能是内存等原因，可以分别运行
#        pid = 1
	if pid==0:
	    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()) )
            merge_pb_weights(args,train_stage=True,model=args.test,save_weights=False)
	    sys.exit(0)
	else:
	    print('I (%s) just created a child process (%s).' % (os.getpid(), pid) )
            merge_pb_weights(args,train_stage=True,model=args.train,is_train_proto=True)
            try:
                os.wait() # wait for child process
            except:
                pass
    elif args.phase=='test':
        if args.test is None:
	    print('ERROR: --test')
	    sys.exit(0)
        merge_pb_weights(args,train_stage=False,model=args.test)
    else:
        print('param error: must be train/test')

if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('phase', help="train/test (train only merging freezed layers)")
    parser.add_argument('weights', help="The weights caffemodel")
    parser.add_argument('--train', help="The net definition prototxt")
    parser.add_argument('--test', help="The net definition prototxt")
    #parser.add_argument('phase', help="0 for test, 1 for train(only merging freezed layers)")
    #parser.add_argument('--output_model')
    #parser.add_argument('--output_weights')
    args = parser.parse_args()
    main(args)

