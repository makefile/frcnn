#coding:utf-8
# contribution from xu huajiang

import caffe
import sys
import numpy as np
import caffe.proto.caffe_pb2 as caffe_pb2

# generate nbn prototxt
def generate_nbn_prototxt(input_prototxt, input_caffemodel, output_prototxt):
    input_network = caffe.Net(input_prototxt, input_caffemodel, caffe.TEST)

    f = open(input_caffemodel, 'rb')
    tmp_model = caffe_pb2.NetParameter()
    tmp_model.ParseFromString(f.read())
    f.close()
    layers = tmp_model.layer

    w = open(output_prototxt, "w")
    w.write('name: "%s"\n' %  tmp_model.name)
    remove_dict = {}
    split_dict = {}
    all_layers = input_network.params.keys()
    layers = tmp_model.layer
    for index, layer in enumerate(layers):
        print "name:", layer.name, layer.type
        res=list()
        if layer.type == "ImageData" or layer.type == "Input":
            try:
                res.append('input: "%s"' %  layers[index+1].bottom[0])
            except:
                res.append('input: "%s"' %  layer.name)
            dim =  input_network.blobs['data'].data.shape
            res.append('input_dim: %d' % dim[0])
            res.append('input_dim: %d' % dim[1])
            res.append('input_dim: %d' % dim[2])
            res.append('input_dim: %d' % dim[3])
        else:
            if layer.type == "BatchNorm" or layer.type == "Scale":
                if layer.bottom[0] != layer.top[0]:
                    remove_dict[layer.top[0]] = layer.bottom[0]
                continue
            if layer.type == "Split":
                for top in layer.top:
                    split_dict[top] = layer.bottom[0]
                continue
            res.append('layer {')
            if layer.name == "loss":
                res.append('name: "prob"' )
            else:
                res.append('name: "%s"' % layer.name)
            if layer.type[-8:] == "WithLoss":
                res.append('type: "%s"' % layer.type[:-8])
            else:
                res.append('type: "%s"' % layer.type)
              
            bottoms = layer.bottom
            for bottom in bottoms:
                if layer.type == "ReLU" or layer.type == "Eltwise":
                    if bottom in remove_dict:
                        res.append('bottom: "%s"' % remove_dict[bottom])
                    elif bottom in split_dict: 
                        res.append('bottom: "%s"' % split_dict[bottom])
                    else:
                        res.append('bottom: "%s"' % bottom)
                elif bottom != "label":
                    if bottom in split_dict:
                        res.append('bottom: "%s"' % split_dict[bottom])
                    else:
                        res.append('bottom: "%s"' % bottom)

            tops = layer.top
            for top in tops:
                if top == "loss":
                     res.append('top: "prob"')
                else:
                    res.append('top: "%s"' % top)
            
            # Eltwise
            if  layer.type == "Eltwise":
                res.append('eltwise_param {')
                res.append('   %s' % layer.eltwise_param)
                res.append('  }')


            # param
            for param in layer.param:
                param_res = list()
                if param.lr_mult is not None:
                    param_res.append('    lr_mult: %s' % param.lr_mult)
                #if param.decay_mult!=1:
                param_res.append('    decay_mult: %s' % param.decay_mult)
                if len(param_res)>0:
                    res.append('  param{')
                    res.extend(param_res)
                    res.append('  }')

            # lrn_param
            if layer.lrn_param is not None:
                lrn_res = list()
                if layer.lrn_param.local_size!=5:
                    lrn_res.append('    local_size: %d' % layer.lrn_param.local_size)
                if layer.lrn_param.alpha!=1:
                    lrn_res.append('    alpha: %f' % layer.lrn_param.alpha)
                if layer.lrn_param.beta!=0.75:
                    lrn_res.append('    beta: %f' % layer.lrn_param.beta)
                NormRegionMapper={'0': 'ACROSS_CHANNELS', '1': 'WITHIN_CHANNEL'}
                if layer.lrn_param.norm_region!=0:
                    lrn_res.append('    norm_region: %s' % NormRegionMapper[str(layer.lrn_param.norm_region)])
                EngineMapper={'0': 'DEFAULT', '1':'CAFFE', '2':'CUDNN'}
                if layer.lrn_param.engine!=0:
                    lrn_res.append('    engine: %s' % EngineMapper[str(layer.lrn_param.engine)])
                if len(lrn_res)>0:
                    res.append('  lrn_param{')
                    res.extend(lrn_res)
                    res.append('  }')

            # convolution_param
            if layer.convolution_param is not None:
                convolution_param_res = list()
                conv_param = layer.convolution_param
                if conv_param.num_output!=0:
                    convolution_param_res.append('    num_output: %d'%conv_param.num_output)
                if len(conv_param.kernel_size) > 0:
                    for kernel_size in conv_param.kernel_size:
                        convolution_param_res.append('    kernel_size: %d' % kernel_size)
                if len(conv_param.pad) > 0:
                    for pad in conv_param.pad:
                        convolution_param_res.append('    pad: %d' % pad)
                if len(conv_param.stride) > 0:
                    for stride in conv_param.stride:
                        convolution_param_res.append('    stride: %d' % stride)
                if conv_param.weight_filler is not None and conv_param.weight_filler.type!='constant':
                    convolution_param_res.append('    weight_filler {')
                    convolution_param_res.append('      type: "%s"'%conv_param.weight_filler.type)
                    convolution_param_res.append('    }')
                if conv_param.bias_filler is not None and layer.type == "Convolution":
                    convolution_param_res.append('    bias_filler {')
                    convolution_param_res.append('      type: "%s"' % conv_param.bias_filler.type)
                    convolution_param_res.append('      value: %s' % conv_param.bias_filler.value)
                    convolution_param_res.append('    }')

                if len(convolution_param_res)>0:
                    res.append('convolution_param {')
                    res.extend(convolution_param_res)
                    res.append('  }')

            # pooling_param
            if layer.pooling_param is not None:
                pooling_param_res = list()
                if layer.pooling_param.kernel_size>0:
                    pooling_param_res.append('    kernel_size: %d' % layer.pooling_param.kernel_size)
                    pooling_param_res.append('    stride: %d' % layer.pooling_param.stride)
                    pooling_param_res.append('    pad: %d' % layer.pooling_param.pad)
                    PoolMethodMapper={'0':'MAX', '1':'AVE', '2':'STOCHASTIC'}
                    pooling_param_res.append('    pool: %s' % PoolMethodMapper[str(layer.pooling_param.pool)])

                if len(pooling_param_res)>0:
                    res.append('pooling_param {')
                    res.extend(pooling_param_res)
                    res.append('  }')

            # inner_product_param
            if layer.inner_product_param is not None:
                inner_product_param_res = list()
                if layer.inner_product_param.num_output!=0:
                    if layer.inner_product_param.weight_filler is not None and layer.inner_product_param.weight_filler.type!='constant':
                        inner_product_param_res.append('    weight_filler {')
                        inner_product_param_res.append('      type: "%s"' % layer.inner_product_param.weight_filler.type)
                        inner_product_param_res.append('      std: %s' % layer.inner_product_param.weight_filler.std)
                        inner_product_param_res.append('    }')
                    if layer.inner_product_param.bias_filler is not None:
                        inner_product_param_res.append('    bias_filler {')
                        inner_product_param_res.append('      type: "%s"' % layer.inner_product_param.bias_filler.type)
                        inner_product_param_res.append('      value: %s' % layer.inner_product_param.bias_filler.value)
                        inner_product_param_res.append('    }')
                    inner_product_param_res.append('    num_output: %d' % layer.inner_product_param.num_output)
                    
                if len(inner_product_param_res)>0:
                    res.append('  inner_product_param {')
                    res.extend(inner_product_param_res)
                    res.append('  }')

            # drop_param
            if layer.dropout_param is not None:
                dropout_param_res = list()
                try:
                    if layer.dropout_param.dropout_ratio!=0.5 or layer.dropout_param.scale_train!=True:
                        dropout_param_res.append('    dropout_ratio: %f' % layer.dropout_param.dropout_ratio)
                        dropout_param_res.append('    scale_train: ' + str(layer.dropout_param.scale_train))
                
                    if len(dropout_param_res)>0:
                        res.append('  dropout_param {')
                        res.extend(dropout_param_res)
                        res.append('  }')
                except Exception, err:
                    pass
                    #print err
            # flatten


            res.append('}')
                
        for line in res:
            #print line
            w.write(line + "\n")          
    w.close()     

# generate nbn caffemodel
def generate_nbn_caffemodel(input_prototxt, input_caffemodel, output_prototxt, output_caffemodel, eps = 0.00001):
    input_network = caffe.Net(input_prototxt, input_caffemodel, caffe.TEST)

    f = open(input_caffemodel, 'rb')
    tmp_model = caffe_pb2.NetParameter()
    tmp_model.ParseFromString(f.read())
    f.close()
    layers = tmp_model.layer

    output_network = caffe.Net(output_prototxt, caffe.TEST)

    for i in range(len(layers)):
        if layers[i].type == "Input" or layers[i].type == "Eltwise" or layers[i].type == "Scale" or layers[i].type == "BatchNorm" or layers[i].type == "ImageData" or layers[i].type == "ReLU" or layers[i].type == "Pooling" or layers[i].type == "Split" or layers[i].type == "Concat" or  layers[i].type == "Flatten" or layers[i].type == "SoftmaxWithLoss":
            continue
        elif layers[i].type == "Convolution":
            if not (layers[i+2].type == "Scale" and layers[i+1].type == "BatchNorm"):
                continue
            bn_conv = layers[i+1].name
            scale_conv = layers[i+2].name
            conv_w = input_network.params[layers[i].name][0].data[...]
            print layers[i].name, layers[i+1].name, layers[i+2].name, layers[i+2].scale_param.bias_term, layers[i].convolution_param.bias_term, conv_w.shape
            if layers[i].convolution_param.bias_term:
                # original conv
                conv_b = input_network.params[layers[i].name][1].data[...]
            else:
                conv_b = np.zeros((conv_w.shape[0],), dtype=np.uint8)

            # original batchnormal
            scale = input_network.params[bn_conv][2].data[...]
            mean = input_network.params[bn_conv][0].data[...]
            var = input_network.params[bn_conv][1].data[...]

            # original scale
            scale_w = input_network.params[scale_conv][0].data[...]
            scale_b = input_network.params[scale_conv][1].data[...]
            #print "scale_w:", scale_w

            # calculate
            var = np.sqrt(var/scale+eps)
            conv_b = conv_b-mean/scale
            conv_b = conv_b/var
            var = scale_w/var
            conv_b = scale_w*conv_b
            conv_b = conv_b + scale_b

            for j in range(len(var)):
                output_network.params[layers[i].name][0].data[j] = var[j]*conv_w[j]
            output_network.params[layers[i].name][1].data[...] = conv_b
        else:
            output_network.params[layers[i].name][0].data[...] = input_network.params[layers[i].name][0].data[...]
            output_network.params[layers[i].name][1].data[...] = input_network.params[layers[i].name][1].data[...]    

    output_network.save(output_caffemodel)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('Usage: shrink_bn_caffe.py input_prototxt input_caffemodel output_prototxt output_caffemodel')
        exit()
    input_prototxt = sys.argv[1]
    input_caffemodel = sys.argv[2]
    output_prototxt = sys.argv[3]
    output_caffemodel = sys.argv[4]

    caffe.set_mode_gpu()
    generate_nbn_prototxt(input_prototxt, input_caffemodel, output_prototxt) 
    generate_nbn_caffemodel(input_prototxt, input_caffemodel, output_prototxt, output_caffemodel, eps = 0.00001)

