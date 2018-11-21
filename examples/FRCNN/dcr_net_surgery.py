import sys, argparse
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('./python/')
import caffe

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Net surgery. init the weight file of DCR model with ResNet')
    parser.add_argument('--model', dest='model_file',
                        help='model prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='weight_file',
                        help='weights prototxt',
                        default=None, type=str)
    parser.add_argument('--net_out', dest='net_out_path',
                        help='saved path',
                        default=None, type=str)
    
    args = parser.parse_args()

    if args.model_file is None or args.weight_file is None or args.net_out_path is None:
        parser.print_help()
        sys.exit(1)

    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Load Caffe Model : ' + args.model_file + ' with weights : ' + args.weight_file
    #net = caffe.Net(args.model_file, args.weight_file, caffe.TEST)
    net = caffe.Net(args.model_file, args.weight_file, caffe.TRAIN)
    # NOTICE that the layer names of DCR module should start with dcr_
    dcr_names = [x for x in net.params if 'dcr_res' in x or 'dcr_scale' in x]
    for name in dcr_names:
        for i in range(len(net.params[name])):
            # copy weight dcr_xxx_layer = xxx_layer
            net.params[name][i].data[...] = net.params[name[4:]][i].data

    net.save(str(args.net_out_path))
    
    print 'Save new model into {}'.format(args.net_out_path)
