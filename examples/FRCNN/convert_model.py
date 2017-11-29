import json
import numpy as np
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
    parser = argparse.ArgumentParser(description='Calculate Recall For Rpn Detector Results')
    parser.add_argument('--model', dest='model_file',
                        help='model prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='weight_file',
                        help='weights prototxt',
                        default=None, type=str)
    parser.add_argument('--config', dest='config_file',
                        help='config json file',
                        default=None, type=str)
    parser.add_argument('--net_out', dest='net_out_path',
                        help='saved path',
                        default=None, type=str)
    
    args = parser.parse_args()

    if args.model_file is None or args.weight_file is None or args.config_file is None or args.net_out_path is None:
        parser.print_help()
        sys.exit(1)

    return args

if __name__ == '__main__':
    args = parse_args()
    config_file = open(args.config_file, 'r')
    config = config_file.read();
    config = config.replace('\n','')
    config_file.close();

    dict = json.loads(config)

    means,stds = None, None
    num_classes = None
    for (key,value) in dict.items():
        if key == 'n_classes':
            num_classes = value
        if key == 'bbox_normalize_means':
            means = value
        if key == 'bbox_normalize_stds':
            stds = value

    if means is None or stds is None or num_classes is None:
        print 'Lose n_classes or bbox_normalize_stds or bbox_normalize_means'
        sys.exit(1)

    means = str(means).replace(',',' ').split(' ')
    for jj in range(means.count('')):
        means.remove('')
    stds = str(stds).replace(',',' ').split(' ')
    for jj in range(stds.count('')):
        stds.remove('')

    num_classes = int(num_classes)
    print 'Load Json File {} Done'.format(args.config_file)
    print 'means: {}\nstds: {}\nnum_classes:{}'.format(means, stds, num_classes)


    means = np.tile( np.array([float(means[0]),float(means[1]),float(means[2]),float(means[3])]), (num_classes, 1))
    stds = np.tile( np.array([float(stds[0]),float(stds[1]),float(stds[2]),float(stds[3])]), (num_classes, 1))
    means = means.ravel()
    stds = stds.ravel()

    print 'Load Caffe Model : ' + args.model_file + ' with weights : ' + args.weight_file
    net = caffe.Net(args.model_file, args.weight_file, caffe.TEST)
    #net = caffe.Net(args.model_file, args.weight_file, caffe.TRAIN)
    # scale and shift with bbox reg unnormalization; then save snapshot
    net.params['bbox_pred'][0].data[...] = \
        (net.params['bbox_pred'][0].data * stds[:, np.newaxis])
    net.params['bbox_pred'][1].data[...] = \
        (net.params['bbox_pred'][1].data * stds + means)

    net.save(str(args.net_out_path))
    
    print 'Save new model into {}'.format(args.net_out_path)
