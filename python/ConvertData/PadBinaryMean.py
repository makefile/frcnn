import numpy as np
import sys
import lmdb
from PIL import Image
import scipy.io as sio
import os

caffe_root = '%s' %(os.getcwd())
if os.path.exists('{}/python/caffe'.format(caffe_root)):
    sys.path.append('{}/python'.format(caffe_root))
else:
    print 'Error : caffe(pycaffe) could not be found'
    sys.exit(0)
import caffe
from caffe.proto import caffe_pb2

def binarypro_pad(pad=4):

    mean = caffe_pb2.BlobProto()
    data = open('{}/examples/cifar10/mean.binaryproto'.format(caffe_root)).read()
    mean.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(mean))
    print 'arr.shape    : {}'.format(arr.shape)
    print 'arr[0].shape : {}'.format(arr[0].shape)
    assert len(arr.shape) == 4 and arr.shape[0] == 1 and arr.shape[1] == 3 and arr.shape[2] == 32 and arr.shape[3] == 32
    pad_shape = (arr.shape[0], arr.shape[1], arr.shape[2] + 2*pad, arr.shape[3] + 2*pad)
    pad_arr = np.zeros(pad_shape, dtype=arr.dtype)
    pad_arr[:,:,pad:arr.shape[2]+pad,pad:arr.shape[3]+pad] = arr

    pad_mean = caffe.io.array_to_blobproto(pad_arr)
    data = pad_mean.SerializeToString()
    open('{}/examples/cifar10/pad_{}_mean.binaryproto'.format(caffe_root, pad), 'w').write(data)
    print 'Save into {}/examples/cifar10/pad_{}_mean.binaryproto'.format(caffe_root, pad)

if __name__=='__main__':
    binarypro_pad()
