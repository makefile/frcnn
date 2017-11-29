import numpy as np
import sys
import lmdb
from PIL import Image
import scipy.io as sio
import os
if os.path.exists('./python/caffe'):
    sys.path.append('./python')
else:
    print 'Error : caffe(pycaffe) could not be found'
    sys.exit(0)
caffe_root = '%s'%(os.getcwd())
import caffe
from caffe.proto import caffe_pb2

def binaryproto2img():

    mean = caffe_pb2.BlobProto()
    data = open('{}/examples/cifar10/mean.binaryproto'.format(caffe_root)).read()
    mean.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(mean))
    print 'arr.shape    : {}'.format(arr.shape)
    print 'arr[0].shape : {}'.format(arr[0].shape)

    """
    im = np.zeros((1024,2048,3),dtype=np.float32)
    im[:,:,0] = arr[0][0,:,:]
    im[:,:,1] = arr[0][1,:,:]
    im[:,:,2] = arr[0][2,:,:]
    im = Image.fromarray(im)
    im.save('mean_cityscape_img_train_sp_500x500.jpg')
    """

    sio.savemat('mean_cifar10.mat',{'mean':arr[0]})

    return

if __name__=='__main__':
    binaryproto2img()
