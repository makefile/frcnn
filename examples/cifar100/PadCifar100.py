## Transfer cifar100 lmdb data to mat
import sys
import lmdb
import numpy as np
from array import array
import scipy.io as sio
import os
if os.path.exists('./python/caffe'):
    sys.path.append('./python')
else:
    print 'Error : caffe(pycaffe) could not be found'
    sys.exit(0)
import caffe
from caffe.proto import caffe_pb2
import argparse
import cv2

def PadCifar100(Train_or_Test, pad_pixels):
    cifar100_train_data = './examples/cifar100/cifar100_train_lmdb'
    cifar100_test_data  = './examples/cifar100/cifar100_test_lmdb'
    if Train_or_Test:
        data_path = cifar100_train_data
        save_path = './examples/cifar100/cifar100_train_lmdb_pad_{}'.format(pad_pixels)
    else:
        data_path = cifar100_test_data
        save_path = './examples/cifar100/cifar100_test_lmdb_pad_{}'.format(pad_pixels)

    if os.path.exists(save_path):
        print 'New LMDB File Existed : {}'.format(save_path)
        __import__('shutil').rmtree(save_path)
        print 'Delete {} done..'.format(save_path)

    print 'LMDB DATA PATH : %s' %data_path 
    lmdb_env = lmdb.open(data_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for (count, (key, value)) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)

        #label = datum.label
        data = caffe.io.datum_to_array(datum)

        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        shape = image.shape
    print 'Count the number of data are %d' %count

    print 'Original Shape : %d %d %d' %(shape[2], shape[0], shape[1])
    print 'Transposed Shape : {}'.format(shape)

    # Save into new LMDB dataset
    in_db = lmdb.open(save_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn :
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
            shape = data.shape
            assert len(shape) == 3 and shape[0] == 3
            new_shape = (shape[0], shape[1] + pad_pixels * 2, shape[2] + pad_pixels * 2)
            new_data = np.zeros(new_shape, dtype=data.dtype)
            new_data[:, pad_pixels:data.shape[1]+pad_pixels,
                pad_pixels:data.shape[2]+pad_pixels] = data
            new_datum = caffe.io.array_to_datum(new_data, datum.label)
            in_txn.put(key, new_datum.SerializeToString())
    in_db.close()
    print 'Convert Done !\nSaved in LMDB : %s'  %save_path

def binarypro_pad(pad=4):
    mean = caffe_pb2.BlobProto()
    data = open('./examples/cifar100/mean.binaryproto', 'r').read()
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
    open('./examples/cifar100/pad_{}_mean.binaryproto'.format(pad), 'w').write(data)

    print 'Save into ./examples/cifar100/pad_{}_mean.binaryproto'.format(pad)

if __name__ == '__main__':
    PadCifar100(True, 4)
    PadCifar100(False, 4)
    binarypro_pad(4)
