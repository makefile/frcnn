## Transfer cifar10 lmdb data to mat
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

def PadCifar10(Train_or_Test, pad_pixels):
    cifar10_train_data = './examples/cifar10/cifar10_train_lmdb'
    cifar10_test_data  = './examples/cifar10/cifar10_test_lmdb'
    if Train_or_Test:
        data_path = cifar10_train_data
        save_path = './examples/cifar10/cifar10_train_lmdb_pad_{}'.format(pad_pixels)
    else:
        data_path = cifar10_test_data
        save_path = './examples/cifar10/cifar10_test_lmdb_pad_{}'.format(pad_pixels)

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

    print 'Count the number of data are %d' %(count+1)

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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cifar10 mat data from lmdb')
    parser.add_argument('--Type', dest='type', help='Convert train data or test, use [Train] or [Test]',
                    default='Train', type=str);
    parser.add_argument('--Pad', dest='pad', help='Pad Image Data with zeros',
                    default=0, type=int);
    args = parser.parse_args()
    print 'Convert cifar10 LMDB %s Data' % args.type
    
    PadCifar10(args.type=='Train', args.pad)
