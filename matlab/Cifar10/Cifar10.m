clear;clc;

model_file = 'cifar10_res20_trainval.proto';
weights_file = 'cifar10_res20_iter_64000.caffemodel';

current_dir = pwd;
caffe_dir = '../../'; cd(caffe_dir); caffe_dir = pwd;
cd(current_dir);
addpath(fullfile(caffe_dir,'matlab'));
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(4);

rand('seed',0);
cifar10_train_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_train_lmdb.mat'));
cifar10_train_data.image = single(permute(cifar10_train_data.image,[3,2,4,1]));
cifar10_test_data = load(fullfile(caffe_dir,'examples','cifar10','cifar10_test_lmdb.mat'));
cifar10_test_data.image = single(permute(cifar10_test_data.image,[3,2,4,1]));
train_num = size(cifar10_train_data.label, 1);
test_num = size(cifar10_test_data.label, 1);

%mean_cifar10 = load('mean_cifar10.mat');
%mean_cifar10 = mean_cifar10.mean;
%mean_cifar10 = permute(mean_cifar10,[3,2,1]);

%sub mean
%cifar10_train_data.image = cifar10_train_data.image - single(repmat(mean_cifar10,1,1,1,train_num));
%cifar10_test_data.image = cifar10_test_data.image - single(repmat(mean_cifar10,1,1,1,test_num));

net = caffe.Net(model_file, weights_file, 'test');
