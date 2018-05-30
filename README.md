**Special Features for This Caffe Repository**

- Faster rcnn/R-FCN approximate joint train, test and evaluate
- support [R-FCN](https://arxiv.org/abs/1605.06409) with OHEM
- support FPN ([Feature Pyramid Network](https://arxiv.org/abs/1612.03144))
- support SSD layers
- script for merging `Conv + BatchNorm + Scale` layers to 1 layer when those layer are freezed to reduce memory: 'examples/FRCNN/res50/gen\_merged\_model.py'. script for merge ResNet: 'examples/FRCNN/merge\_resnet.sh'.
- support snapshot after got -SIGTERM (kill command's default signal)
- logger tools by VisualDL which can visualize loss scalars and feature images .etc
- support NMS and IOU calc on GPU
- Action recognition (Two Stream CNN)
- CTPN layers for scene text detection, port from [tianzhi0549/CTPN](https://github.com/tianzhi0549/CTPN)
- support CUDA 8.0, CuDNN 7.0

**Special layers**

- ROIAlign proposed in [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- FocalLoss in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Swish Activation function in [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- Eltwise layer using in-place sum to reduce memory, from [this PR](https://github.com/BVLC/caffe/pull/3708)
- caffe layer module, layer definition and usage like `Python layer`,from caffe [PR#5294](https://github.com/BVLC/caffe/pull/5294)
- CuDNNDeconv layer, Depth-wise Conv layer
- CTPN layers include LSTM layer implemented by [@junhyukoh](https://github.com/junhyukoh/caffe-lstm),which is faster than upstream master branch of Caffe.

**Data Preprocess**

data enhancement:
- support Histogram equalization of color image
- haze-free algorithm

data augmentation:
- random flip horizontal
- random jitter
- hue, saturation, exposure
- rotate(multiple of 90 degree)

**TODO list**

- [ ] support batch image greater than 1 (on branch batch)
- [x] support Rotated R-CNN for rotated bounding box (on branch r-frcnn)

## Installation

This repository uses C++11 features, so make sure to use compiler that is compatible of C++11.
```shell
cd $CAFFE_ROOT
cp Makefile.config.example Makefile.config
# modify the content in Makefile.config to adapt your system
# if you like to use VisualDL to log losses, set USE_VISUALDL to 1,
# and cd src/logger && make
make -j
make pyfrcnn # if you need use python to demo
```

All following steps, you should do these in the `$CAFFE_ROOT` path.

## Faster R-CNN

### Disclaimer
The official [Faster R-CNN](https://arxiv.org/abs/1506.01497) code of NIPS 2015 paper (written in MATLAB) is [available](https://github.com/ShaoqingRen/faster_rcnn) here. It is worth noticing that:

- This repository contains a C++ reimplementation of the Python code([py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)), which is built on [caffe1](https://github.com/BVLC/caffe).
- This repository used code from [caffe-faster-rcnn](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev) `commit 8ba1d26` as base framework.

### Demo
Using `sh example/FRCNN/demo_frcnn.sh`, the will process five pictures in the `examples/FRCNN/images`, and put results into `examples/FRCNN/results`.

Note: You should prepare the trained caffemodel into `models/FRCNN`, such as `ZF_faster_rcnn_final.caffemodel` for ZF model.

### Prepare for training and evaluation
- The list of training data is `examples/FRCNN/dataset/voc2007.trainval`.
- The list of testing data is `examples/FRCNN/dataset/voc2007.trainval`.
- Create symlinks for the PASCAL VOC dataset `ln -s $YOUR_VOCdevkit_Path $CAFFE_ROOT/VOCdevkit`.

As shown in VGG example `models/FRCNN/vgg16/train_val.proto`, the original pictures should appear at `$CAFFE_ROOT/VOCdevkit/VOC2007/JPEGImages/`. (Check window\_data\_param in FrcnnRoiData)

If you want to train Faster R-CNN on your own dataset, you may prepare custom dataset list.
The format is as below
```
# image-id
image-name
number of boxes
label x1 y1 x2 y2 difficulty
...
```

### Training
`sh examples/FRCNN/zf/train_frcnn.sh` will start training process of voc2007 data using ZF model.

The ImageNet pre-trained models can be found in [this link](https://drive.google.com/drive/folders/1xjFL-ZeVzXkY584ZsEnr9O6O3P1Ypjwd?usp=sharing)

If you use the provided training script, please make sure:
- VOCdevkit is within $CAFFE\_ROOT and VOC2007 in within VOCdevkit
- ZF pretrain model should be put into models/FRCNN/ as ZF.v2.caffemodel

`examples/FRCNN/convert_model.py` transform the parameters of `bbox_pred` layer by mean and stds values,
because the regression value is normalized during training and we should recover it to obtain the final model.

### Evaluation
`sh examples/FRCNN/zf/test_frcnn.sh` the will evaluate the performance of voc2007 test data using the trained ZF model.

- First Step of This Shell : Test all voc-2007-test images and output results in a text file.
- Second Step of This Shell : Compare the results with the ground truth file and calculate the mAP.

### Config

The program use config file named like `config.json` to set params. Special params need to be cared about:

- `data_jitter`: data augmentation, if set <0 then no jitter,hue,saturation,exposure
- `im_size_align`: set to stride of last conv layer of FPN to avoid Deconv shape problem, such as 64, set to 0 to disable
- `bbox_normalize_targets`: do bbox norm in training, and do unnorm at testing(do not need convert model weight before testing)
- `test_rpn_score_thresh`: you can set >0 to speed up NMS at testing

### Detail

Scripts and prototxts for different models are listed in the `examples/FRCNN`

More details about the code in include and src directory:

- `api/FRCNN` for demo and test api
- `caffe/FRCNN` contains codes related to Faster R-CNN
- `caffe/RFCN` for R-FCN
- `caffe/DeformConv` for Deformable Conv
- `caffe/SSD` for SSD
- `logger` dir relates to logger tools
- `modules` and `yaml-cpp` relate to Caffe module layers, which include FPN layers .etc
- `python/frcnn` relates to pybind11 interface for demo
- `caffe/ACTION_REC` Two-Stream Convolutional Networks for Action Recognition in Video
- `caffe/CTPN` relates to CTPN special layers for scene text detection
- `caffe/PR` for some layers from caffe PR

### Commands, Rebase From Caffe Master

**For synchronous with official caffe**

- git remote add caffe https://github.com/BVLC/caffe.git
- git fetch caffe
- git checkout master
- git rebase caffe/master

**Rebase the dev branch**
- git checkout dev
- git rebase master 
- git push -f origin dev

## QA

- CUB not found, when compile for GPU version, `frcnn_proposal_layer.cu` requires a head file `<cub/cub.cuh>`. CUB is library contained in the official Cuda Toolkit, usually can be found in ` /usr/local/cuda/include/thrust/system/cuda/detail/`. You should add this path in your `Makefile.config` (try `locate cub.cuh` to find cub on your system)
- When Get `error: RPC failed; result=22, HTTP code = 0`, use `git config http.postBuffer 524288000`, increases git buffer to 500mb
- Cannot load module layer dynamic library, the program search the modules first in enviroment variable `CAFFE_LAYER_PATH` then in predefined `DEFAULT_LAYER_PATH` in Makefile. So try to set `CAFFE_LAYER_PATH` in shell script. And this could be happen when using pycaffe.
- about R-FCN: currently not support class-agnostic (although it is easy to modify), and OHEM method has very little improvement in joint train. also remember to set `bg_thresh_lo` to 0 when use OHEM.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite the following papers in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
    @inproceedings{girshick2015fast,
      title={Fast R-CNN},
      author={Girshick, Ross},
      booktitle={International Conference on Computer Vision},
      pages={1440--1448},
      year={2015}
    }
    @inproceedings{ren2015faster,
      title={Faster {R-CNN}: Towards real-time object detection with region proposal networks},
      author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
      booktitle={Neural Information Processing Systems},
      pages={91--99},
      year={2015}
    }
    @article{ren2017faster,
      title={Faster {R-CNN}: Towards real-time object detection with region proposal networks},
      author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      volume={39},
      number={6},
      pages={1137--1149},
      year={2017},
      publisher={IEEE}
    }
    @article{dai16rfcn,
        Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
        Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
        Journal = {arXiv preprint arXiv:1605.06409},
        Year = {2016}
    }

