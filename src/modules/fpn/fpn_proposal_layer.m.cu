// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#include <cub/cub.cuh>
#include <iomanip>

#include "fpn_proposal_layer.m.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"  

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FPNProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  Forward_cpu(bottom, top);
  return ;
}

template <typename Dtype>
void FPNProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FPNProposalLayer);

} // namespace frcnn

} // namespace caffe
